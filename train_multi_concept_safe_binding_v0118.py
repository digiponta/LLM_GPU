# train_multi_concept_safe_binding_v0118.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from multi_concept_safe_binding_v0118 import (
    CONCEPTS,
    CANONICAL,
    MultiConceptSafeBoost,
    save_checkpoint,
)
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from evaluate_generalization_v07 import CASES
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_ROLE = "model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.11.8-multi-concept-safe-binding.pt"

SEED = 42

TRAIN_ROWS = [
    ("大量の同種計算を並列実行する演算装置は何ですか。", "tech_gpu"),
    ("機械学習の行列演算を高速化する並列演算装置は何ですか。", "tech_gpu"),
    ("CPUと協調して並列計算を担当する装置は何ですか。", "tech_gpu"),
    ("多数のスレッドを同時に実行する演算装置を答えてください。", "tech_gpu"),
    ("汎用命令を実行してシステム全体を制御する装置は何ですか。", "tech_cpu"),
    ("複雑で多様な命令を順に処理する中心的な演算装置は何ですか。", "tech_cpu"),
    ("GPUへ処理を割り当てる側のプロセッサは何ですか。", "tech_cpu"),
    ("コンピュータの汎用処理と制御を担当する装置は何ですか。", "tech_cpu"),
    ("大量の文章から言語パターンを学習する大規模言語モデルは何ですか。", "tech_llm"),
    ("文章生成や質問応答に使われる大規模な言語モデルを何と呼びますか。", "tech_llm"),
    ("自然言語を学習して文章を生成するモデルの種類は何ですか。", "tech_llm"),
    ("言語タスク向けに大量テキストで学習されたモデルは何ですか。", "tech_llm"),
    ("Self-Attentionを中心に構成されたニューラルネットワーク構造は何ですか。", "tech_transformer"),
    ("Attention機構を積み重ねて系列を処理する代表的なモデル構造は何ですか。", "tech_transformer"),
    ("LLMで広く使われるAttention中心のアーキテクチャは何ですか。", "tech_transformer"),
    ("EncoderやDecoderにAttentionを使う代表的な構造を答えてください。", "tech_transformer"),
    ("NVIDIA GPUを汎用計算に使うためのプラットフォームは何ですか。", "tech_cuda"),
    ("NVIDIA製GPU向けの並列計算技術を何と呼びますか。", "tech_cuda"),
    ("GPUカーネルをNVIDIA環境で実行するための技術は何ですか。", "tech_cuda"),
    ("NVIDIA GPU上でGPGPU計算を行う仕組みは何ですか。", "tech_cuda"),
    ("読みやすい文法で広く使われる汎用プログラミング言語は何ですか。", "tech_python"),
    ("機械学習やデータ分析でも使われる高水準言語は何ですか。", "tech_python"),
    ("インデントを構文に使う代表的なプログラミング言語は何ですか。", "tech_python"),
    ("初心者にも読みやすいことで知られる汎用言語は何ですか。", "tech_python"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--role-checkpoint", default=DEFAULT_ROLE)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=360)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--max-boost", type=float, default=12.0)
    p.add_argument("--initial-boost", type=float, default=0.20)
    p.add_argument("--target-margin", type=float, default=0.5)
    p.add_argument("--l2-weight", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model, args.intent_head, args.role_checkpoint):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    dev = {str(case["prompt"]) for case in CASES}
    overlap = [prompt for prompt, _ in TRAIN_ROWS if prompt in dev]
    if overlap:
        raise RuntimeError("Exact DEV overlap: " + repr(overlap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ck = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()

    intent_head, intent_ck, intent_labels = load_intent_head(args.intent_head, model, device)
    role_head, _old_binding, role_ck = load_role_checkpoint(args.role_checkpoint, intent_labels, device)
    role_head.eval()

    for module in (model, intent_head, role_head):
        for p in module.parameters():
            p.requires_grad_(False)

    missing = [label for label in CONCEPTS if label not in intent_labels]
    if missing:
        raise ValueError("Missing technical intent labels: " + repr(missing))

    token_ids = {}
    for label in CONCEPTS:
        ids = tokenizer.encode(CANONICAL[label])
        if not ids:
            raise RuntimeError("Empty tokenization for " + CANONICAL[label])
        token_ids[label] = int(ids[0])

    features = []
    base_logits_rows = []
    target_ids = []
    concept_indices = []
    labels = []

    with torch.no_grad():
        for prompt, label in TRAIN_ROWS:
            ids = tokenizer.encode(f"人: {prompt}\nAI: ", add_bos=True)[-model.context_length:]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            hidden = model.forward_hidden(x)
            h = hidden[:, -1, :]
            ip = torch.sigmoid(intent_head(h))
            rp = torch.sigmoid(role_head(h))
            logits = model.lm_head(h)[0]

            features.append(torch.cat([ip[0], rp[0]], dim=-1))
            base_logits_rows.append(logits)
            target_ids.append(token_ids[label])
            concept_indices.append(CONCEPTS.index(label))
            labels.append(label)

    features = torch.stack(features)
    base_logits_rows = torch.stack(base_logits_rows)
    target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)
    concept_indices = torch.tensor(concept_indices, dtype=torch.long, device=device)

    val_list = []
    for label in CONCEPTS:
        idxs = [i for i, x in enumerate(labels) if x == label]
        val_list.append(idxs[-1])
    val_set = set(val_list)
    train_list = [i for i in range(len(labels)) if i not in val_set]

    train_idx = torch.tensor(train_list, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_list, dtype=torch.long, device=device)

    adapter = MultiConceptSafeBoost(
        feature_dim=features.size(1),
        num_concepts=len(CONCEPTS),
        hidden_dim=args.hidden_dim,
        max_boost=args.max_boost,
        initial_boost=args.initial_boost,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)

    def compute(indices):
        feats = features[indices]
        base = base_logits_rows[indices]
        tids = target_ids[indices]
        cids = concept_indices[indices]

        boost = adapter(feats, cids)
        corrected = base.clone()
        corrected[torch.arange(len(indices), device=device), tids] += boost

        target = corrected[torch.arange(len(indices), device=device), tids]
        masked = corrected.clone()
        masked[torch.arange(len(indices), device=device), tids] = -torch.inf
        competitor, competitor_ids = masked.max(dim=1)

        hinge = F.relu(args.target_margin - (target - competitor)).mean()
        reg = boost.pow(2).mean()
        total = hinge + args.l2_weight * reg
        return total, hinge, reg, boost

    with torch.no_grad():
        initial = float(adapter(features[:1], concept_indices[:1])[0].item())

    print("====================================================")
    print(" Multi-Concept Safe Binding v0.11.8")
    print("====================================================")
    print("Device:", device)
    print("Concepts:", ", ".join(CONCEPTS))
    print("Rows:", len(labels), "Train:", len(train_idx), "Val:", len(val_idx))
    print("Exact DEV overlap:", len(overlap))
    print("Target full-vocab margin:", args.target_margin)
    print("Configured initial boost:", args.initial_boost)
    print("Actual initial boost:", f"{initial:.6f}")
    for label in CONCEPTS:
        print(
            f"{label:16s} -> {CANONICAL[label]:11s} "
            f"token={token_ids[label]} "
            f"piece={tokenizer.decode([token_ids[label]], skip_special_tokens=False)!r}"
        )

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        loss, hinge, reg, boost = compute(train_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        adapter.eval()
        with torch.no_grad():
            val, vhinge, vreg, vboost = compute(val_idx)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"| train={loss.item():.4f} hinge={hinge.item():.4f} "
                f"| val={val.item():.4f} hinge={vhinge.item():.4f} "
                f"| boost={boost.mean().item():.3f} val_boost={vboost.mean().item():.3f}"
            )

        value = float(val.item())
        if value < best - 1e-6:
            best = value
            best_epoch = epoch
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    adapter.load_state_dict(best_state)

    save_checkpoint(
        args.output,
        adapter,
        intent_labels,
        args.role_checkpoint,
        token_ids,
        epoch=best_epoch,
        loss=best,
        base_model=args.model,
        intent_head=args.intent_head,
        target_margin=args.target_margin,
        technical_confidence_threshold=0.70,
        cpu_gpu_intent_margin=0.10,
        cpu_gpu_role_margin=0.05,
        learning_rate=args.lr,
    )

    print("Best epoch:", best_epoch)
    print("Best val loss:", f"{best:.6f}")
    print("Checkpoint:", args.output)


if __name__ == "__main__":
    main()
