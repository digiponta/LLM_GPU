# train_technical_intent_calibrator_v0119.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from technical_intent_calibrator_v0119 import (
    TECH_LABELS,
    TechnicalIntentCalibrator,
    extract_technical_logits,
    save_checkpoint,
)
from evaluate_generalization_v07 import CASES
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.11.9-technical-intent-calibrator.pt"

SEED = 42

ROWS = [
    ("大量の同種計算を並列処理する装置は何ですか。", "tech_gpu"),
    ("機械学習で大量の並列計算を行う演算装置を答えてください。", "tech_gpu"),
    ("多数のスレッドを同時実行するのが得意な装置は何ですか。", "tech_gpu"),
    ("行列演算を高速に並列実行するハードウェアは何ですか。", "tech_gpu"),
    ("汎用命令の実行とシステム制御を担当する装置は何ですか。", "tech_cpu"),
    ("コンピュータの中心で幅広い命令を処理する演算装置は何ですか。", "tech_cpu"),
    ("複雑な命令を順に実行する汎用プロセッサは何ですか。", "tech_cpu"),
    ("GPUへ処理を割り当てる側のプロセッサは何ですか。", "tech_cpu"),
    ("大量の文章を学習して文章を生成する大規模言語モデルは何ですか。", "tech_llm"),
    ("質問応答や文章生成に使う大規模言語モデルを何と呼びますか。", "tech_llm"),
    ("自然言語を学習して文章を生成するモデルの種類は何ですか。", "tech_llm"),
    ("テキストで事前学習された大規模な言語モデルは何ですか。", "tech_llm"),
    ("Self-Attentionを中心に使う代表的なモデル構造は何ですか。", "tech_transformer"),
    ("Attention機構を積み重ねるニューラルネット構造は何ですか。", "tech_transformer"),
    ("LLMの基盤として広く使われるAttention中心の構造は何ですか。", "tech_transformer"),
    ("EncoderとDecoderでAttentionを使う代表的な構造を答えてください。", "tech_transformer"),
    ("NVIDIA GPUを汎用計算に使う技術は何ですか。", "tech_cuda"),
    ("NVIDIA製GPU向けの並列計算プラットフォームは何ですか。", "tech_cuda"),
    ("GPUカーネルをNVIDIA環境で実行する仕組みは何ですか。", "tech_cuda"),
    ("NVIDIA GPU上でGPGPU計算を行う技術を答えてください。", "tech_cuda"),
    ("読みやすい文法で知られる汎用プログラミング言語は何ですか。", "tech_python"),
    ("機械学習やデータ分析でもよく使う高水準言語は何ですか。", "tech_python"),
    ("インデントを構文として使う代表的な言語は何ですか。", "tech_python"),
    ("初心者にも読みやすいことで知られるプログラミング言語は何ですか。", "tech_python"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--confidence-target", type=float, default=0.75)
    p.add_argument("--confidence-weight", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model, args.intent_head):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    benchmark = {str(case["prompt"]) for case in CASES}
    overlap = [p for p, _ in ROWS if p in benchmark]
    if overlap:
        raise RuntimeError("Exact DEV overlap: " + repr(overlap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    intent_head, intent_ck, intent_labels = load_intent_head(args.intent_head, model, device)

    for module in (model, intent_head):
        for p in module.parameters():
            p.requires_grad_(False)

    tech_logits_rows = []
    targets = []
    labels = []

    with torch.no_grad():
        for prompt, label in ROWS:
            ids = tokenizer.encode(f"人: {prompt}\nAI: ", add_bos=True)[-model.context_length:]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            hidden = model.forward_hidden(x)
            h = hidden[:, -1, :]
            intent_logits = intent_head(h)
            tech_logits = extract_technical_logits(intent_logits, intent_labels)[0]
            tech_logits_rows.append(tech_logits)
            targets.append(TECH_LABELS.index(label))
            labels.append(label)

    x_all = torch.stack(tech_logits_rows)
    y_all = torch.tensor(targets, dtype=torch.long, device=device)

    val_list = []
    for label in TECH_LABELS:
        idxs = [i for i, x in enumerate(labels) if x == label]
        val_list.append(idxs[-1])
    val_set = set(val_list)
    train_list = [i for i in range(len(labels)) if i not in val_set]

    train_idx = torch.tensor(train_list, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_list, dtype=torch.long, device=device)

    calibrator = TechnicalIntentCalibrator(
        num_labels=len(TECH_LABELS),
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=args.lr, weight_decay=0.01)

    def loss_for(indices):
        adjusted = calibrator(x_all[indices])
        target = y_all[indices]
        ce = F.cross_entropy(adjusted, target)
        probs = torch.softmax(adjusted, dim=-1)
        target_prob = probs[torch.arange(len(indices), device=device), target]
        conf = F.relu(args.confidence_target - target_prob).mean()
        total = ce + args.confidence_weight * conf
        acc = (adjusted.argmax(dim=-1) == target).float().mean()
        return total, ce, conf, acc, target_prob

    print("====================================================")
    print(" Technical Intent Repair / Calibration v0.11.9")
    print("====================================================")
    print("Device:", device)
    print("Technical labels:", ", ".join(TECH_LABELS))
    print("Rows:", len(labels), "Train:", len(train_idx), "Val:", len(val_idx))
    print("Exact DEV overlap:", len(overlap))
    print("Confidence target:", args.confidence_target)

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        calibrator.train()
        optimizer.zero_grad(set_to_none=True)
        loss, ce, conf, acc, target_prob = loss_for(train_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 1.0)
        optimizer.step()

        calibrator.eval()
        with torch.no_grad():
            val, vce, vconf, vacc, vprob = loss_for(val_idx)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"| train={loss.item():.4f} ce={ce.item():.4f} conf={conf.item():.4f} acc={acc.item():.3f} "
                f"| val={val.item():.4f} ce={vce.item():.4f} conf={vconf.item():.4f} acc={vacc.item():.3f} "
                f"| val_p={vprob.mean().item():.3f}"
            )

        value = float(val.item())
        if value < best - 1e-6:
            best = value
            best_epoch = epoch
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in calibrator.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    calibrator.load_state_dict(best_state)

    save_checkpoint(
        args.output,
        calibrator,
        intent_labels,
        epoch=best_epoch,
        loss=best,
        base_model=args.model,
        intent_head=args.intent_head,
        confidence_target=args.confidence_target,
        confidence_weight=args.confidence_weight,
    )

    print("Best epoch:", best_epoch)
    print("Best val loss:", f"{best:.6f}")
    print("Checkpoint:", args.output)


if __name__ == "__main__":
    main()
