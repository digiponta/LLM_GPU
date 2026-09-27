# train_entity_target_logit_alignment_v0121.py
#
# v0.12.1 Entity-Target Logit Alignment
#
# Train only the direct logit adapter using explicit entity-name targets.
# Exact fixed 30-case prompts, including G05, are excluded.

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from entity_target_logit_alignment_v0121 import (
    ENTITY_TARGETS,
    save_checkpoint,
)
from evaluate_partial_intent_v09 import CASES
from semantic_encoder_adapter_v01 import ATTRIBUTE_LABELS, CONCEPT_LABELS
from semantic_encoder_adapter_v08 import SEMANTIC_HIERARCHY_LABELS
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
from semantic_generation_integration_v09 import (
    forward_semantic_conditioned,
    infer_semantic_condition,
)
from semantic_lexical_logit_alignment_v012 import (
    SemanticLexicalLogitAdapter,
    load_frozen_v011,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_NAME_BINDING = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
DEFAULT_V011 = "model/model-gpu-v0.9.1-lexical-generation-v011.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-entity-target-logit-v0121.pt"
SEED = 42


ENTITY_ROWS = [
    # GPU
    ("大量の同種計算を並列実行する演算装置の名称を答えてください。", "tech_gpu"),
    ("画像処理や並列計算を得意とするプロセッサは何ですか。", "tech_gpu"),
    ("スループット重視のデータ並列プロセッサの名前は何ですか。", "tech_gpu"),
    ("多数のデータ要素へ同じ計算を適用する装置を一語で答えてください。", "tech_gpu"),

    # CPU -- exact G05 deliberately excluded
    ("汎用命令と制御処理を担う中央のプロセッサ名は何ですか。", "tech_cpu"),
    ("分岐やメモリ操作を含む多様な命令を実行する処理装置の名称は何ですか。", "tech_cpu"),
    ("Central Processing Unitの略称を答えてください。", "tech_cpu"),
    ("中央処理装置を英字三文字で答えてください。", "tech_cpu"),
    ("中央演算処理装置の一般的な略称は何ですか。", "tech_cpu"),

    # LLM
    ("大規模言語モデルの英字略称を答えてください。", "tech_llm"),
    ("大量の文章を学習して文章生成を行うモデルの略称は何ですか。", "tech_llm"),
    ("Large Language Modelを三文字で表すと何ですか。", "tech_llm"),

    # Transformer
    ("Self-Attentionを中心に構成されるモデル構造の名前は何ですか。", "tech_transformer"),
    ("Attentionを主要機構として使うニューラルネット構造を答えてください。", "tech_transformer"),
    ("LLMで広く使われるAttentionベースの構造名は何ですか。", "tech_transformer"),

    # CUDA
    ("NVIDIA GPU向け汎用計算プラットフォームの名称は何ですか。", "tech_cuda"),
    ("NVIDIAのGPU計算基盤を四文字で答えてください。", "tech_cuda"),
    ("GPUを汎用計算へ利用するNVIDIAの技術名は何ですか。", "tech_cuda"),

    # Python
    ("読みやすい文法で知られる汎用プログラミング言語名は何ですか。", "tech_python"),
    ("機械学習でも広く使われる汎用スクリプト言語を答えてください。", "tech_python"),
    ("インデントを構文に使う代表的な言語名は何ですか。", "tech_python"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--name-binding", default=DEFAULT_NAME_BINDING)
    p.add_argument("--v011-checkpoint", default=DEFAULT_V011)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.30)
    p.add_argument("--l2-weight", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    return p.parse_args()


def exact_dev_overlap():
    dev = {str(case["prompt"]) for case in CASES}
    return sorted(prompt for prompt, _label in ENTITY_ROWS if prompt in dev)


@torch.no_grad()
def encode_prompt(
    text,
    tokenizer,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    v011_projection,
):
    device = next(semantic_model.parameters()).device
    ids = tokenizer.encode(f"人: {text}\nAI: ", add_bos=True)
    ids = ids[-semantic_model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_index = torch.tensor([x.size(1)-1], dtype=torch.long, device=device)

    adapted, concept_prob, attribute_prob, hierarchy_prob = infer_semantic_condition(
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        x,
        prompt_index,
    )
    lexical_identity = name_binding(adapted)
    condition = torch.cat(
        [
            adapted,
            concept_prob,
            attribute_prob,
            hierarchy_prob,
            lexical_identity,
        ],
        dim=-1,
    )

    semantic_bias = v011_projection(
        adapted,
        concept_prob,
        attribute_prob,
        hierarchy_prob,
        lexical_identity,
    )
    hidden = forward_semantic_conditioned(
        semantic_model if False else None,
        x,
        semantic_bias,
        v011_projection.inject_after,
    )
    return x, condition, semantic_bias


@torch.no_grad()
def frozen_first_logits(generation_model, v011_projection, x, semantic_bias):
    hidden = forward_semantic_conditioned(
        generation_model,
        x,
        semantic_bias,
        v011_projection.inject_after,
    )
    return generation_model.lm_head(hidden[:, -1, :])


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (
        args.tokenizer,
        args.base_model,
        args.semantic_adapter,
        args.name_binding,
        args.v011_checkpoint,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    overlap = exact_dev_overlap()
    if overlap:
        raise RuntimeError("Exact fixed-DEV overlap: " + repr(overlap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)

    (
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        base_checkpoint,
        semantic_checkpoint,
    ) = load_frozen_semantic_path_v08(
        args.base_model,
        args.semantic_adapter,
        device,
    )
    name_binding, name_checkpoint = load_cpu_name_binding_checkpoint(
        args.name_binding,
        device,
    )
    name_binding.eval()
    for p in name_binding.parameters():
        p.requires_grad_(False)

    generation_model, v011_projection, v011_checkpoint = load_frozen_v011(
        args.v011_checkpoint,
        device,
    )

    condition_dim = v011_projection.input_dim
    adapter = SemanticLexicalLogitAdapter(
        input_dim=condition_dim,
        vocab_size=generation_model.vocab_size,
        rank=args.rank,
        beta=args.beta,
    ).to(device)

    target_token_ids = {}
    target_token_sequences = {}
    for label, entity in ENTITY_TARGETS.items():
        ids = tokenizer.encode(entity)
        if not ids:
            raise RuntimeError(f"Tokenizer produced no tokens for {entity}")
        target_token_ids[label] = int(ids[0])
        target_token_sequences[label] = list(map(int, ids))

    conditions = []
    base_logits_rows = []
    targets = []

    for prompt, label in ENTITY_ROWS:
        ids = tokenizer.encode(f"人: {prompt}\nAI: ", add_bos=True)
        ids = ids[-generation_model.context_length:]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        prompt_index = torch.tensor([x.size(1)-1], dtype=torch.long, device=device)

        with torch.no_grad():
            adapted, concept_prob, attribute_prob, hierarchy_prob = infer_semantic_condition(
                semantic_model,
                semantic_adapter,
                semantic_heads,
                hierarchy_head,
                x,
                prompt_index,
            )
            lexical_identity = name_binding(adapted)
            condition = torch.cat(
                [
                    adapted,
                    concept_prob,
                    attribute_prob,
                    hierarchy_prob,
                    lexical_identity,
                ],
                dim=-1,
            )
            semantic_bias = v011_projection(
                adapted,
                concept_prob,
                attribute_prob,
                hierarchy_prob,
                lexical_identity,
            )
            hidden = forward_semantic_conditioned(
                generation_model,
                x,
                semantic_bias,
                v011_projection.inject_after,
            )
            base_logits = generation_model.lm_head(hidden[:, -1, :])

        conditions.append(condition[0])
        base_logits_rows.append(base_logits[0])
        targets.append(target_token_ids[label])

    conditions = torch.stack(conditions, dim=0)
    base_logits_rows = torch.stack(base_logits_rows, dim=0)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    # Deterministic stratified split: last example per class is validation.
    train_idx = []
    val_idx = []
    by_label = {}
    for i, (_prompt, label) in enumerate(ENTITY_ROWS):
        by_label.setdefault(label, []).append(i)
    for label, idxs in by_label.items():
        val_idx.append(idxs[-1])
        train_idx.extend(idxs[:-1])

    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)

    print()
    print("====================================================")
    print(" Entity-Target Logit Alignment v0.12.1")
    print("====================================================")
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Semantic adapter loss  :", semantic_checkpoint.get("loss"))
    print("Name binding loss      :", name_checkpoint.get("loss"))
    print("v0.11 integration loss :", v011_checkpoint.get("loss"))
    print("Exact DEV overlap      :", len(overlap))
    print("Entity rows            :", len(ENTITY_ROWS))
    print("Train rows             :", len(train_idx))
    print("Validation rows        :", len(val_idx))
    print("Condition dim          :", condition_dim)
    print("Logit adapter          :", f"{condition_dim} -> {args.rank} -> {generation_model.vocab_size}")
    print("Beta                   :", args.beta)
    print("LM/v0.11/semantic      : frozen")
    print()
    print("Entity target tokenization")
    print("--------------------------")
    for label, entity in ENTITY_TARGETS.items():
        ids = target_token_sequences[label]
        pieces = [tokenizer.decode([i], skip_special_tokens=True) for i in ids]
        print(f"{label:16s} -> {entity:12s} ids={ids} pieces={pieces}")
    print()

    def loss_for(indices):
        bias = adapter(conditions[indices])
        combined = base_logits_rows[indices] + bias
        ce = F.cross_entropy(combined, targets[indices])
        reg = bias.pow(2).mean()
        return ce + args.l2_weight * reg, ce, reg

    best = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        loss, ce, reg = loss_for(train_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        adapter.eval()
        with torch.no_grad():
            val, val_ce, val_reg = loss_for(val_idx)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"| train={loss.item():.4f} ce={ce.item():.4f} reg={reg.item():.6f} "
                f"| val={val.item():.4f} ce={val_ce.item():.4f} reg={val_reg.item():.6f}"
            )

        value = float(val.item())
        if value < best - 1e-6:
            best = value
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in adapter.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    adapter.load_state_dict(best_state)

    save_checkpoint(
        args.output,
        adapter,
        epoch=best_epoch,
        loss=best,
        v011_checkpoint=args.v011_checkpoint,
        semantic_adapter_checkpoint=args.semantic_adapter,
        name_binding_checkpoint=args.name_binding,
        learning_rate=args.lr,
        l2_weight=args.l2_weight,
        target_token_ids=target_token_ids,
    )

    print()
    print("Entity-Target Logit Alignment v0.12.1 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
