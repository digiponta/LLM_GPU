# train_cpu_gpu_local_margin_v0123.py
#
# v0.12.3 CPU-GPU Local Margin Refinement
#
# Objective on local prompts uses FINAL logits:
#   CPU rows: final(CPU) >= final(GPU) + margin
#   GPU rows: final(GPU) >= final(CPU) + margin
#
# Exact fixed G05 is excluded. v0.12.2 gate is frozen.
# Replay prompts preserve v0.12.2 effective direct-bias behavior.

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from cpu_gpu_local_margin_v0123 import save_checkpoint
from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from entity_contrastive_logit_alignment_v0122 import load_checkpoint as load_v0122
from evaluate_partial_intent_v09 import CASES
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
from semantic_generation_integration_v09 import (
    forward_semantic_conditioned,
    infer_semantic_condition,
)
from semantic_lexical_logit_alignment_v012 import load_frozen_v011
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_NAME_BINDING = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
DEFAULT_V011 = "model/model-gpu-v0.9.1-lexical-generation-v011.pt"
DEFAULT_V0122 = "model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-cpu-gpu-local-margin-v0123.pt"
SEED = 42


LOCAL_ROWS = [
    # CPU neighborhood: exact G05 is intentionally absent.
    ("コンピュータの中央で異なる種類の命令を幅広く実行するプロセッサは何ですか。", "cpu"),
    ("演算だけでなく分岐やメモリ操作も含む多様な命令を扱う中央装置は何ですか。", "cpu"),
    ("制御・演算・データ移動など異種命令を処理する汎用プロセッサを答えてください。", "cpu"),
    ("コンピュータ全体を制御しながら様々な命令を順に実行する装置は何ですか。", "cpu"),
    ("中央で汎用的な命令実行を担当する処理装置の名称は何ですか。", "cpu"),
    ("分岐、ロード、ストア、算術命令を組み合わせて実行する装置は何ですか。", "cpu"),
    ("多様な命令列の実行と制御を主に担当するプロセッサは何ですか。", "cpu"),
    ("汎用処理と制御処理を担当する中央プロセッサを英字で答えてください。", "cpu"),

    # GPU contrast neighborhood.
    ("多数の同じ計算を大量データへ並列適用するプロセッサは何ですか。", "gpu"),
    ("スループット重視で同種演算を並列実行する装置は何ですか。", "gpu"),
    ("画像処理や行列計算のようなデータ並列処理を得意とする装置は何ですか。", "gpu"),
    ("同じ演算を多数の要素へ同時に適用するプロセッサを答えてください。", "gpu"),
    ("大量の並列計算を効率よく処理する演算装置の名称は何ですか。", "gpu"),
    ("多数の計算スレッドを並列に走らせるのが得意なプロセッサは何ですか。", "gpu"),
]


PRESERVATION_ROWS = [
    # Other technical entities.
    "文章を学習して生成する大規模言語モデルの略称は何ですか。",
    "Self-Attentionを主要機構とする代表的構造名は何ですか。",
    "NVIDIA GPU向けの汎用計算技術名は何ですか。",
    "読みやすさで知られる汎用プログラミング言語名は何ですか.",
    # Non-entity/control behavior.
    "コードのエラー原因を調べたいです。",
    "二つの実験を公平に比較したいです。",
    "別の話題へ移りたいです。",
    "もう一度説明してください。",
    "少し疲れました。",
    "助かりました、ありがとう。",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--name-binding", default=DEFAULT_NAME_BINDING)
    p.add_argument("--v011-checkpoint", default=DEFAULT_V011)
    p.add_argument("--source-checkpoint", default=DEFAULT_V0122)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--classification-weight", type=float, default=0.25)
    p.add_argument("--preservation-weight", type=float, default=0.20)
    p.add_argument("--patience", type=int, default=25)
    return p.parse_args()


def exact_dev_overlap():
    dev = {str(case["prompt"]) for case in CASES}
    rows = [p for p, _label in LOCAL_ROWS] + list(PRESERVATION_ROWS)
    return sorted(p for p in rows if p in dev)


@torch.no_grad()
def encode_state(
    text,
    tokenizer,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    generation_model,
    v011_projection,
):
    device = next(generation_model.parameters()).device
    ids = tokenizer.encode(f"人: {text}\nAI: ", add_bos=True)
    ids = ids[-generation_model.context_length:]
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
        generation_model,
        x,
        semantic_bias,
        v011_projection.inject_after,
    )
    base_logits = generation_model.lm_head(hidden[:, -1, :])
    return condition[0], base_logits[0]


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
        args.source_checkpoint,
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
    generation_model, v011_projection, v011_checkpoint = load_frozen_v011(
        args.v011_checkpoint,
        device,
    )
    adapter, gate, source_checkpoint = load_v0122(
        args.source_checkpoint,
        device,
    )

    # Freeze gate; refine only the direct logit adapter.
    gate.eval()
    for p in gate.parameters():
        p.requires_grad_(False)
    adapter.train()

    cpu_ids = tokenizer.encode("CPU")
    gpu_ids = tokenizer.encode("GPU")
    if not cpu_ids or not gpu_ids:
        raise RuntimeError("CPU/GPU tokenization failed.")
    cpu_id = int(cpu_ids[0])
    gpu_id = int(gpu_ids[0])

    conditions = []
    base_rows = []
    labels = []
    for prompt, label in LOCAL_ROWS:
        c, b = encode_state(
            prompt,
            tokenizer,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            generation_model,
            v011_projection,
        )
        conditions.append(c)
        base_rows.append(b)
        labels.append(label)
    conditions = torch.stack(conditions)
    base_rows = torch.stack(base_rows)

    preserve_conditions = []
    for prompt in PRESERVATION_ROWS:
        c, _b = encode_state(
            prompt,
            tokenizer,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            generation_model,
            v011_projection,
        )
        preserve_conditions.append(c)
    preserve_conditions = torch.stack(preserve_conditions)

    # Snapshot v0.12.2 effective bias for replay/preservation.
    adapter.eval()
    with torch.no_grad():
        preserve_gate = gate(preserve_conditions)
        preserve_target = (
            preserve_gate.unsqueeze(-1) * adapter(preserve_conditions)
        ).detach()

    # Deterministic split: final CPU row and final GPU row are validation.
    cpu_idx = [i for i, x in enumerate(labels) if x == "cpu"]
    gpu_idx = [i for i, x in enumerate(labels) if x == "gpu"]
    val_list = [cpu_idx[-1], gpu_idx[-1]]
    train_list = cpu_idx[:-1] + gpu_idx[:-1]
    train_idx = torch.tensor(train_list, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_list, dtype=torch.long, device=device)

    target_is_cpu = torch.tensor(
        [1 if label == "cpu" else 0 for label in labels],
        dtype=torch.bool,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    print()
    print("====================================================")
    print(" CPU-GPU Local Margin Refinement v0.12.3")
    print("====================================================")
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Semantic adapter loss  :", semantic_checkpoint.get("loss"))
    print("Name binding loss      :", name_checkpoint.get("loss"))
    print("v0.11 integration loss :", v011_checkpoint.get("loss"))
    print("v0.12.2 source loss    :", source_checkpoint.get("loss"))
    print("Exact DEV overlap      :", len(overlap))
    print("Local CPU rows         :", len(cpu_idx))
    print("Local GPU rows         :", len(gpu_idx))
    print("Train local rows       :", len(train_idx))
    print("Validation local rows  :", len(val_idx))
    print("Preservation rows      :", len(PRESERVATION_ROWS))
    print("Gate                   : frozen from v0.12.2")
    print("Adapter                : trainable")
    print("Target final margin    :", args.margin)
    print("Preservation weight    :", args.preservation_weight)
    print("Classification weight  :", args.classification_weight)
    print("Learning rate          :", args.lr)
    print()

    def losses(indices):
        c = conditions[indices]
        base = base_rows[indices]
        gate_prob = gate(c)
        bias = adapter(c)
        final = base + gate_prob.unsqueeze(-1) * bias

        cpu = final[:, cpu_id]
        gpu = final[:, gpu_id]
        cpu_mask = target_is_cpu[indices]

        signed_margin = torch.where(cpu_mask, cpu - gpu, gpu - cpu)
        margin_loss = F.relu(args.margin - signed_margin).mean()

        pair_scores = torch.stack([gpu, cpu], dim=1)
        pair_target = cpu_mask.long()
        classify = F.cross_entropy(pair_scores, pair_target)

        current_preserve = (
            preserve_gate.unsqueeze(-1) * adapter(preserve_conditions)
        )
        preserve = F.mse_loss(current_preserve, preserve_target)

        total = (
            margin_loss
            + args.classification_weight * classify
            + args.preservation_weight * preserve
        )
        return total, margin_loss, classify, preserve, signed_margin

    best = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        loss, margin_loss, cls, preserve, signed = losses(train_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        adapter.eval()
        with torch.no_grad():
            val, vmargin, vcls, vpreserve, vsigned = losses(val_idx)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"| train={loss.item():.4f} margin={margin_loss.item():.4f} "
                f"cls={cls.item():.4f} preserve={preserve.item():.6f} "
                f"mean_signed={signed.mean().item():+.4f} "
                f"| val={val.item():.4f} margin={vmargin.item():.4f} "
                f"mean_signed={vsigned.mean().item():+.4f}"
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
        gate,
        epoch=best_epoch,
        loss=best,
        source_checkpoint=args.source_checkpoint,
        learning_rate=args.lr,
        margin=args.margin,
        preservation_weight=args.preservation_weight,
        classification_weight=args.classification_weight,
        cpu_token_id=cpu_id,
        gpu_token_id=gpu_id,
    )

    print()
    print("CPU-GPU Local Margin Refinement v0.12.3 completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
