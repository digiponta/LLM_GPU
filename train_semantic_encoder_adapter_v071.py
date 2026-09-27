# train_semantic_encoder_adapter_v071.py
#
# Semantic Encoder Adapter v0.7.1
# G05 Instruction-Computation Relation Refinement
#
# Exact G05 is excluded.
# Base encoder, semantic adapter and concept/attribute heads are frozen.
# Only the 13-axis hierarchy head is refined.

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v07 import (
    SEMANTIC_HIERARCHY_LABELS,
    load_semantic_adapter_v07_checkpoint,
)
from semantic_encoder_adapter_v071 import (
    save_semantic_adapter_v071_checkpoint,
)
from tokenizer_bpe import Tokenizer
from train_semantic_encoder_adapter_v01 import AdapterDataset
from train_semantic_encoder_adapter_v03 import build_rows


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INIT = "model/model-gpu-v0.9.1-semantic-adapter-v07.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-semantic-adapter-v071.pt"
SEED = 42


# G05-neighborhood rows. Exact G05 is intentionally absent.
RELATION_ROWS: List[str] = [
    "コンピュータの中心で幅広い命令を処理する装置は、計算だけでなく制御やメモリ操作も実行します。",
    "中核となるCPUは多様な命令を処理し、その中には計算命令と非計算命令があります。",
    "多種類の命令を扱う処理装置では、計算は命令実行全体の一部分です。",
    "幅広い命令の実行には、算術計算、分岐、ロード、ストアが含まれます。",
    "さまざまな命令を処理するとは、計算だけを行うことではありません。",
    "汎用CPUの命令実行は、計算、制御、メモリアクセス、データ移動を組み合わせます。",
    "異なる命令を処理するCPUでは、計算は命令列を構成する要素の一つです。",
    "プログラム全体の処理では、計算命令だけでなく分岐命令やメモリ命令も実行されます。",
    "中心的な処理装置が多様な命令を扱うとき、instruction execution は computation より広い概念です。",
    "CPUが扱う幅広い命令には、計算に加えて制御フローとデータ移動が含まれます。",
    "命令実行は計算を含みますが、計算だけには限定されません。",
    "多様な命令列を処理する装置では、instruction execution が computation を包含します。",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train v0.7.1 G05 instruction-computation relation refinement."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--init-adapter", default=DEFAULT_INIT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--relation-margin", type=float, default=0.15)
    p.add_argument("--relation-weight", type=float, default=2.0)
    p.add_argument("--preservation-weight", type=float, default=1.0)
    return p.parse_args()


@torch.no_grad()
def encode_prompt_hidden(model, tokenizer, prompt):
    device = next(model.parameters()).device
    text = f"人: {prompt}\nAI: "
    ids = tokenizer.encode(text, add_bos=True)[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model.forward_hidden(x)[:, -1, :][0].detach()


def exact_overlap():
    dev = {str(case["prompt"]) for case in CASES}
    return sorted(prompt for prompt in RELATION_ROWS if prompt in dev)


def relation_margin_loss(logits, margin):
    probs = torch.sigmoid(logits)
    instruction_i = SEMANTIC_HIERARCHY_LABELS.index("instruction_execution")
    computation_i = SEMANTIC_HIERARCHY_LABELS.index("computation")
    instruction = probs[:, instruction_i]
    computation = probs[:, computation_i]
    return F.relu(margin + computation - instruction).mean()


def child_parent_loss(logits):
    probs = torch.sigmoid(logits)
    idx = {
        name: SEMANTIC_HIERARCHY_LABELS.index(name)
        for name in SEMANTIC_HIERARCHY_LABELS
    }
    pairs = [
        ("computation", "instruction_execution"),
        ("arithmetic_logic", "computation"),
        ("repeated_computation", "computation"),
        ("control_flow", "instruction_execution"),
        ("memory_operation", "instruction_execution"),
        ("data_movement", "instruction_execution"),
        ("heterogeneous_instruction_stream", "instruction_execution"),
    ]
    losses = []
    for child, parent in pairs:
        losses.append(
            F.relu(
                probs[:, idx[child]]
                - probs[:, idx[parent]]
            ).mean()
        )
    return torch.stack(losses).mean()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model, args.init_adapter):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    overlaps = exact_overlap()
    if overlaps:
        raise RuntimeError(
            "Exact overlap with fixed DEV prompts: " + repr(overlaps)
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    base_model, base_checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    adapter, heads, hierarchy_head, init_checkpoint = (
        load_semantic_adapter_v07_checkpoint(args.init_adapter, device)
    )

    adapter.eval()
    heads.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    for p in heads.parameters():
        p.requires_grad_(False)

    relation_hidden = torch.stack(
        [
            encode_prompt_hidden(base_model, tokenizer, prompt)
            for prompt in RELATION_ROWS
        ],
        dim=0,
    ).to(device)

    base_rows = build_rows()
    base_set = AdapterDataset(base_rows, base_model, tokenizer)
    preserve_hidden = torch.stack(
        [item[0] for item in base_set.items],
        dim=0,
    ).to(device)

    with torch.no_grad():
        start_preserve_probs = torch.sigmoid(
            hierarchy_head(adapter(preserve_hidden))
        ).detach()

    params = list(hierarchy_head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=0.01,
    )

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.7.1 Training")
    print("====================================================")
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Initialization         :", args.init_adapter)
    print("Initial adapter loss   :", init_checkpoint.get("loss"))
    print("Base encoder           : frozen")
    print("Semantic adapter       : frozen")
    print("Concept/attr heads     : frozen")
    print("Hierarchy head         : trainable")
    print("Relation rows          :", len(RELATION_ROWS))
    print("Exact DEV overlap      :", len(overlaps))
    print("Learning rate          :", args.lr)
    print("Relation margin        :", args.relation_margin)
    print("Relation weight        :", args.relation_weight)
    print("Preservation weight    :", args.preservation_weight)
    print()

    best = float("inf")
    best_epoch = 0
    best_head = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        hierarchy_head.train()
        optimizer.zero_grad(set_to_none=True)

        logits = hierarchy_head(adapter(relation_hidden))
        direct_relation = relation_margin_loss(
            logits,
            args.relation_margin,
        )
        hierarchy_relation = child_parent_loss(logits)

        preserve_probs = torch.sigmoid(
            hierarchy_head(adapter(preserve_hidden))
        )
        preservation = F.mse_loss(
            preserve_probs,
            start_preserve_probs,
        )

        total = (
            args.relation_weight * direct_relation
            + 0.5 * hierarchy_relation
            + args.preservation_weight * preservation
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        value = float(total.item())
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| loss={value:.4f} "
            f"direct={direct_relation.item():.4f} "
            f"hier={hierarchy_relation.item():.4f} "
            f"pres={preservation.item():.5f}"
        )

        if value < best - 1e-5:
            best = value
            best_epoch = epoch
            bad = 0
            best_head = {
                k: v.detach().cpu().clone()
                for k, v in hierarchy_head.state_dict().items()
            }
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    hierarchy_head.load_state_dict(best_head)

    save_semantic_adapter_v071_checkpoint(
        args.output,
        adapter,
        heads,
        hierarchy_head,
        epoch=best_epoch,
        loss=best,
        base_model=args.model,
        init_adapter=args.init_adapter,
        learning_rate=args.lr,
        relation_margin=args.relation_margin,
        relation_weight=args.relation_weight,
        preservation_weight=args.preservation_weight,
    )

    print()
    print("Semantic Encoder Adapter v0.7.1 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best loss        :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
