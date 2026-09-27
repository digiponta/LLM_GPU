# train_semantic_encoder_adapter_v072.py
#
# Semantic Encoder Adapter v0.7.2
# Program Composition Hierarchy
#
# Core:
#   program_execution
#     -> instruction_sequence
#       -> instruction_execution
#         -> computation / control / memory / data movement

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v072 import (
    SEMANTIC_HIERARCHY_LABELS,
    initialize_from_v071,
    save_semantic_adapter_v072_checkpoint,
)
from tokenizer_bpe import Tokenizer
from train_semantic_encoder_adapter_v01 import AdapterDataset
from train_semantic_encoder_adapter_v03 import build_rows


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INIT = "model/model-gpu-v0.9.1-semantic-adapter-v071.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-semantic-adapter-v072.pt"
SEED = 42


PROGRAM_ROWS = [
    (
        "プログラムの実行は、複数の命令からなる命令列を順に処理することで進みます。",
        (1,1,1,1,1,0,1,1,1,1,0,1,1,0,0),
    ),
    (
        "プログラムは命令列として実行され、その命令には計算、分岐、メモリアクセス、データ移動が含まれます。",
        (1,1,1,1,1,1,1,1,1,1,0,1,1,0,0),
    ),
    (
        "プログラム実行は計算だけではなく、制御命令やメモリ命令を含む命令列全体の実行です。",
        (1,1,1,1,1,0,1,1,1,1,0,1,1,0,0),
    ),
    (
        "計算はプログラム実行を構成する命令処理の一部です。",
        (1,1,1,1,1,1,0,0,0,0,0,1,0,0,0),
    ),
    (
        "分岐はプログラム実行の一部ですが、数値計算そのものではありません。",
        (1,1,1,1,0,0,1,0,0,1,0,1,1,0,0),
    ),
    (
        "ロードとストアはプログラム実行を支えるメモリ操作命令です。",
        (1,1,1,1,0,0,0,1,1,1,0,1,1,0,0),
    ),
    (
        "CPUは多様な命令列を実行し、計算、分岐、ロード、ストアを組み合わせてプログラムを処理します。",
        (1,1,1,1,1,1,1,1,1,1,0,1,1,0,0),
    ),
    (
        "汎用CPUでは、プログラム処理は異種命令のシーケンスとして進みます。",
        (1,1,1,1,1,0,1,1,1,1,0,1,1,0,0),
    ),
    (
        "GPUもプログラム中の命令を実行しますが、同種計算を反復して並列処理する部分に強みがあります。",
        (1,1,1,1,1,1,0,0,1,0,1,0,0,1,1),
    ),
    (
        "GPUカーネルも命令列として実行され、その中で反復計算が多数のデータに適用されます。",
        (1,1,1,1,1,1,0,1,1,0,1,0,0,1,1),
    ),
    (
        "プログラム実行は命令列を含み、命令列は個々の命令実行から構成されます。",
        (1,1,1,1,0,0,0,0,0,1,0,1,0,0,0),
    ),
    (
        "instruction sequence は computation より広く、計算以外の命令も含みます。",
        (1,1,1,1,1,0,1,1,1,1,0,1,1,0,0),
    ),
]


def parse_args():
    p = argparse.ArgumentParser(description="Train v0.7.2 Program Composition Hierarchy.")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--init-adapter", default=DEFAULT_INIT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--hierarchy-weight", type=float, default=1.0)
    p.add_argument("--relation-weight", type=float, default=1.5)
    p.add_argument("--preservation-weight", type=float, default=1.0)
    return p.parse_args()


@torch.no_grad()
def encode_prompt_hidden(model, tokenizer, prompt):
    device = next(model.parameters()).device
    ids = tokenizer.encode(f"人: {prompt}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model.forward_hidden(x)[:, -1, :][0].detach()


def exact_overlap():
    dev = {str(case["prompt"]) for case in CASES}
    return sorted(prompt for prompt, _ in PROGRAM_ROWS if prompt in dev)


def relation_loss(logits):
    p = torch.sigmoid(logits)
    idx = {n: SEMANTIC_HIERARCHY_LABELS.index(n) for n in SEMANTIC_HIERARCHY_LABELS}
    pairs = [
        ("instruction_sequence", "program_execution"),
        ("instruction_execution", "instruction_sequence"),
        ("computation", "instruction_execution"),
        ("arithmetic_logic", "computation"),
        ("repeated_computation", "computation"),
        ("control_flow", "instruction_execution"),
        ("memory_operation", "instruction_execution"),
        ("data_movement", "instruction_execution"),
        ("heterogeneous_instruction_stream", "instruction_sequence"),
    ]
    losses = []
    for child, parent in pairs:
        losses.append(F.relu(p[:, idx[child]] - p[:, idx[parent]]).mean())

    # program/instruction sequence should remain broader than computation.
    losses.append(
        F.relu(
            0.10 + p[:, idx["computation"]] - p[:, idx["instruction_sequence"]]
        ).mean()
    )
    losses.append(
        F.relu(
            0.10 + p[:, idx["computation"]] - p[:, idx["program_execution"]]
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
        raise RuntimeError("Exact overlap with fixed DEV prompts: " + repr(overlaps))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    base_model, base_checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    adapter, heads, hierarchy_head, init_checkpoint = initialize_from_v071(
        args.init_adapter, device
    )
    adapter.eval()
    heads.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    for p in heads.parameters():
        p.requires_grad_(False)

    program_hidden = torch.stack(
        [encode_prompt_hidden(base_model, tokenizer, prompt) for prompt, _ in PROGRAM_ROWS],
        dim=0,
    ).to(device)
    targets = torch.tensor(
        [target for _prompt, target in PROGRAM_ROWS],
        dtype=torch.float32,
        device=device,
    )

    base_rows = build_rows()
    base_set = AdapterDataset(base_rows, base_model, tokenizer)
    preserve_hidden = torch.stack([item[0] for item in base_set.items], dim=0).to(device)

    # Preserve old labels only. New program labels are learned freely.
    old_label_indices = [
        SEMANTIC_HIERARCHY_LABELS.index(name)
        for name in SEMANTIC_HIERARCHY_LABELS
        if name not in ("program_execution", "instruction_sequence")
    ]
    with torch.no_grad():
        start_preserve = torch.sigmoid(
            hierarchy_head(adapter(preserve_hidden))
        )[:, old_label_indices].detach()

    params = list(hierarchy_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.7.2 Training")
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
    print("Hierarchy labels       :", ", ".join(SEMANTIC_HIERARCHY_LABELS))
    print("Program rows           :", len(PROGRAM_ROWS))
    print("Exact DEV overlap      :", len(overlaps))
    print("Learning rate          :", args.lr)
    print("Hierarchy weight       :", args.hierarchy_weight)
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

        logits = hierarchy_head(adapter(program_hidden))
        hierarchy = F.binary_cross_entropy_with_logits(logits, targets)
        relation = relation_loss(logits)

        preserve_probs = torch.sigmoid(
            hierarchy_head(adapter(preserve_hidden))
        )[:, old_label_indices]
        preservation = F.mse_loss(preserve_probs, start_preserve)

        total = (
            args.hierarchy_weight * hierarchy
            + args.relation_weight * relation
            + args.preservation_weight * preservation
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        value = float(total.item())
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| loss={value:.4f} hier={hierarchy.item():.4f} "
            f"relation={relation.item():.4f} pres={preservation.item():.5f}"
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

    save_semantic_adapter_v072_checkpoint(
        args.output,
        adapter,
        heads,
        hierarchy_head,
        epoch=best_epoch,
        loss=best,
        base_model=args.model,
        init_adapter=args.init_adapter,
        learning_rate=args.lr,
        hierarchy_weight=args.hierarchy_weight,
        relation_weight=args.relation_weight,
        preservation_weight=args.preservation_weight,
    )

    print()
    print("Semantic Encoder Adapter v0.7.2 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best loss        :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
