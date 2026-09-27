# train_semantic_encoder_adapter_v07.py
#
# Semantic Encoder Adapter v0.7:
# Instruction-Computation Hierarchy
#
# The adapter remains frozen. The new hierarchy head learns that:
#   computation ⊂ instruction_execution
# and that instruction execution also includes control, memory and data movement.

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v07 import (
    SEMANTIC_HIERARCHY_LABELS,
    initialize_from_v062,
    save_semantic_adapter_v07_checkpoint,
)
from tokenizer_bpe import Tokenizer
from train_semantic_encoder_adapter_v01 import AdapterDataset
from train_semantic_encoder_adapter_v03 import build_rows


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INIT = "model/model-gpu-v0.9.1-semantic-adapter-v062.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-semantic-adapter-v07.pt"
SEED = 42

# label order:
# processor, instruction_execution, computation, arithmetic_logic,
# control_flow, memory_operation, data_movement,
# heterogeneous_instruction_stream, repeated_computation,
# general_purpose, control_oriented, throughput_oriented, data_parallel
HIERARCHY_ROWS: List[Tuple[str, Tuple[int, ...]]] = [
    # Relation-defining statements.
    ("計算は命令実行の一部であり、命令実行そのものと同義ではありません。",
     (1,1,1,0,0,0,0,0,0,0,0,0,0)),
    ("命令実行には計算だけでなく、分岐、メモリ操作、データ移動も含まれます。",
     (1,1,1,0,1,1,1,1,0,1,1,0,0)),
    ("算術演算は命令として実行される計算処理の一種です。",
     (1,1,1,1,0,0,0,0,0,0,0,0,0)),
    ("分岐やジャンプは命令実行ですが、数値計算そのものではありません。",
     (1,1,0,0,1,0,0,1,0,1,1,0,0)),
    ("ロードとストアはメモリを扱う命令であり、算術計算とは別の役割です。",
     (1,1,0,0,0,1,1,1,0,1,1,0,0)),
    ("レジスタやメモリ間のデータ移動も命令実行の一部です。",
     (1,1,0,0,0,1,1,1,0,1,1,0,0)),

    # CPU-oriented instruction streams.
    ("CPUは算術、分岐、ロード、ストアなど異なる種類の命令を組み合わせて実行します。",
     (1,1,1,1,1,1,1,1,0,1,1,0,0)),
    ("コンピュータの中心で多様な命令列を処理するCPUは汎用的で制御にも向きます。",
     (1,1,1,1,1,1,1,1,0,1,1,0,0)),
    ("CPUは計算命令だけでなく、制御命令やメモリ命令も含む異種命令列を扱います。",
     (1,1,1,1,1,1,1,1,0,1,1,0,0)),
    ("CPUではプログラム処理が複数種類の命令の組み合わせとして進みます。",
     (1,1,1,1,1,1,1,1,0,1,1,0,0)),

    # GPU-oriented computation inside instruction execution.
    ("GPUも命令を実行しますが、同種の計算を多数のデータへ並列適用することに強みがあります。",
     (1,1,1,1,0,1,1,0,1,0,0,1,1)),
    ("GPUは命令実行の中でも反復される計算処理を高スループットで並列化します。",
     (1,1,1,1,0,1,1,0,1,0,0,1,1)),
    ("同じ計算を大量のデータ要素へ適用するdata-parallel処理はGPU向きです。",
     (1,1,1,1,0,0,1,0,1,0,0,1,1)),
    ("GPUの並列計算も命令実行の一形態であり、命令と計算を同一視する必要はありません。",
     (1,1,1,1,0,0,0,0,1,0,0,1,1)),

    # Program-level composition.
    ("プログラムは、計算、分岐、メモリアクセス、データ移動など複数の命令を組み合わせて処理されます。",
     (1,1,1,1,1,1,1,1,0,1,1,0,0)),
    ("一つの処理は計算命令だけでなく制御命令やメモリ命令の組み合わせで実現されます。",
     (1,1,1,1,1,1,1,1,0,1,1,0,0)),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Semantic Encoder Adapter v0.7 instruction-computation hierarchy."
    )
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
    text = f"人: {prompt}\nAI: "
    ids = tokenizer.encode(text, add_bos=True)[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model.forward_hidden(x)[:, -1, :][0].detach()


def exact_overlap():
    dev = {str(case["prompt"]) for case in CASES}
    return sorted(prompt for prompt, _ in HIERARCHY_ROWS if prompt in dev)


def relation_loss(logits):
    p = torch.sigmoid(logits)
    idx = {name: SEMANTIC_HIERARCHY_LABELS.index(name)
           for name in SEMANTIC_HIERARCHY_LABELS}

    instruction = p[:, idx["instruction_execution"]]
    computation = p[:, idx["computation"]]

    # child <= parent for the explicit hierarchy.
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
        c = p[:, idx[child]]
        par = p[:, idx[parent]]
        losses.append(F.relu(c - par).mean())

    # computation and instruction execution should not collapse to the same axis.
    # On average instruction_execution must retain room for non-computation roles.
    losses.append(F.relu(0.05 + computation.mean() - instruction.mean()))

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
    base_model, base_checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    adapter, heads, hierarchy_head, init_checkpoint = initialize_from_v062(
        args.init_adapter,
        device,
    )

    # Preserve semantic geometry; train only the new hierarchy head.
    adapter.eval()
    heads.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    for p in heads.parameters():
        p.requires_grad_(False)

    hierarchy_hidden = torch.stack(
        [
            encode_prompt_hidden(base_model, tokenizer, prompt)
            for prompt, _target in HIERARCHY_ROWS
        ],
        dim=0,
    ).to(device)
    hierarchy_targets = torch.tensor(
        [target for _prompt, target in HIERARCHY_ROWS],
        dtype=torch.float32,
        device=device,
    )

    base_rows = build_rows()
    base_set = AdapterDataset(base_rows, base_model, tokenizer)
    preserve_hidden = torch.stack(
        [item[0] for item in base_set.items],
        dim=0,
    ).to(device)

    with torch.no_grad():
        initial_probs = torch.sigmoid(
            hierarchy_head(adapter(preserve_hidden))
        ).detach()

    params = list(hierarchy_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.7 Training")
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
    print("Hierarchy rows         :", len(HIERARCHY_ROWS))
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

        logits = hierarchy_head(adapter(hierarchy_hidden))
        hierarchy = F.binary_cross_entropy_with_logits(
            logits,
            hierarchy_targets,
        )
        relation = relation_loss(logits)

        preserve_probs = torch.sigmoid(
            hierarchy_head(adapter(preserve_hidden))
        )
        preservation = F.mse_loss(preserve_probs, initial_probs)

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
            f"| loss={value:.4f} "
            f"hier={hierarchy.item():.4f} "
            f"relation={relation.item():.4f} "
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

    save_semantic_adapter_v07_checkpoint(
        args.output,
        adapter,
        heads,
        hierarchy_head,
        epoch=best_epoch,
        loss=best,
        base_model=args.model,
        init_adapter=args.init_adapter,
        learning_rate=args.lr,
        relation_weight=args.relation_weight,
        hierarchy_weight=args.hierarchy_weight,
        preservation_weight=args.preservation_weight,
    )

    print()
    print("Semantic Encoder Adapter v0.7 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best loss        :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
