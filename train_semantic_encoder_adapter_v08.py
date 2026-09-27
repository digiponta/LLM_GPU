# train_semantic_encoder_adapter_v08.py
#
# Train v0.8 Hierarchy-Constrained Semantic Head.
#
# The semantic adapter is frozen. Only the constrained hierarchy head trains.
# Parent-child ordering is guaranteed structurally by the head.

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic_encoder_adapter_v08 import (
    SEMANTIC_HIERARCHY_LABELS,
    initialize_from_v072,
    save_semantic_adapter_v08_checkpoint,
)
from tokenizer_bpe import Tokenizer
from train_semantic_encoder_adapter_v01 import AdapterDataset
from train_semantic_encoder_adapter_v03 import build_rows
from train_semantic_encoder_adapter_v072 import PROGRAM_ROWS


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INIT = "model/model-gpu-v0.9.1-semantic-adapter-v072.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
SEED = 42


EXTRA_ROWS = [
    (
        "命令実行は計算より広い概念で、計算、分岐、メモリ操作、データ移動を含みます。",
        (1,1,1,1,1,0,1,1,1,1,0,1,1,0,0),
    ),
    (
        "反復計算は計算の一種であり、計算は命令実行の一部です。",
        (1,1,1,1,1,1,0,0,0,0,1,0,0,1,1),
    ),
    (
        "GPUの反復並列計算も命令列の中で実行される計算処理です。",
        (1,1,1,1,1,1,0,0,1,0,1,0,0,1,1),
    ),
    (
        "プログラムは命令列を実行し、命令列には計算以外の制御やメモリ操作も含まれます。",
        (1,1,1,1,1,0,1,1,1,1,0,1,1,0,0),
    ),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train v0.8 Hierarchy-Constrained Semantic Head."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--init-adapter", default=DEFAULT_INIT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hierarchy-weight", type=float, default=1.0)
    p.add_argument("--preservation-weight", type=float, default=0.5)
    return p.parse_args()


@torch.no_grad()
def encode_prompt_hidden(model, tokenizer, prompt):
    device = next(model.parameters()).device
    ids = tokenizer.encode(f"人: {prompt}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model.forward_hidden(x)[:, -1, :][0].detach()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model, args.init_adapter):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    base_model, base_checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    adapter, heads, hierarchy_head, init_checkpoint = initialize_from_v072(
        args.init_adapter,
        device,
    )
    adapter.eval()
    heads.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    for p in heads.parameters():
        p.requires_grad_(False)

    rows = list(PROGRAM_ROWS) + list(EXTRA_ROWS)
    hidden = torch.stack(
        [encode_prompt_hidden(base_model, tokenizer, prompt) for prompt, _ in rows],
        dim=0,
    ).to(device)
    targets = torch.tensor(
        [target for _prompt, target in rows],
        dtype=torch.float32,
        device=device,
    )

    base_rows = build_rows()
    base_set = AdapterDataset(base_rows, base_model, tokenizer)
    preserve_hidden = torch.stack(
        [item[0] for item in base_set.items],
        dim=0,
    ).to(device)

    # Preserve v0.7.2 marginals as a soft target while respecting v0.8's hard
    # structural constraints.
    from semantic_encoder_adapter_v072 import load_semantic_adapter_v072_checkpoint
    _a, _h, old_head, _c = load_semantic_adapter_v072_checkpoint(
        args.init_adapter,
        device,
    )
    with torch.no_grad():
        old_probs = torch.sigmoid(
            old_head(adapter(preserve_hidden))
        ).detach()

    params = list(hierarchy_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.8 Training")
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
    print("Constrained head       : trainable")
    print("Hierarchy labels       :", ", ".join(SEMANTIC_HIERARCHY_LABELS))
    print("Training rows          :", len(rows))
    print("Learning rate          :", args.lr)
    print("Hierarchy weight       :", args.hierarchy_weight)
    print("Preservation weight    :", args.preservation_weight)
    print("Hierarchy violations   : structurally impossible")
    print()

    best = float("inf")
    best_epoch = 0
    best_head = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        hierarchy_head.train()
        optimizer.zero_grad(set_to_none=True)

        logits = hierarchy_head(adapter(hidden))
        hierarchy = F.binary_cross_entropy_with_logits(logits, targets)

        preserve_probs = torch.sigmoid(
            hierarchy_head(adapter(preserve_hidden))
        )
        preservation = F.mse_loss(preserve_probs, old_probs)

        total = (
            args.hierarchy_weight * hierarchy
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

    save_semantic_adapter_v08_checkpoint(
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
        preservation_weight=args.preservation_weight,
    )

    print()
    print("Semantic Encoder Adapter v0.8 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best loss        :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
