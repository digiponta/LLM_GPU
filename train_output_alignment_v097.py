# train_output_alignment_v097.py
#
# Continue from v0.9.6 and align the frozen-output geometry to the adapted
# semantic hidden representation.
#
# Trainable:
#   Blocks 4-6 + FinalNorm : small LR
#   Semantic head          : small LR
#   LM head                : very small LR
#
# A weight anchor keeps the LM head close to the v0.9.6 starting geometry.

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augment_sft_v07 import augment_pairs
from evaluate_partial_intent_v09 import CASES
from output_alignment_v097 import configure_output_alignment, save_checkpoint
from semantic_encoder_adaptation_v096 import load_checkpoint as load_v096
from tokenizer_bpe import Tokenizer
from train_semantic_encoder_adaptation_v096 import SemanticAdaptDataset
from train_sft_v09 import (
    REPLAY_TAGS,
    TECHNICAL_TAGS,
    deduplicate_pairs,
    oversample,
    parse_dialogues,
    parse_instruction_pairs,
    stratified_split,
)


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_SOURCE = "model/model-gpu-v0.9.6-semantic-encoder-adapted.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.7-output-aligned.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION = "data/instruction-ja.txt"
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--instruction-data", default=DEFAULT_INSTRUCTION)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--technical-repeat", type=int, default=2)
    p.add_argument("--replay-repeat", type=int, default=2)
    p.add_argument("--freeze-through", type=int, default=3)
    p.add_argument("--block-lr", type=float, default=1e-6)
    p.add_argument("--semantic-lr", type=float, default=1e-4)
    p.add_argument("--lm-head-lr", type=float, default=1e-6)
    p.add_argument("--semantic-weight", type=float, default=0.20)
    p.add_argument("--lm-head-anchor-weight", type=float, default=1e-4)
    return p.parse_args()


def loss_fn(
    model,
    semantic_head,
    input_ids,
    targets,
    lm_mask,
    prompt_index,
    semantic_targets,
    label_smoothing,
    semantic_weight,
    lm_head_anchor,
    anchor_weight,
):
    hidden = model.forward_hidden(input_ids)
    logits = model.lm_head(hidden)

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(targets)
    lm = (token_loss * lm_mask).sum() / lm_mask.sum().clamp_min(1.0)

    batch = torch.arange(hidden.size(0), device=hidden.device)
    prompt_hidden = hidden[batch, prompt_index]
    semantic_logits = semantic_head(prompt_hidden)
    semantic = F.binary_cross_entropy_with_logits(
        semantic_logits, semantic_targets
    )

    anchor = F.mse_loss(model.lm_head.weight, lm_head_anchor)
    total = lm + semantic_weight * semantic + anchor_weight * anchor
    return total, lm, semantic, anchor


@torch.no_grad()
def evaluate(
    model,
    semantic_head,
    loader,
    device,
    label_smoothing,
    semantic_weight,
    lm_head_anchor,
    anchor_weight,
):
    model.eval()
    semantic_head.eval()
    sums = [0.0, 0.0, 0.0, 0.0]
    batches = 0
    for input_ids, targets, mask, prompt_index, semantic_targets in loader:
        values = loss_fn(
            model,
            semantic_head,
            input_ids.to(device),
            targets.to(device),
            mask.to(device),
            prompt_index.to(device),
            semantic_targets.to(device),
            label_smoothing,
            semantic_weight,
            lm_head_anchor,
            anchor_weight,
        )
        for i, value in enumerate(values):
            sums[i] += float(value.item())
        batches += 1
    n = max(1, batches)
    return tuple(v / n for v in sums)


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (
        args.tokenizer, args.source, args.data, args.instruction_data
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, semantic_head, labels, source_ckpt = load_v096(args.source, device)

    conversation = parse_dialogues(Path(args.data).read_text(encoding="utf-8"))
    instruction = parse_instruction_pairs(
        Path(args.instruction_data).read_text(encoding="utf-8")
    )
    base_pairs = deduplicate_pairs(conversation + instruction)
    rows = augment_pairs(base_pairs, variants_per_intent=args.variants_per_intent)

    fixed_prompts = {str(case["prompt"]).strip() for case in CASES}
    before = len(rows)
    rows = [row for row in rows if row[0].strip() not in fixed_prompts]
    excluded = before - len(rows)

    train_rows, val_rows = stratified_split(
        rows, args.validation_ratio, SEED
    )
    base_train_count = len(train_rows)
    train_rows = oversample(train_rows, args.technical_repeat, TECHNICAL_TAGS)
    train_rows = oversample(train_rows, args.replay_repeat, REPLAY_TAGS)

    train_set = SemanticAdaptDataset(
        train_rows, tokenizer, model.context_length, labels
    )
    val_set = SemanticAdaptDataset(
        val_rows, tokenizer, model.context_length, labels
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    configure_output_alignment(model, args.freeze_through)
    for p in semantic_head.parameters():
        p.requires_grad_(True)

    lm_head_anchor = model.lm_head.weight.detach().clone()

    late_params = []
    for index, block in enumerate(model.blocks, start=1):
        if index > args.freeze_through:
            late_params.extend(list(block.parameters()))
    late_params.extend(list(model.final_norm.parameters()))

    optimizer = torch.optim.AdamW(
        [
            {"params": late_params, "lr": args.block_lr},
            {"params": semantic_head.parameters(), "lr": args.semantic_lr},
            {"params": model.lm_head.parameters(), "lr": args.lm_head_lr},
        ],
        weight_decay=0.01,
    )

    print()
    print("====================================")
    print(" v0.9.7 Output Alignment")
    print("====================================")
    print("Device               :", device)
    print("Source checkpoint    :", args.source)
    print("Source loss          :", source_ckpt.get("loss"))
    print("Blocks 1-3           : frozen")
    print("Blocks 4-6           : trainable")
    print("FinalNorm            : trainable")
    print("Semantic head        : trainable")
    print("LM head              : trainable")
    print("Exact fixed prompts  : excluded")
    print("Excluded rows        :", excluded)
    print("Train rows base      :", base_train_count)
    print("Train rows final     :", len(train_rows))
    print("Validation rows      :", len(val_rows))
    print("Block LR             :", args.block_lr)
    print("Semantic-head LR     :", args.semantic_lr)
    print("LM-head LR           :", args.lm_head_lr)
    print("LM-head anchor weight:", args.lm_head_anchor_weight)
    print()

    best = float("inf")
    best_epoch = 0
    best_model = None
    best_head = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        semantic_head.train()
        sums = [0.0, 0.0, 0.0, 0.0]
        batches = 0
        started = time.perf_counter()

        for input_ids, targets, mask, prompt_index, semantic_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            values = loss_fn(
                model,
                semantic_head,
                input_ids.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                prompt_index.to(device, non_blocking=True),
                semantic_targets.to(device, non_blocking=True),
                args.label_smoothing,
                args.semantic_weight,
                lm_head_anchor,
                args.lm_head_anchor_weight,
            )
            values[0].backward()
            torch.nn.utils.clip_grad_norm_(
                late_params
                + list(semantic_head.parameters())
                + list(model.lm_head.parameters()),
                1.0,
            )
            optimizer.step()

            for i, value in enumerate(values):
                sums[i] += float(value.item())
            batches += 1

        n = max(1, batches)
        train_values = [v / n for v in sums]
        val_values = evaluate(
            model,
            semantic_head,
            val_loader,
            device,
            args.label_smoothing,
            args.semantic_weight,
            lm_head_anchor,
            args.lm_head_anchor_weight,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train={train_values[0]:.4f} lm={train_values[1]:.4f} "
            f"sem={train_values[2]:.4f} anchor={train_values[3]:.7f} "
            f"| val={val_values[0]:.4f} lm={val_values[1]:.4f} "
            f"sem={val_values[2]:.4f} anchor={val_values[3]:.7f} "
            f"| {time.perf_counter()-started:.2f}s"
        )

        if val_values[0] < best - 1e-5:
            best = val_values[0]
            best_epoch = epoch
            bad = 0
            best_model = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_head = {
                k: v.detach().cpu().clone()
                for k, v in semantic_head.state_dict().items()
            }
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    if best_model is None or best_head is None:
        raise RuntimeError("No valid v0.9.7 checkpoint.")

    model.load_state_dict(best_model)
    semantic_head.load_state_dict(best_head)

    save_checkpoint(
        args.output,
        model,
        semantic_head,
        labels,
        epoch=best_epoch,
        loss=best,
        source_checkpoint=args.source,
        freeze_through=args.freeze_through,
        block_learning_rate=args.block_lr,
        semantic_learning_rate=args.semantic_lr,
        lm_head_learning_rate=args.lm_head_lr,
        semantic_weight=args.semantic_weight,
        lm_head_anchor_weight=args.lm_head_anchor_weight,
    )

    print()
    print("Completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
