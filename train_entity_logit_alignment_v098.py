# train_entity_logit_alignment_v098.py
#
# v0.9.8: directly connect semantic technical states to entity output logits.
# Exact fixed benchmark prompts remain excluded.

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from augment_sft_v07 import augment_pairs
from entity_logit_alignment_v098 import (
    ENTITY_BY_TAG,
    blocker_margin_loss,
    entity_margin_loss,
    entity_token_ids,
    save_checkpoint,
    single_entity_tag,
)
from evaluate_partial_intent_v09 import CASES
from output_alignment_v097 import configure_output_alignment, load_checkpoint as load_v097
from tokenizer_bpe import Tokenizer
from train_sft_v09 import (
    AI_PREFIX,
    USER_PREFIX,
    REPLAY_TAGS,
    TECHNICAL_TAGS,
    deduplicate_pairs,
    oversample,
    parse_dialogues,
    parse_instruction_pairs,
    stratified_split,
)


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_SOURCE = "model/model-gpu-v0.9.7-output-aligned.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.8-entity-logit-aligned.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION = "data/instruction-ja.txt"
SEED = 42


class EntityDataset(Dataset):
    def __init__(self, rows, tokenizer, context_length, labels, token_by_tag):
        self.rows = []
        label_to_id = {label: i for i, label in enumerate(labels)}

        for user_text, answer_text, tags in rows:
            prompt = f"{USER_PREFIX}{user_text}\n{AI_PREFIX}"
            prompt_ids = tokenizer.encode(prompt, add_bos=True)
            answer_ids = tokenizer.encode(answer_text, add_eos=True)

            max_sequence = context_length + 1
            if len(prompt_ids) + len(answer_ids) > max_sequence:
                keep_prompt = max(1, max_sequence - len(answer_ids))
                prompt_ids = prompt_ids[-keep_prompt:]
            if len(prompt_ids) + len(answer_ids) > max_sequence:
                room = max_sequence - len(prompt_ids)
                answer_ids = answer_ids[:room]
                if answer_ids:
                    answer_ids[-1] = tokenizer.eos_id

            sequence = prompt_ids + answer_ids
            answer_start = len(prompt_ids)
            x = sequence[:-1]
            y = sequence[1:]
            lm_mask = [
                1.0 if (i + 1) >= answer_start else 0.0
                for i in range(len(y))
            ]
            prompt_index = min(max(0, answer_start - 1), len(x) - 1)

            semantic_target = [0.0] * len(labels)
            for tag in tags:
                if tag in label_to_id:
                    semantic_target[label_to_id[tag]] = 1.0

            entity_tag = single_entity_tag(tags)
            entity_target = token_by_tag.get(entity_tag, -1)

            starts_entity = False
            if entity_tag is not None:
                name = ENTITY_BY_TAG[entity_tag]
                gold = tokenizer.encode(answer_text)
                entity_ids = tokenizer.encode(name)
                starts_entity = bool(
                    gold and entity_ids and gold[0] == entity_ids[0]
                )

            pad = context_length - len(x)
            if pad < 0:
                raise RuntimeError("SFT row exceeded context length.")
            x += [tokenizer.pad_id] * pad
            y += [tokenizer.pad_id] * pad
            lm_mask += [0.0] * pad

            self.rows.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(lm_mask, dtype=torch.float32),
                torch.tensor(prompt_index, dtype=torch.long),
                torch.tensor(semantic_target, dtype=torch.float32),
                torch.tensor(entity_target, dtype=torch.long),
                torch.tensor(starts_entity, dtype=torch.bool),
            ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--instruction-data", default=DEFAULT_INSTRUCTION)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--technical-repeat", type=int, default=2)
    p.add_argument("--replay-repeat", type=int, default=2)
    p.add_argument("--freeze-through", type=int, default=3)
    p.add_argument("--block-lr", type=float, default=5e-7)
    p.add_argument("--semantic-lr", type=float, default=5e-5)
    p.add_argument("--lm-head-lr", type=float, default=2e-6)
    p.add_argument("--semantic-weight", type=float, default=0.10)
    p.add_argument("--entity-weight", type=float, default=0.20)
    p.add_argument("--blocker-weight", type=float, default=0.05)
    p.add_argument("--entity-margin", type=float, default=1.0)
    p.add_argument("--blocker-margin", type=float, default=0.25)
    p.add_argument("--anchor-weight", type=float, default=1e-4)
    return p.parse_args()


def batch_loss(
    model,
    semantic_head,
    batch,
    device,
    *,
    label_smoothing,
    semantic_weight,
    entity_ids,
    entity_weight,
    blocker_weight,
    entity_margin,
    blocker_margin,
    lm_anchor,
    anchor_weight,
):
    (
        input_ids, targets, lm_mask, prompt_index,
        semantic_targets, entity_target, starts_entity,
    ) = batch
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    lm_mask = lm_mask.to(device)
    prompt_index = prompt_index.to(device)
    semantic_targets = semantic_targets.to(device)
    entity_target = entity_target.to(device)
    starts_entity = starts_entity.to(device)

    hidden = model.forward_hidden(input_ids)
    logits = model.lm_head(hidden)

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(targets)
    lm = (token_loss * lm_mask).sum() / lm_mask.sum().clamp_min(1.0)

    rows = torch.arange(hidden.size(0), device=device)
    prompt_hidden = hidden[rows, prompt_index]
    semantic_logits = semantic_head(prompt_hidden)
    semantic = F.binary_cross_entropy_with_logits(
        semantic_logits, semantic_targets
    )

    prompt_logits = logits[rows, prompt_index]
    entity = entity_margin_loss(
        prompt_logits, entity_target, entity_ids, entity_margin
    )
    blocker = blocker_margin_loss(
        prompt_logits, entity_target, starts_entity, blocker_margin
    )
    anchor = F.mse_loss(model.lm_head.weight, lm_anchor)

    total = (
        lm
        + semantic_weight * semantic
        + entity_weight * entity
        + blocker_weight * blocker
        + anchor_weight * anchor
    )
    return total, lm, semantic, entity, blocker, anchor


@torch.no_grad()
def evaluate(model, semantic_head, loader, device, kwargs):
    model.eval()
    semantic_head.eval()
    sums = [0.0] * 6
    batches = 0
    for batch in loader:
        values = batch_loss(model, semantic_head, batch, device, **kwargs)
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
    model, semantic_head, labels, source_ckpt = load_v097(args.source, device)

    token_by_tag = entity_token_ids(tokenizer)
    entity_ids = list(dict.fromkeys(token_by_tag.values()))

    conversation = parse_dialogues(Path(args.data).read_text(encoding="utf-8"))
    instruction = parse_instruction_pairs(
        Path(args.instruction_data).read_text(encoding="utf-8")
    )
    rows = augment_pairs(
        deduplicate_pairs(conversation + instruction),
        variants_per_intent=args.variants_per_intent,
    )

    fixed_prompts = {str(case["prompt"]).strip() for case in CASES}
    before = len(rows)
    rows = [row for row in rows if row[0].strip() not in fixed_prompts]
    excluded = before - len(rows)

    train_rows, val_rows = stratified_split(rows, args.validation_ratio, SEED)
    base_train_count = len(train_rows)
    train_rows = oversample(train_rows, args.technical_repeat, TECHNICAL_TAGS)
    train_rows = oversample(train_rows, args.replay_repeat, REPLAY_TAGS)

    train_set = EntityDataset(
        train_rows, tokenizer, model.context_length, labels, token_by_tag
    )
    val_set = EntityDataset(
        val_rows, tokenizer, model.context_length, labels, token_by_tag
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    configure_output_alignment(model, args.freeze_through)
    for p in semantic_head.parameters():
        p.requires_grad_(True)

    late_params = []
    for index, block in enumerate(model.blocks, start=1):
        if index > args.freeze_through:
            late_params.extend(list(block.parameters()))
    late_params.extend(list(model.final_norm.parameters()))

    lm_anchor = model.lm_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [
            {"params": late_params, "lr": args.block_lr},
            {"params": semantic_head.parameters(), "lr": args.semantic_lr},
            {"params": model.lm_head.parameters(), "lr": args.lm_head_lr},
        ],
        weight_decay=0.01,
    )

    loss_kwargs = dict(
        label_smoothing=args.label_smoothing,
        semantic_weight=args.semantic_weight,
        entity_ids=entity_ids,
        entity_weight=args.entity_weight,
        blocker_weight=args.blocker_weight,
        entity_margin=args.entity_margin,
        blocker_margin=args.blocker_margin,
        lm_anchor=lm_anchor,
        anchor_weight=args.anchor_weight,
    )

    print()
    print("==========================================")
    print(" v0.9.8 Semantic Entity Logit Alignment")
    print("==========================================")
    print("Device               :", device)
    print("Source               :", args.source)
    print("Source loss          :", source_ckpt.get("loss"))
    print("Exact fixed prompts  : excluded")
    print("Excluded rows        :", excluded)
    print("Train rows base      :", base_train_count)
    print("Train rows final     :", len(train_rows))
    print("Validation rows      :", len(val_rows))
    print("Block LR             :", args.block_lr)
    print("Semantic-head LR     :", args.semantic_lr)
    print("LM-head LR           :", args.lm_head_lr)
    print("Entity margin/weight :", args.entity_margin, "/", args.entity_weight)
    print("Blocker margin/weight:", args.blocker_margin, "/", args.blocker_weight)
    print("Entity first tokens  :")
    for tag, name in ENTITY_BY_TAG.items():
        tid = token_by_tag[tag]
        print(f"  {tag:16s} {name:12s} id={tid} decoded={tokenizer.decode([tid])!r}")
    print()

    best = float("inf")
    best_epoch = 0
    best_model = best_head = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        semantic_head.train()
        sums = [0.0] * 6
        batches = 0
        started = time.perf_counter()

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            values = batch_loss(
                model, semantic_head, batch, device, **loss_kwargs
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
            model, semantic_head, val_loader, device, loss_kwargs
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train={train_values[0]:.4f} lm={train_values[1]:.4f} "
            f"sem={train_values[2]:.4f} ent={train_values[3]:.4f} "
            f"blk={train_values[4]:.4f} "
            f"| val={val_values[0]:.4f} lm={val_values[1]:.4f} "
            f"sem={val_values[2]:.4f} ent={val_values[3]:.4f} "
            f"blk={val_values[4]:.4f} "
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
        raise RuntimeError("No valid v0.9.8 checkpoint.")

    model.load_state_dict(best_model)
    semantic_head.load_state_dict(best_head)
    save_checkpoint(
        args.output, model, semantic_head, labels,
        epoch=best_epoch,
        loss=best,
        source_checkpoint=args.source,
        entity_margin=args.entity_margin,
        blocker_margin=args.blocker_margin,
        entity_weight=args.entity_weight,
        blocker_weight=args.blocker_weight,
    )

    print()
    print("Completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
