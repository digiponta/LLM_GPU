# train_dynamic_top_competitor_v099.py
#
# v0.9.9: make the correct entity first token beat the dynamic global
# top competitor for entity-answer rows only.

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from augment_sft_v07 import augment_pairs
from dynamic_top_competitor_v099 import (
    dynamic_top_competitor_loss,
    entity_token_ids,
    save_checkpoint,
    single_entity_tag,
)
from entity_logit_alignment_v098 import ENTITY_BY_TAG, load_checkpoint as load_v098
from evaluate_partial_intent_v09 import CASES
from output_alignment_v097 import configure_output_alignment
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
DEFAULT_SOURCE = "model/model-gpu-v0.9.8-entity-logit-aligned.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.9-dynamic-top-competitor.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION = "data/instruction-ja.txt"
SEED = 42


def row_entity_eligible(row, tokenizer):
    prompt, answer, tags = row
    tag = single_entity_tag(tags)
    if tag is None:
        return None
    entity_ids = tokenizer.encode(ENTITY_BY_TAG[tag])
    answer_ids = tokenizer.encode(answer)
    if not entity_ids or not answer_ids or answer_ids[0] != entity_ids[0]:
        return None
    return tag


def ensure_entity_validation(train_rows, val_rows, tokenizer):
    train_rows = list(train_rows)
    val_rows = list(val_rows)
    val_tags = {
        tag for row in val_rows
        if (tag := row_entity_eligible(row, tokenizer)) is not None
    }
    moved = []

    for tag in ENTITY_BY_TAG:
        if tag in val_tags:
            continue
        for i, row in enumerate(train_rows):
            if row_entity_eligible(row, tokenizer) == tag:
                moved.append(train_rows.pop(i))
                val_rows.append(moved[-1])
                val_tags.add(tag)
                break

    return train_rows, val_rows, moved


class DynamicEntityDataset(Dataset):
    def __init__(self, rows, tokenizer, context_length, labels, token_by_tag):
        self.rows = []
        self.eligible_count = 0
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
            entity_answer = False
            if entity_tag is not None:
                entity_ids = tokenizer.encode(ENTITY_BY_TAG[entity_tag])
                gold_ids = tokenizer.encode(answer_text)
                entity_answer = bool(
                    entity_ids and gold_ids and entity_ids[0] == gold_ids[0]
                )
            self.eligible_count += int(entity_answer)

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
                torch.tensor(entity_answer, dtype=torch.bool),
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
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--technical-repeat", type=int, default=2)
    p.add_argument("--replay-repeat", type=int, default=2)
    p.add_argument("--freeze-through", type=int, default=3)
    p.add_argument("--block-lr", type=float, default=3e-7)
    p.add_argument("--semantic-lr", type=float, default=3e-5)
    p.add_argument("--lm-head-lr", type=float, default=3e-6)
    p.add_argument("--semantic-weight", type=float, default=0.08)
    p.add_argument("--top-weight", type=float, default=0.15)
    p.add_argument("--top-margin", type=float, default=0.75)
    p.add_argument("--anchor-weight", type=float, default=2e-4)
    return p.parse_args()


def batch_loss(
    model,
    semantic_head,
    batch,
    device,
    *,
    label_smoothing,
    semantic_weight,
    top_weight,
    top_margin,
    ignored_token_ids,
    lm_anchor,
    anchor_weight,
):
    (
        input_ids, targets, lm_mask, prompt_index,
        semantic_targets, entity_target, eligible,
    ) = batch
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    lm_mask = lm_mask.to(device)
    prompt_index = prompt_index.to(device)
    semantic_targets = semantic_targets.to(device)
    entity_target = entity_target.to(device)
    eligible = eligible.to(device)

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
    top_loss, eligible_n, target_vals, competitor_vals, competitor_ids = (
        dynamic_top_competitor_loss(
            prompt_logits,
            entity_target,
            eligible,
            top_margin,
            ignored_token_ids,
        )
    )

    anchor = F.mse_loss(model.lm_head.weight, lm_anchor)
    total = (
        lm
        + semantic_weight * semantic
        + top_weight * top_loss
        + anchor_weight * anchor
    )
    return (
        total, lm, semantic, top_loss, anchor,
        eligible_n, target_vals, competitor_vals, competitor_ids,
    )


@torch.no_grad()
def evaluate(model, semantic_head, loader, device, kwargs):
    model.eval()
    semantic_head.eval()
    sums = [0.0] * 5
    batches = 0
    eligible_total = 0

    for batch in loader:
        values = batch_loss(model, semantic_head, batch, device, **kwargs)
        for i in range(5):
            sums[i] += float(values[i].item())
        eligible_total += int(values[5])
        batches += 1

    n = max(1, batches)
    return tuple(v / n for v in sums), eligible_total


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
    model, semantic_head, labels, source_ckpt = load_v098(args.source, device)
    token_by_tag = entity_token_ids(tokenizer)

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
    train_rows, val_rows, moved = ensure_entity_validation(
        train_rows, val_rows, tokenizer
    )
    base_train_count = len(train_rows)
    train_rows = oversample(train_rows, args.technical_repeat, TECHNICAL_TAGS)
    train_rows = oversample(train_rows, args.replay_repeat, REPLAY_TAGS)

    train_set = DynamicEntityDataset(
        train_rows, tokenizer, model.context_length, labels, token_by_tag
    )
    val_set = DynamicEntityDataset(
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
    ignored = [
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": late_params, "lr": args.block_lr},
            {"params": semantic_head.parameters(), "lr": args.semantic_lr},
            {"params": model.lm_head.parameters(), "lr": args.lm_head_lr},
        ],
        weight_decay=0.01,
    )

    kwargs = dict(
        label_smoothing=args.label_smoothing,
        semantic_weight=args.semantic_weight,
        top_weight=args.top_weight,
        top_margin=args.top_margin,
        ignored_token_ids=ignored,
        lm_anchor=lm_anchor,
        anchor_weight=args.anchor_weight,
    )

    print()
    print("===========================================")
    print(" v0.9.9 Dynamic Top-Competitor Alignment")
    print("===========================================")
    print("Device                 :", device)
    print("Source                 :", args.source)
    print("Source loss            :", source_ckpt.get("loss"))
    print("Exact fixed prompts    : excluded")
    print("Excluded rows          :", excluded)
    print("Entity val rows moved  :", len(moved))
    print("Train rows base        :", base_train_count)
    print("Train rows final       :", len(train_rows))
    print("Validation rows        :", len(val_rows))
    print("Train eligible rows    :", train_set.eligible_count)
    print("Validation eligible    :", val_set.eligible_count)
    print("Block LR               :", args.block_lr)
    print("Semantic-head LR       :", args.semantic_lr)
    print("LM-head LR             :", args.lm_head_lr)
    print("Top margin / weight    :", args.top_margin, "/", args.top_weight)
    print("LM-head anchor weight  :", args.anchor_weight)
    print()

    best = float("inf")
    best_epoch = 0
    best_model = best_head = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        semantic_head.train()
        sums = [0.0] * 5
        batches = 0
        train_eligible = 0
        started = time.perf_counter()

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            values = batch_loss(model, semantic_head, batch, device, **kwargs)
            values[0].backward()
            torch.nn.utils.clip_grad_norm_(
                late_params
                + list(semantic_head.parameters())
                + list(model.lm_head.parameters()),
                1.0,
            )
            optimizer.step()
            for i in range(5):
                sums[i] += float(values[i].item())
            train_eligible += int(values[5])
            batches += 1

        n = max(1, batches)
        train_values = [v / n for v in sums]
        val_values, val_eligible = evaluate(
            model, semantic_head, val_loader, device, kwargs
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train={train_values[0]:.4f} lm={train_values[1]:.4f} "
            f"sem={train_values[2]:.4f} top={train_values[3]:.4f} "
            f"eligible={train_eligible} "
            f"| val={val_values[0]:.4f} lm={val_values[1]:.4f} "
            f"sem={val_values[2]:.4f} top={val_values[3]:.4f} "
            f"eligible={val_eligible} "
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
        raise RuntimeError("No valid v0.9.9 checkpoint.")

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
        top_margin=args.top_margin,
        top_weight=args.top_weight,
        lm_head_learning_rate=args.lm_head_lr,
    )

    print()
    print("Completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
