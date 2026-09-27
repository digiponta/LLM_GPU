# train_semantic_encoder_adaptation_v096.py
#
# v0.9.6: adapt late Transformer representation with joint LM + semantic loss.
# Exact fixed 30 benchmark prompts are removed from the augmented training rows.

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from augment_sft_v07 import augment_pairs
from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adaptation_v096 import (
    SemanticTagHead,
    configure_semantic_adaptation,
    save_checkpoint,
)
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
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.6-semantic-encoder-adapted.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION = "data/instruction-ja.txt"
SEED = 42


class SemanticAdaptDataset(Dataset):
    def __init__(self, rows, tokenizer, context_length, labels):
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

            pad = context_length - len(x)
            if pad < 0:
                raise RuntimeError("SFT row exceeded context length.")
            x += [tokenizer.pad_id] * pad
            y += [tokenizer.pad_id] * pad
            lm_mask += [0.0] * pad

            target = [0.0] * len(labels)
            for tag in tags:
                if tag in label_to_id:
                    target[label_to_id[tag]] = 1.0

            self.rows.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(lm_mask, dtype=torch.float32),
                torch.tensor(prompt_index, dtype=torch.long),
                torch.tensor(target, dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--instruction-data", default=DEFAULT_INSTRUCTION)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--technical-repeat", type=int, default=2)
    p.add_argument("--replay-repeat", type=int, default=2)
    p.add_argument("--freeze-through", type=int, default=3)
    p.add_argument("--block-lr", type=float, default=3e-6)
    p.add_argument("--semantic-lr", type=float, default=3e-4)
    p.add_argument("--semantic-weight", type=float, default=0.20)
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
        semantic_logits,
        semantic_targets,
    )
    total = lm + semantic_weight * semantic
    return total, lm, semantic


@torch.no_grad()
def evaluate(
    model,
    semantic_head,
    loader,
    device,
    label_smoothing,
    semantic_weight,
):
    model.eval()
    semantic_head.eval()
    sums = [0.0, 0.0, 0.0]
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
        args.tokenizer, args.model, args.data, args.instruction_data
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ckpt = LanguageModel.load_checkpoint(args.model, device=device)

    conversation = parse_dialogues(Path(args.data).read_text(encoding="utf-8"))
    instruction = parse_instruction_pairs(
        Path(args.instruction_data).read_text(encoding="utf-8")
    )
    base_pairs = deduplicate_pairs(conversation + instruction)
    rows = augment_pairs(base_pairs, variants_per_intent=args.variants_per_intent)

    fixed_prompts = {str(case["prompt"]).strip() for case in CASES}
    before_exclusion = len(rows)
    rows = [row for row in rows if row[0].strip() not in fixed_prompts]
    excluded = before_exclusion - len(rows)

    labels = sorted({
        tag
        for _prompt, _answer, tags in rows
        for tag in tags
    })
    if not labels:
        raise RuntimeError("No semantic labels found.")

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

    configure_semantic_adaptation(model, args.freeze_through)
    semantic_head = SemanticTagHead(model.d_model, len(labels)).to(device)

    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": model_params, "lr": args.block_lr},
            {"params": semantic_head.parameters(), "lr": args.semantic_lr},
        ],
        weight_decay=0.01,
    )

    print()
    print("===========================================")
    print(" v0.9.6 Semantic Encoder Adaptation")
    print("===========================================")
    print("Device               :", device)
    print("Base loss            :", base_ckpt.get("loss"))
    print("Blocks 1-3           : frozen")
    print("Blocks 4-6           : trainable")
    print("FinalNorm            : trainable")
    print("LM head              : frozen")
    print("Semantic head        : trainable")
    print("Semantic labels      :", len(labels))
    print("Exact fixed prompts  : excluded")
    print("Excluded rows        :", excluded)
    print("Train rows base      :", base_train_count)
    print("Train rows final     :", len(train_rows))
    print("Validation rows      :", len(val_rows))
    print("Block LR             :", args.block_lr)
    print("Semantic-head LR     :", args.semantic_lr)
    print("Semantic weight      :", args.semantic_weight)
    print("Trainable model par. :", sum(p.numel() for p in model_params))
    print()

    best = float("inf")
    best_epoch = 0
    best_model = None
    best_head = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        semantic_head.train()
        sums = [0.0, 0.0, 0.0]
        batches = 0
        started = time.perf_counter()

        for input_ids, targets, mask, prompt_index, semantic_targets in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            prompt_index = prompt_index.to(device, non_blocking=True)
            semantic_targets = semantic_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            values = loss_fn(
                model,
                semantic_head,
                input_ids,
                targets,
                mask,
                prompt_index,
                semantic_targets,
                args.label_smoothing,
                args.semantic_weight,
            )
            values[0].backward()
            torch.nn.utils.clip_grad_norm_(
                model_params + list(semantic_head.parameters()), 1.0
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
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train={train_values[0]:.4f} lm={train_values[1]:.4f} "
            f"sem={train_values[2]:.4f} "
            f"| val={val_values[0]:.4f} lm={val_values[1]:.4f} "
            f"sem={val_values[2]:.4f} "
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
        raise RuntimeError("No valid v0.9.6 checkpoint.")

    model.load_state_dict(best_model)
    semantic_head.load_state_dict(best_head)
    save_checkpoint(
        args.output,
        model,
        semantic_head,
        labels,
        epoch=best_epoch,
        loss=best,
        base_model=args.model,
        freeze_through=args.freeze_through,
        block_learning_rate=args.block_lr,
        semantic_learning_rate=args.semantic_lr,
        semantic_weight=args.semantic_weight,
    )

    print()
    print("Completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
