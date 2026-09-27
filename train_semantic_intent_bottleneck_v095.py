# train_semantic_intent_bottleneck_v095.py
#
# LLM_GPU v0.9.5 Semantic Intent Bottleneck
#
# Frozen:
#   - v0.8 pairwise-best LM
#   - v0.9.2 corrected intent head
#
# Trainable:
#   - 256 -> 64 semantic bottleneck
#   - 64 -> 24 supervised semantic tag head
#   - fused semantic/intention FiLM coupling

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from augment_sft_v07 import augment_pairs
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from semantic_intent_bottleneck_v095 import (
    SemanticIntentBottleneck,
    forward_semantic_intent,
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
DEFAULT_HEAD = "model/model-gpu-v0.9.2-intent-head-implicit-cpu.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.5-semantic-intent-bottleneck.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION = "data/instruction-ja.txt"
SEED = 42


class SemanticCouplingDataset(Dataset):
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

            semantic_target = [0.0] * len(labels)
            for tag in tags:
                if tag in label_to_id:
                    semantic_target[label_to_id[tag]] = 1.0

            self.rows.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(lm_mask, dtype=torch.float32),
                torch.tensor(prompt_index, dtype=torch.long),
                torch.tensor(semantic_target, dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_HEAD)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--instruction-data", default=DEFAULT_INSTRUCTION)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--learning-rate", type=float, default=7e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--technical-repeat", type=int, default=2)
    p.add_argument("--replay-repeat", type=int, default=2)
    p.add_argument("--semantic-dim", type=int, default=64)
    p.add_argument("--fusion-dim", type=int, default=128)
    p.add_argument("--inject-after", type=int, default=3)
    p.add_argument("--scale-limit", type=float, default=0.50)
    p.add_argument("--shift-limit", type=float, default=0.50)
    p.add_argument("--semantic-weight", type=float, default=0.25)
    p.add_argument("--film-l2", type=float, default=1e-4)
    return p.parse_args()


def conditioned_loss(
    model,
    intent_head,
    coupling,
    input_ids,
    targets,
    lm_mask,
    prompt_index,
    semantic_targets,
    label_smoothing,
    semantic_weight,
    film_l2,
):
    with torch.no_grad():
        base_hidden = model.forward_hidden(input_ids)
        batch = torch.arange(base_hidden.size(0), device=base_hidden.device)
        prompt_hidden = base_hidden[batch, prompt_index]
        frozen_intent = torch.sigmoid(intent_head(prompt_hidden))

    scale, shift, semantic_logits, _semantic_prob, _z = coupling(
        prompt_hidden,
        frozen_intent,
    )

    hidden = forward_semantic_intent(
        model,
        input_ids,
        scale,
        shift,
        coupling.inject_after,
    )
    logits = model.lm_head(hidden)

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(targets)
    lm = (token_loss * lm_mask).sum() / lm_mask.sum().clamp_min(1.0)

    semantic = F.binary_cross_entropy_with_logits(
        semantic_logits,
        semantic_targets,
    )

    reg = scale.pow(2).mean() + shift.pow(2).mean()
    total = lm + semantic_weight * semantic + film_l2 * reg
    return total, lm, semantic, reg


@torch.no_grad()
def evaluate(
    model,
    intent_head,
    coupling,
    loader,
    device,
    label_smoothing,
    semantic_weight,
    film_l2,
):
    coupling.eval()
    sums = [0.0, 0.0, 0.0, 0.0]
    batches = 0
    for input_ids, targets, mask, prompt_index, semantic_targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        prompt_index = prompt_index.to(device)
        semantic_targets = semantic_targets.to(device)

        values = conditioned_loss(
            model,
            intent_head,
            coupling,
            input_ids,
            targets,
            mask,
            prompt_index,
            semantic_targets,
            label_smoothing,
            semantic_weight,
            film_l2,
        )
        for i, value in enumerate(values):
            sums[i] += float(value.item())
        batches += 1

    n = max(1, batches)
    return tuple(value / n for value in sums)


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (
        args.tokenizer,
        args.model,
        args.intent_head,
        args.data,
        args.instruction_data,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ckpt = LanguageModel.load_checkpoint(args.model, device=device)
    head, head_ckpt, labels = load_intent_head(args.intent_head, model, device)

    model.eval()
    head.eval()
    for module in (model, head):
        for p in module.parameters():
            p.requires_grad_(False)

    coupling = SemanticIntentBottleneck(
        num_labels=len(labels),
        d_model=model.d_model,
        semantic_dim=args.semantic_dim,
        fusion_dim=args.fusion_dim,
        inject_after=args.inject_after,
        scale_limit=args.scale_limit,
        shift_limit=args.shift_limit,
    ).to(device)

    conversation = parse_dialogues(Path(args.data).read_text(encoding="utf-8"))
    instruction = parse_instruction_pairs(
        Path(args.instruction_data).read_text(encoding="utf-8")
    )
    base_pairs = deduplicate_pairs(conversation + instruction)
    rows = augment_pairs(base_pairs, variants_per_intent=args.variants_per_intent)

    train_rows, val_rows = stratified_split(rows, args.validation_ratio, SEED)
    base_train_count = len(train_rows)
    train_rows = oversample(train_rows, args.technical_repeat, TECHNICAL_TAGS)
    train_rows = oversample(train_rows, args.replay_repeat, REPLAY_TAGS)

    train_set = SemanticCouplingDataset(
        train_rows, tokenizer, model.context_length, labels
    )
    val_set = SemanticCouplingDataset(
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

    optimizer = torch.optim.AdamW(
        coupling.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    print()
    print("==============================================")
    print(" v0.9.5 Semantic Intent Bottleneck Training")
    print("==============================================")
    print("Device              :", device)
    print("Base loss           :", base_ckpt.get("loss"))
    print("Intent-head loss    :", head_ckpt.get("loss"))
    print("Base LM             : frozen")
    print("v0.9.2 intent head  : frozen")
    print("Semantic bottleneck : trainable")
    print("Semantic dims       :", f"{model.d_model} -> {args.semantic_dim}")
    print("Semantic tag head   :", f"{args.semantic_dim} -> {len(labels)}")
    print(
        "Fusion input        :",
        args.semantic_dim + 2 * len(labels),
        "(semantic + learned tags + frozen intents)",
    )
    print("Fusion hidden       :", args.fusion_dim)
    print("Inject after block  :", args.inject_after)
    print("Scale/shift limit   :", args.scale_limit, "/", args.shift_limit)
    print("Semantic weight     :", args.semantic_weight)
    print("Trainable params    :", sum(p.numel() for p in coupling.parameters()))
    print("Train rows base     :", base_train_count)
    print("Train rows final    :", len(train_rows))
    print("Validation rows     :", len(val_rows))
    print()

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        coupling.train()
        sums = [0.0, 0.0, 0.0, 0.0]
        batches = 0
        started = time.perf_counter()

        for input_ids, targets, mask, prompt_index, semantic_targets in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            prompt_index = prompt_index.to(device, non_blocking=True)
            semantic_targets = semantic_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            values = conditioned_loss(
                model,
                head,
                coupling,
                input_ids,
                targets,
                mask,
                prompt_index,
                semantic_targets,
                args.label_smoothing,
                args.semantic_weight,
                args.film_l2,
            )
            values[0].backward()
            torch.nn.utils.clip_grad_norm_(coupling.parameters(), 1.0)
            optimizer.step()

            for i, value in enumerate(values):
                sums[i] += float(value.item())
            batches += 1

        n = max(1, batches)
        train_values = [value / n for value in sums]
        val_values = evaluate(
            model,
            head,
            coupling,
            val_loader,
            device,
            args.label_smoothing,
            args.semantic_weight,
            args.film_l2,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train={train_values[0]:.4f} lm={train_values[1]:.4f} "
            f"sem={train_values[2]:.4f} reg={train_values[3]:.5f} "
            f"| val={val_values[0]:.4f} lm={val_values[1]:.4f} "
            f"sem={val_values[2]:.4f} reg={val_values[3]:.5f} "
            f"| {time.perf_counter()-started:.2f}s"
        )

        if val_values[0] < best - 1e-5:
            best = val_values[0]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in coupling.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("No valid v0.9.5 checkpoint.")

    coupling.load_state_dict(best_state)
    save_checkpoint(
        args.output,
        coupling,
        labels,
        epoch=best_epoch,
        loss=best,
        base_model=args.model,
        intent_head=args.intent_head,
        learning_rate=args.learning_rate,
        semantic_weight=args.semantic_weight,
    )

    print()
    print("Completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
