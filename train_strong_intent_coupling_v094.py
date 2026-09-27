# train_strong_intent_coupling_v094.py
#
# LLM_GPU v0.9.4 Strong Intent -> Generation Coupling
#
# Frozen:
#   - v0.8 pairwise-best LM
#   - v0.9.2 corrected intent head
#
# Trainable:
#   - FiLM-style coupling only

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augment_sft_v07 import augment_pairs
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from strong_intent_coupling_v094 import (
    StrongIntentFiLM,
    forward_strong_intent,
    save_checkpoint,
)
from tokenizer_bpe import Tokenizer
from train_sft_v09 import (
    ProjectionDataset,
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
DEFAULT_OUTPUT = "model/model-gpu-v0.9.4-strong-intent-film.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION = "data/instruction-ja.txt"
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_HEAD)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--instruction-data", default=DEFAULT_INSTRUCTION)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--technical-repeat", type=int, default=2)
    p.add_argument("--replay-repeat", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--inject-after", type=int, default=3)
    p.add_argument("--scale-limit", type=float, default=0.50)
    p.add_argument("--shift-limit", type=float, default=0.50)
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
    label_smoothing,
    film_l2,
):
    with torch.no_grad():
        base_hidden = model.forward_hidden(input_ids)
        batch = torch.arange(base_hidden.size(0), device=base_hidden.device)
        prompt_repr = base_hidden[batch, prompt_index]
        intent_prob = torch.sigmoid(intent_head(prompt_repr))

    scale, shift = coupling(intent_prob)
    hidden = forward_strong_intent(
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
    reg = scale.pow(2).mean() + shift.pow(2).mean()
    return lm + film_l2 * reg, lm, reg


@torch.no_grad()
def evaluate(
    model,
    intent_head,
    coupling,
    loader,
    device,
    label_smoothing,
    film_l2,
):
    coupling.eval()
    total = lm_total = reg_total = 0.0
    batches = 0
    for input_ids, targets, mask, prompt_index in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        prompt_index = prompt_index.to(device)
        loss, lm, reg = conditioned_loss(
            model,
            intent_head,
            coupling,
            input_ids,
            targets,
            mask,
            prompt_index,
            label_smoothing,
            film_l2,
        )
        total += float(loss.item())
        lm_total += float(lm.item())
        reg_total += float(reg.item())
        batches += 1
    n = max(1, batches)
    return total / n, lm_total / n, reg_total / n


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

    coupling = StrongIntentFiLM(
        num_labels=len(labels),
        d_model=model.d_model,
        hidden_dim=args.hidden_dim,
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

    train_set = ProjectionDataset(train_rows, tokenizer, model.context_length)
    val_set = ProjectionDataset(val_rows, tokenizer, model.context_length)
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
    print("=============================================")
    print(" v0.9.4 Strong Intent -> Generation Training")
    print("=============================================")
    print("Device              :", device)
    print("Base loss           :", base_ckpt.get("loss"))
    print("Intent-head loss    :", head_ckpt.get("loss"))
    print("Base LM             : frozen")
    print("Intent head         : frozen")
    print("Coupling            : FiLM trainable")
    print("Intent dim          :", len(labels))
    print("MLP                 :", f"{len(labels)} -> {args.hidden_dim} -> {2*model.d_model}")
    print("Inject after block  :", args.inject_after)
    print("Scale limit         :", args.scale_limit)
    print("Shift limit         :", args.shift_limit)
    print("Trainable params    :", sum(p.numel() for p in coupling.parameters()))
    print("Train rows base     :", base_train_count)
    print("Train rows final    :", len(train_rows))
    print("Validation rows     :", len(val_rows))
    print("Technical repeat    :", args.technical_repeat)
    print()

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        coupling.train()
        total = lm_total = reg_total = 0.0
        batches = 0
        started = time.perf_counter()

        for input_ids, targets, mask, prompt_index in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            prompt_index = prompt_index.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss, lm, reg = conditioned_loss(
                model,
                head,
                coupling,
                input_ids,
                targets,
                mask,
                prompt_index,
                args.label_smoothing,
                args.film_l2,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(coupling.parameters(), 1.0)
            optimizer.step()

            total += float(loss.item())
            lm_total += float(lm.item())
            reg_total += float(reg.item())
            batches += 1

        n = max(1, batches)
        train_loss = total / n
        train_lm = lm_total / n
        train_reg = reg_total / n
        val_loss, val_lm, val_reg = evaluate(
            model,
            head,
            coupling,
            val_loader,
            device,
            args.label_smoothing,
            args.film_l2,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train={train_loss:.4f} lm={train_lm:.4f} reg={train_reg:.5f} "
            f"| val={val_loss:.4f} lm={val_lm:.4f} reg={val_reg:.5f} "
            f"| {time.perf_counter()-started:.2f}s"
        )

        if val_loss < best - 1e-5:
            best = val_loss
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
        raise RuntimeError("No valid v0.9.4 coupling checkpoint.")

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
    )

    print()
    print("Completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
