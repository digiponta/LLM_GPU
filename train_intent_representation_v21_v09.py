# train_intent_representation_v21_v09.py
#
# LLM_GPU v0.9 Intent Representation v2.1 training.
#
# Conditioning:
#   frozen intent model prompt hidden (256)
#   -> Linear(256 -> 32) -> tanh
#   + frozen intent probabilities (24)
#   -> Linear(56 -> 256), zero-init
#   -> inject after Block 3
#
# Data configuration intentionally matches Boundary v1 only.

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
from intent_representation_v21_v09 import (
    IntentRepresentationV21Projection,
    forward_representation_v21_conditioned,
    infer_intent_representation_v21,
    save_representation_v21_checkpoint,
)
from mid_intent_conditioning_v09 import validate_injection_point
from model import LanguageModel
from partial_intent_conditioning_v09 import (
    configure_partial_finetune,
    trainable_model_parameters,
    trainable_parameter_count,
)
from tokenizer_bpe import Tokenizer
from train_partial_intent_v09 import (
    AI_PREFIX,
    USER_PREFIX,
    REPLAY_TAGS,
    TECHNICAL_TAGS,
    PartialIntentDataset,
    deduplicate_pairs,
    oversample,
    parse_dialogues,
    parse_instruction_pairs,
    stratified_split,
)


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INTENT_HEAD = "model/model-gpu-v0.8-intent-head.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9-intent-representation-v21.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION_DATA = "data/instruction-ja.txt"
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(
        description="Train v0.9 Intent Representation v2.1."
    )
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--instruction-data", default=DEFAULT_INSTRUCTION_DATA)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT_HEAD)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--technical-repeat", type=int, default=1)
    p.add_argument("--replay-repeat", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--semantic-dim", type=int, default=32)
    p.add_argument("--inject-after", type=int, default=3)
    p.add_argument("--projection-lr", type=float, default=1e-3)
    p.add_argument("--block-lr", type=float, default=1e-5)
    return p.parse_args()


def conditioned_loss(
    generation_model,
    intent_model,
    intent_head,
    projection,
    input_ids,
    targets,
    lm_mask,
    prompt_index,
    label_smoothing,
):
    with torch.no_grad():
        intent_prob, semantic_hidden = infer_intent_representation_v21(
            intent_model,
            intent_head,
            input_ids,
            prompt_index,
        )

    conditioning_bias = projection(intent_prob, semantic_hidden)
    hidden = forward_representation_v21_conditioned(
        generation_model,
        input_ids,
        conditioning_bias,
        projection.inject_after,
    )
    logits = generation_model.lm_head(hidden)

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(targets)

    return (token_loss * lm_mask).sum() / lm_mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    generation_model,
    intent_model,
    intent_head,
    projection,
    loader,
    device,
    label_smoothing,
):
    generation_model.eval()
    projection.eval()
    total = 0.0
    batches = 0

    for input_ids, targets, mask, prompt_index in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        prompt_index = prompt_index.to(device)

        loss = conditioned_loss(
            generation_model,
            intent_model,
            intent_head,
            projection,
            input_ids,
            targets,
            mask,
            prompt_index,
            label_smoothing,
        )
        total += float(loss.item())
        batches += 1

    return total / max(1, batches)


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (
        args.data,
        args.instruction_data,
        args.tokenizer,
        args.base_model,
        args.intent_head,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)

    intent_model, base_checkpoint = LanguageModel.load_checkpoint(
        args.base_model,
        device=device,
    )
    generation_model, _ = LanguageModel.load_checkpoint(
        args.base_model,
        device=device,
    )

    intent_head, head_checkpoint, labels = load_intent_head(
        args.intent_head,
        intent_model,
        device,
    )

    validate_injection_point(generation_model, args.inject_after)

    intent_model.eval()
    for p in intent_model.parameters():
        p.requires_grad_(False)
    intent_head.eval()
    for p in intent_head.parameters():
        p.requires_grad_(False)

    configure_partial_finetune(
        generation_model,
        inject_after=args.inject_after,
        unfreeze_lm_head=False,
    )

    projection = IntentRepresentationV21Projection(
        num_labels=len(labels),
        d_model=generation_model.d_model,
        semantic_dim=args.semantic_dim,
        alpha=args.alpha,
        inject_after=args.inject_after,
    ).to(device)

    conversation = parse_dialogues(
        Path(args.data).read_text(encoding="utf-8")
    )
    instruction = parse_instruction_pairs(
        Path(args.instruction_data).read_text(encoding="utf-8")
    )
    base_pairs = deduplicate_pairs(conversation + instruction)

    # Boundary v1 only: exact comparison to the 25/30 baseline.
    rows = augment_pairs(
        base_pairs,
        variants_per_intent=args.variants_per_intent,
        include_targeted_boundary=True,
        include_targeted_boundary_v2=False,
        include_protected_boundary_replay=False,
        include_balanced_control_replay=False,
    )

    train_rows, val_rows = stratified_split(
        rows,
        args.validation_ratio,
        SEED,
    )
    base_train_count = len(train_rows)

    train_rows = oversample(
        train_rows,
        args.technical_repeat,
        TECHNICAL_TAGS,
    )
    train_rows = oversample(
        train_rows,
        args.replay_repeat,
        REPLAY_TAGS,
    )

    train_set = PartialIntentDataset(
        train_rows,
        tokenizer,
        generation_model.context_length,
    )
    val_set = PartialIntentDataset(
        val_rows,
        tokenizer,
        generation_model.context_length,
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

    model_trainable = trainable_model_parameters(generation_model)

    optimizer = torch.optim.AdamW(
        [
            {"params": projection.parameters(), "lr": args.projection_lr},
            {"params": model_trainable, "lr": args.block_lr},
        ],
        weight_decay=0.01,
    )

    print()
    print("====================================================")
    print(" LLM_GPU v0.9 Intent Representation v2.1 Training")
    print("====================================================")
    print("Device                :", device)
    if device.type == "cuda":
        print("GPU                   :", torch.cuda.get_device_name(0))
    print("Base model loss       :", base_checkpoint.get("loss"))
    print("Intent-head loss      :", head_checkpoint.get("loss"))
    print("Intent labels         :", len(labels))
    print("Semantic hidden dim   :", generation_model.d_model)
    print("Semantic bottleneck   :", args.semantic_dim)
    print(
        "Fused representation :",
        len(labels) + args.semantic_dim,
    )
    print("Projection output dim :", generation_model.d_model)
    print("Inject after block    :", args.inject_after)
    print("Frozen blocks         :", f"1-{args.inject_after}")
    print(
        "Trainable blocks      :",
        f"{args.inject_after + 1}-{generation_model.num_layers}",
    )
    print("FinalNorm trainable   : True")
    print("LM Head trainable     : False")
    print("Boundary data         : v1 only")
    print("Projection alpha      :", args.alpha)
    print("Projection LR         :", args.projection_lr)
    print("Block/FinalNorm LR    :", args.block_lr)
    print(
        "Projection params     :",
        sum(p.numel() for p in projection.parameters()),
    )
    print(
        "Model trainable params:",
        trainable_parameter_count(generation_model),
    )
    print("Train rows (base)     :", base_train_count)
    print("Train rows (final)    :", len(train_rows))
    print("Validation rows       :", len(val_rows))
    print()

    best_val = float("inf")
    best_model_state = None
    best_projection_state = None
    best_epoch = 0
    bad_epochs = 0

    trainable_for_clip = list(projection.parameters()) + model_trainable

    for epoch in range(1, args.epochs + 1):
        generation_model.train()
        intent_model.eval()
        intent_head.eval()
        projection.train()

        total = 0.0
        batches = 0
        started = time.perf_counter()

        for input_ids, targets, mask, prompt_index in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            prompt_index = prompt_index.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = conditioned_loss(
                generation_model,
                intent_model,
                intent_head,
                projection,
                input_ids,
                targets,
                mask,
                prompt_index,
                args.label_smoothing,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_for_clip, 1.0)
            optimizer.step()

            total += float(loss.item())
            batches += 1

        train_loss = total / max(1, batches)
        val_loss = evaluate(
            generation_model,
            intent_model,
            intent_head,
            projection,
            val_loader,
            device,
            args.label_smoothing,
        )
        elapsed = time.perf_counter() - started

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train={train_loss:.4f} val={val_loss:.4f} "
            f"| {elapsed:.2f}s"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in generation_model.state_dict().items()
            }
            best_projection_state = {
                k: v.detach().cpu().clone()
                for k, v in projection.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_model_state is None or best_projection_state is None:
        raise RuntimeError("No valid Intent Representation v2.1 checkpoint.")

    generation_model.load_state_dict(best_model_state)
    projection.load_state_dict(best_projection_state)

    save_representation_v21_checkpoint(
        args.output,
        generation_model,
        projection,
        labels,
        epoch=best_epoch,
        loss=best_val,
        base_model=args.base_model,
        intent_head=args.intent_head,
        block_learning_rate=args.block_lr,
        projection_learning_rate=args.projection_lr,
    )

    print()
    print("Intent Representation v2.1 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best val LM loss :", f"{best_val:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
