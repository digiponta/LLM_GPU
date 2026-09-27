# train_semantic_cross_attention_v091.py
#
# LLM_GPU v0.9.1
# Train Semantic-to-Generation Interface v0.9: Semantic Cross-Attention.

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augment_sft_v07 import augment_pairs
from mid_intent_conditioning_v09 import validate_injection_point
from model import LanguageModel
from semantic_cross_attention_v091 import (
    SemanticCrossAttention,
    configure_cross_attention_generation_model,
    forward_semantic_cross_attention,
    load_frozen_balanced_projector,
    save_cross_attention_checkpoint,
)
from semantic_generation_integration_v09 import (
    infer_semantic_condition,
    load_frozen_semantic_path,
)
from tokenizer_bpe import Tokenizer
from train_partial_intent_v09 import (
    PartialIntentDataset,
    deduplicate_pairs,
    parse_dialogues,
    parse_instruction_pairs,
    stratified_split,
)


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9-semantic-adapter-v05.pt"
DEFAULT_BALANCED = "model/model-gpu-v0.9.1-semantic-token-balanced-v08.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-semantic-cross-attn-v09.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION_DATA = "data/instruction-ja.txt"
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(
        description="Train v0.9.1 Semantic Cross-Attention."
    )
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--instruction-data", default=DEFAULT_INSTRUCTION_DATA)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--balanced", default=DEFAULT_BALANCED)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--inject-after", type=int, default=3)
    p.add_argument("--cross-lr", type=float, default=1e-3)
    p.add_argument("--block-lr", type=float, default=1e-5)
    p.add_argument("--cross-heads", type=int, default=8)
    p.add_argument("--initial-cross-scale", type=float, default=0.1)
    return p.parse_args()


def conditioned_loss(
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    balanced_projector,
    cross_attention,
    input_ids,
    targets,
    lm_mask,
    prompt_index,
    label_smoothing,
):
    with torch.no_grad():
        adapted, concept_prob, attribute_prob, hierarchy_prob = infer_semantic_condition(
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            input_ids,
            prompt_index,
        )
        semantic_tokens = balanced_projector(
            adapted,
            concept_prob,
            attribute_prob,
            hierarchy_prob,
        )

    hidden = forward_semantic_cross_attention(
        generation_model,
        cross_attention,
        input_ids,
        semantic_tokens,
        balanced_projector.inject_after,
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
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    balanced_projector,
    cross_attention,
    loader,
    device,
    label_smoothing,
):
    generation_model.eval()
    cross_attention.eval()

    total = 0.0
    batches = 0

    for input_ids, targets, mask, prompt_index in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        prompt_index = prompt_index.to(device)

        loss = conditioned_loss(
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            balanced_projector,
            cross_attention,
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
        args.semantic_adapter,
        args.balanced,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)

    (
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        base_checkpoint,
        semantic_checkpoint,
    ) = load_frozen_semantic_path(
        args.base_model,
        args.semantic_adapter,
        device,
    )

    balanced_projector, balanced_checkpoint = load_frozen_balanced_projector(
        args.balanced,
        device,
    )

    generation_model, _ = LanguageModel.load_checkpoint(
        args.base_model,
        device=device,
    )
    validate_injection_point(generation_model, args.inject_after)
    configure_cross_attention_generation_model(
        generation_model,
        args.inject_after,
    )

    cross_attention = SemanticCrossAttention(
        d_model=generation_model.d_model,
        num_heads=args.cross_heads,
        initial_residual_scale=args.initial_cross_scale,
    ).to(device)

    conversation = parse_dialogues(
        Path(args.data).read_text(encoding="utf-8")
    )
    instruction = parse_instruction_pairs(
        Path(args.instruction_data).read_text(encoding="utf-8")
    )
    base_pairs = deduplicate_pairs(conversation + instruction)

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

    late_model_params = [
        p for p in generation_model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": cross_attention.parameters(),
                "lr": args.cross_lr,
            },
            {
                "params": late_model_params,
                "lr": args.block_lr,
            },
        ],
        weight_decay=0.01,
    )

    print()
    print("====================================================")
    print(" Semantic-to-Generation Interface v0.9 Training")
    print("====================================================")
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Semantic adapter loss  :", semantic_checkpoint.get("loss"))
    print("Balanced token loss    :", balanced_checkpoint.get("loss"))
    print("Semantic path          : frozen")
    print("Balanced projector     : frozen")
    print("Cross-Attention        : Q=text, K/V=semantic tokens")
    print("Cross heads            :", args.cross_heads)
    print("Initial cross scale    :", args.initial_cross_scale)
    print("Insert after Block     :", args.inject_after)
    print("Blocks 1-3             : frozen")
    print("Blocks 4-6 + FinalNorm : trainable")
    print("LM Head                : frozen")
    print("Boundary v1            : enabled")
    print("Train rows             :", len(train_rows))
    print("Validation rows        :", len(val_rows))
    print("Cross LR               :", args.cross_lr)
    print("Block LR               :", args.block_lr)
    print(
        "Cross-Attn params      :",
        sum(p.numel() for p in cross_attention.parameters()),
    )
    print()

    best_val = float("inf")
    best_epoch = 0
    best_model = None
    best_cross = None
    bad_epochs = 0

    trainable = (
        list(cross_attention.parameters())
        + late_model_params
    )

    for epoch in range(1, args.epochs + 1):
        generation_model.train()
        cross_attention.train()

        running = 0.0
        batches = 0

        for input_ids, targets, mask, prompt_index in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            prompt_index = prompt_index.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = conditioned_loss(
                generation_model,
                semantic_model,
                semantic_adapter,
                semantic_heads,
                hierarchy_head,
                balanced_projector,
                cross_attention,
                input_ids,
                targets,
                mask,
                prompt_index,
                args.label_smoothing,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            running += float(loss.item())
            batches += 1

        train_loss = running / max(1, batches)
        val_loss = evaluate(
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            balanced_projector,
            cross_attention,
            val_loader,
            device,
            args.label_smoothing,
        )

        scale = float(cross_attention.residual_scale().detach().cpu().item())

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| train={train_loss:.4f} "
            f"val={val_loss:.4f} "
            f"cross_scale={scale:.4f}"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_model = {
                k: v.detach().cpu().clone()
                for k, v in generation_model.state_dict().items()
            }
            best_cross = {
                k: v.detach().cpu().clone()
                for k, v in cross_attention.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_model is None or best_cross is None:
        raise RuntimeError("No valid cross-attention checkpoint.")

    generation_model.load_state_dict(best_model)
    cross_attention.load_state_dict(best_cross)

    save_cross_attention_checkpoint(
        args.output,
        generation_model,
        cross_attention,
        epoch=best_epoch,
        loss=best_val,
        base_model=args.base_model,
        semantic_adapter_checkpoint=args.semantic_adapter,
        balanced_checkpoint=args.balanced,
        inject_after=args.inject_after,
        cross_learning_rate=args.cross_lr,
        block_learning_rate=args.block_lr,
    )

    print()
    print("Semantic Cross-Attention training completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best_val:.6f}")
    print(
        "Final cross scale:",
        float(cross_attention.residual_scale().detach().cpu().item()),
    )
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
