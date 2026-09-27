# train_semantic_generation_integration_v08.py
#
# Controlled experiment:
# v0.8 Hierarchy-Constrained Semantic Head -> original v0.9 additive integration.

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
from semantic_encoder_adapter_v01 import ATTRIBUTE_LABELS, CONCEPT_LABELS
from semantic_encoder_adapter_v08 import SEMANTIC_HIERARCHY_LABELS
from semantic_generation_integration_v08 import (
    load_frozen_semantic_path_v08,
    save_integration_checkpoint_v08,
)
from semantic_generation_integration_v09 import (
    SemanticGenerationProjection,
    configure_generation_model,
    forward_semantic_conditioned,
    infer_semantic_condition,
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
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-semantic-generation-v08.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION_DATA = "data/instruction-ja.txt"
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--instruction-data", default=DEFAULT_INSTRUCTION_DATA)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--inject-after", type=int, default=3)
    p.add_argument("--projection-lr", type=float, default=1e-3)
    p.add_argument("--block-lr", type=float, default=1e-5)
    return p.parse_args()


def conditioned_loss(
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    projection,
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

    semantic_bias = projection(
        adapted, concept_prob, attribute_prob, hierarchy_prob
    )
    hidden = forward_semantic_conditioned(
        generation_model,
        input_ids,
        semantic_bias,
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
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
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
            generation_model, semantic_model, semantic_adapter,
            semantic_heads, hierarchy_head, projection,
            input_ids, targets, mask, prompt_index, label_smoothing,
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
    ) = load_frozen_semantic_path_v08(
        args.base_model, args.semantic_adapter, device
    )

    generation_model, _ = LanguageModel.load_checkpoint(
        args.base_model, device=device
    )
    validate_injection_point(generation_model, args.inject_after)
    configure_generation_model(generation_model, args.inject_after)

    projection = SemanticGenerationProjection(
        semantic_dim=generation_model.d_model,
        concept_dim=len(CONCEPT_LABELS),
        attribute_dim=len(ATTRIBUTE_LABELS),
        hierarchy_dim=len(SEMANTIC_HIERARCHY_LABELS),
        d_model=generation_model.d_model,
        alpha=args.alpha,
        inject_after=args.inject_after,
    ).to(device)

    conversation = parse_dialogues(Path(args.data).read_text(encoding="utf-8"))
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
        rows, args.validation_ratio, SEED
    )

    train_set = PartialIntentDataset(
        train_rows, tokenizer, generation_model.context_length
    )
    val_set = PartialIntentDataset(
        val_rows, tokenizer, generation_model.context_length
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
        p for p in generation_model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": projection.parameters(), "lr": args.projection_lr},
            {"params": late_model_params, "lr": args.block_lr},
        ],
        weight_decay=0.01,
    )

    print()
    print("====================================================")
    print(" Semantic v0.8 -> v0.9 Generation Integration")
    print("====================================================")
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Semantic adapter loss  :", semantic_checkpoint.get("loss"))
    print("Semantic path          : frozen")
    print(
        "Condition representation:",
        f"256 semantic + {len(CONCEPT_LABELS)} concept + "
        f"{len(ATTRIBUTE_LABELS)} attribute + "
        f"{len(SEMANTIC_HIERARCHY_LABELS)} hierarchy "
        f"= {projection.input_dim}",
    )
    print("Projection             :", f"{projection.input_dim} -> 256")
    print("Alpha                  :", args.alpha)
    print("Inject after           :", args.inject_after)
    print("Blocks 1-3             : frozen")
    print("Blocks 4-6 + FinalNorm : trainable")
    print("LM Head                : frozen")
    print("Boundary v1            : enabled")
    print("Train rows             :", len(train_rows))
    print("Validation rows        :", len(val_rows))
    print()

    best_val = float("inf")
    best_epoch = 0
    best_model = None
    best_projection = None
    bad_epochs = 0
    trainable = list(projection.parameters()) + late_model_params

    for epoch in range(1, args.epochs + 1):
        generation_model.train()
        projection.train()
        running = 0.0
        batches = 0

        for input_ids, targets, mask, prompt_index in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            prompt_index = prompt_index.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = conditioned_loss(
                generation_model, semantic_model, semantic_adapter,
                semantic_heads, hierarchy_head, projection,
                input_ids, targets, mask, prompt_index,
                args.label_smoothing,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            running += float(loss.item())
            batches += 1

        train_loss = running / max(1, batches)
        val_loss = evaluate(
            generation_model, semantic_model, semantic_adapter,
            semantic_heads, hierarchy_head, projection,
            val_loader, device, args.label_smoothing,
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| train={train_loss:.4f} val={val_loss:.4f}"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_model = {
                k: v.detach().cpu().clone()
                for k, v in generation_model.state_dict().items()
            }
            best_projection = {
                k: v.detach().cpu().clone()
                for k, v in projection.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    generation_model.load_state_dict(best_model)
    projection.load_state_dict(best_projection)

    save_integration_checkpoint_v08(
        args.output,
        generation_model,
        projection,
        epoch=best_epoch,
        loss=best_val,
        base_model=args.base_model,
        semantic_adapter_checkpoint=args.semantic_adapter,
        block_learning_rate=args.block_lr,
        projection_learning_rate=args.projection_lr,
    )

    print()
    print("v0.8 constrained semantic generation integration completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best_val:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
