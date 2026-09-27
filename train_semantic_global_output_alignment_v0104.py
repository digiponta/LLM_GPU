# train_semantic_global_output_alignment_v0104.py
#
# v0.10.4 Semantic-to-Global-Output Alignment
#
# Controlled extension of v0.10.3:
#   1) target entity competes against the full vocabulary at answer start
#   2) first N answer tokens receive an additional continuation loss.

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
from semantic_aware_lm_head_v010 import (
    save_semantic_aware_lm_head_checkpoint,
)
from semantic_consistency_v09 import (
    SemanticConsistencyHead,
    gather_prompt_hidden,
    semantic_target,
)
from semantic_encoder_adapter_v01 import ATTRIBUTE_LABELS, CONCEPT_LABELS
from semantic_encoder_adapter_v08 import SEMANTIC_HIERARCHY_LABELS
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
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
DEFAULT_OUTPUT = "model/model-gpu-v0.10.4-semantic-global-output-aligned.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION_DATA = "data/instruction-ja.txt"
SEED = 42

ENTITY_TARGETS = {
    "tech_gpu": "GPU",
    "tech_cpu": "CPU",
    "tech_llm": "LLM",
    "tech_transformer": "Transformer",
    "tech_cuda": "CUDA",
    "tech_python": "Python",
}


def build_entity_token_ids(tokenizer):
    ids = []
    for label in CONCEPT_LABELS:
        text = ENTITY_TARGETS[label]
        encoded = tokenizer.encode(text, add_bos=False, add_eos=False)
        if not encoded:
            raise RuntimeError(f"Tokenizer produced no token for {text!r}")
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise RuntimeError(
            "Entity first-token IDs are not unique: "
            + str(dict(zip(CONCEPT_LABELS, ids)))
        )
    return ids


def semantic_output_alignment_loss(
    logits,
    prompt_index,
    concept_prob,
    entity_token_ids,
    margin,
    confidence_threshold,
):
    # v0.10.3 local entity competition: target entity vs other entity tokens.
    batch = torch.arange(logits.size(0), device=logits.device)
    first_logits = logits[batch, prompt_index.long(), :]
    entity_ids = torch.tensor(
        entity_token_ids, dtype=torch.long, device=logits.device
    )
    concept_logits = first_logits.index_select(1, entity_ids)

    target_idx = concept_prob.argmax(dim=-1)
    confidence = concept_prob.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    eligible = confidence >= confidence_threshold

    target_logit = concept_logits.gather(
        1, target_idx.unsqueeze(1)
    ).squeeze(1)
    masked = concept_logits.clone()
    masked.scatter_(1, target_idx.unsqueeze(1), float("-inf"))
    competitor = masked.max(dim=1).values

    per_row = F.relu(float(margin) - target_logit + competitor)
    if eligible.any():
        weights = confidence[eligible].detach()
        return (per_row[eligible] * weights).sum() / weights.sum().clamp_min(1e-6)
    return first_logits.sum() * 0.0


def semantic_global_alignment_loss(
    logits,
    prompt_index,
    concept_prob,
    entity_token_ids,
    margin,
    confidence_threshold,
):
    # v0.10.4 global competition: target entity vs strongest OTHER vocab token.
    batch = torch.arange(logits.size(0), device=logits.device)
    first_logits = logits[batch, prompt_index.long(), :]

    target_idx = concept_prob.argmax(dim=-1)
    confidence = concept_prob.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    eligible = confidence >= confidence_threshold

    entity_ids = torch.tensor(
        entity_token_ids, dtype=torch.long, device=logits.device
    )
    target_token_ids = entity_ids[target_idx]
    target_logit = first_logits.gather(
        1, target_token_ids.unsqueeze(1)
    ).squeeze(1)

    masked = first_logits.clone()
    masked.scatter_(1, target_token_ids.unsqueeze(1), float("-inf"))
    competitor = masked.max(dim=1).values

    per_row = F.relu(float(margin) - target_logit + competitor)
    if eligible.any():
        weights = confidence[eligible].detach()
        return (per_row[eligible] * weights).sum() / weights.sum().clamp_min(1e-6)
    return first_logits.sum() * 0.0


def continuation_alignment_loss(token_loss, lm_mask, prompt_index, tokens):
    # Extra LM pressure on the first N answer tokens. This extends alignment
    # beyond the first entity token without introducing hand-written phrases.
    rows = []
    for b in range(token_loss.size(0)):
        start = int(prompt_index[b].item())
        stop = min(token_loss.size(1), start + int(tokens))
        if stop <= start:
            continue
        local_mask = lm_mask[b, start:stop]
        denom = local_mask.sum()
        if float(denom.item()) > 0.0:
            rows.append(
                (token_loss[b, start:stop] * local_mask).sum()
                / denom.clamp_min(1.0)
            )
    if rows:
        return torch.stack(rows).mean()
    return token_loss.sum() * 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Train Semantic-to-Global-Output Alignment v0.10.4.")
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
    p.add_argument("--consistency-lr", type=float, default=1e-3)
    p.add_argument("--lm-head-lr", type=float, default=3e-6)
    p.add_argument("--consistency-weight", type=float, default=0.50)
    p.add_argument("--alignment-weight", type=float, default=0.05)
    p.add_argument("--alignment-margin", type=float, default=0.75)
    p.add_argument("--alignment-confidence", type=float, default=0.70)
    p.add_argument("--global-alignment-weight", type=float, default=0.05)
    p.add_argument("--global-alignment-margin", type=float, default=0.50)
    p.add_argument("--continuation-weight", type=float, default=0.05)
    p.add_argument("--continuation-tokens", type=int, default=4)
    return p.parse_args()


def losses(
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    projection,
    consistency_head,
    input_ids,
    targets,
    lm_mask,
    prompt_index,
    label_smoothing,
    consistency_weight,
    entity_token_ids,
    alignment_weight,
    alignment_margin,
    alignment_confidence,
    global_alignment_weight,
    global_alignment_margin,
    continuation_weight,
    continuation_tokens,
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
        target_sem = semantic_target(
            concept_prob, attribute_prob, hierarchy_prob
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
    lm_loss = (token_loss * lm_mask).sum() / lm_mask.sum().clamp_min(1.0)

    prompt_hidden = gather_prompt_hidden(hidden, prompt_index)
    predicted_sem_logits = consistency_head(prompt_hidden)
    consistency_loss = F.binary_cross_entropy_with_logits(
        predicted_sem_logits,
        target_sem,
    )

    alignment_loss = semantic_output_alignment_loss(
        logits,
        prompt_index,
        concept_prob,
        entity_token_ids,
        alignment_margin,
        alignment_confidence,
    )
    global_alignment_loss = semantic_global_alignment_loss(
        logits,
        prompt_index,
        concept_prob,
        entity_token_ids,
        global_alignment_margin,
        alignment_confidence,
    )
    continuation_loss = continuation_alignment_loss(
        token_loss,
        lm_mask,
        prompt_index,
        continuation_tokens,
    )

    total = (
        lm_loss
        + consistency_weight * consistency_loss
        + alignment_weight * alignment_loss
        + global_alignment_weight * global_alignment_loss
        + continuation_weight * continuation_loss
    )
    return (
        total,
        lm_loss,
        consistency_loss,
        alignment_loss,
        global_alignment_loss,
        continuation_loss,
    )


@torch.no_grad()
def evaluate(
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    projection,
    consistency_head,
    loader,
    device,
    label_smoothing,
    consistency_weight,
    entity_token_ids,
    alignment_weight,
    alignment_margin,
    alignment_confidence,
):
    generation_model.eval()
    projection.eval()
    consistency_head.eval()
    total = lm = sem = align = global_align = continuation = 0.0
    batches = 0

    for input_ids, targets, mask, prompt_index in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        prompt_index = prompt_index.to(device)

        loss, lm_loss, sem_loss, align_loss, global_loss, cont_loss = losses(
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            projection,
            consistency_head,
            input_ids,
            targets,
            mask,
            prompt_index,
            label_smoothing,
            consistency_weight,
            entity_token_ids,
            alignment_weight,
            alignment_margin,
            alignment_confidence,
            global_alignment_weight,
            global_alignment_margin,
            continuation_weight,
            continuation_tokens,
        )
        total += float(loss.item())
        lm += float(lm_loss.item())
        sem += float(sem_loss.item())
        align += float(align_loss.item())
        global_align += float(global_loss.item())
        continuation += float(cont_loss.item())
        batches += 1

    n = max(1, batches)
    return total/n, lm/n, sem/n, align/n, global_align/n, continuation/n


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
    entity_token_ids = build_entity_token_ids(tokenizer)

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

    # Controlled v0.10 change: unfreeze ONLY the LM head.
    for p in generation_model.lm_head.parameters():
        p.requires_grad_(True)

    projection = SemanticGenerationProjection(
        semantic_dim=generation_model.d_model,
        concept_dim=len(CONCEPT_LABELS),
        attribute_dim=len(ATTRIBUTE_LABELS),
        hierarchy_dim=len(SEMANTIC_HIERARCHY_LABELS),
        d_model=generation_model.d_model,
        alpha=args.alpha,
        inject_after=args.inject_after,
    ).to(device)

    consistency_head = SemanticConsistencyHead(
        d_model=generation_model.d_model,
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

    late_block_params = []
    for index, block in enumerate(generation_model.blocks, start=1):
        if index > args.inject_after:
            late_block_params.extend(
                p for p in block.parameters() if p.requires_grad
            )
    late_block_params.extend(
        p for p in generation_model.final_norm.parameters() if p.requires_grad
    )
    lm_head_params = [
        p for p in generation_model.lm_head.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": projection.parameters(), "lr": args.projection_lr},
            {"params": late_block_params, "lr": args.block_lr},
            {"params": consistency_head.parameters(), "lr": args.consistency_lr},
            {"params": lm_head_params, "lr": args.lm_head_lr},
        ],
        weight_decay=0.01,
    )
    trainable = (
        list(projection.parameters())
        + late_block_params
        + list(consistency_head.parameters())
        + lm_head_params
    )

    print()
    print("====================================================")
    print(" Semantic-to-Global-Output Alignment v0.10.4")
    print("====================================================")
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Semantic adapter loss  :", semantic_checkpoint.get("loss"))
    print("Semantic teacher       : v0.8 constrained, frozen")
    print("Semantic target        : 6 concept + 4 attribute + 15 hierarchy = 25")
    print("Projection             : 281 -> 256")
    print("Consistency head       : 256 -> 25")
    print("Consistency weight     :", args.consistency_weight)
    print("Alignment objective    : semantic concept -> first entity token")
    print("Entity alignment weight:", args.alignment_weight)
    print("Entity alignment margin:", args.alignment_margin)
    print("Alignment confidence   :", args.alignment_confidence)
    print("Global alignment weight:", args.global_alignment_weight)
    print("Global alignment margin:", args.global_alignment_margin)
    print("Continuation weight    :", args.continuation_weight)
    print("Continuation tokens    :", args.continuation_tokens)
    print(
        "Entity token IDs       :",
        dict(zip(CONCEPT_LABELS, entity_token_ids)),
    )
    print("Inject after           :", args.inject_after)
    print("Blocks 1-3             : frozen")
    print("Blocks 4-6 + FinalNorm : trainable")
    print("LM Head                : trainable")
    print("Projection LR          :", args.projection_lr)
    print("Block LR               :", args.block_lr)
    print("Consistency LR         :", args.consistency_lr)
    print("LM Head LR             :", args.lm_head_lr)
    print("Train rows             :", len(train_rows))
    print("Validation rows        :", len(val_rows))
    print()

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        generation_model.train()
        projection.train()
        consistency_head.train()
        total = lm_total = sem_total = align_total = global_total = cont_total = 0.0
        batches = 0

        for input_ids, targets, mask, prompt_index in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            prompt_index = prompt_index.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, lm_loss, sem_loss, align_loss, global_loss, cont_loss = losses(
                generation_model,
                semantic_model,
                semantic_adapter,
                semantic_heads,
                hierarchy_head,
                projection,
                consistency_head,
                input_ids,
                targets,
                mask,
                prompt_index,
                args.label_smoothing,
                args.consistency_weight,
                entity_token_ids,
                args.alignment_weight,
                args.alignment_margin,
                args.alignment_confidence,
                args.global_alignment_weight,
                args.global_alignment_margin,
                args.continuation_weight,
                args.continuation_tokens,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total += float(loss.item())
            lm_total += float(lm_loss.item())
            sem_total += float(sem_loss.item())
            align_total += float(align_loss.item())
            global_total += float(global_loss.item())
            cont_total += float(cont_loss.item())
            batches += 1

        n = max(1, batches)
        val_total, val_lm, val_sem, val_align, val_global, val_cont = evaluate(
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            projection,
            consistency_head,
            val_loader,
            device,
            args.label_smoothing,
            args.consistency_weight,
            entity_token_ids,
            args.alignment_weight,
            args.alignment_margin,
            args.alignment_confidence,
            args.global_alignment_weight,
            args.global_alignment_margin,
            args.continuation_weight,
            args.continuation_tokens,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| train={total/n:.4f} lm={lm_total/n:.4f} "
            f"sem={sem_total/n:.4f} ent={align_total/n:.4f} "
            f"glob={global_total/n:.4f} cont={cont_total/n:.4f} "
            f"| val={val_total:.4f} lm={val_lm:.4f} sem={val_sem:.4f} "
            f"ent={val_align:.4f} glob={val_global:.4f} cont={val_cont:.4f}"
        )

        if val_total < best_val - 1e-5:
            best_val = val_total
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                "model": {
                    k: v.detach().cpu().clone()
                    for k, v in generation_model.state_dict().items()
                },
                "projection": {
                    k: v.detach().cpu().clone()
                    for k, v in projection.state_dict().items()
                },
                "consistency": {
                    k: v.detach().cpu().clone()
                    for k, v in consistency_head.state_dict().items()
                },
                "lm_loss": val_lm,
                "sem_loss": val_sem,
                "align_loss": val_align,
                "global_loss": val_global,
                "cont_loss": val_cont,
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    generation_model.load_state_dict(best_state["model"])
    projection.load_state_dict(best_state["projection"])
    consistency_head.load_state_dict(best_state["consistency"])

    save_semantic_aware_lm_head_checkpoint(
        args.output,
        generation_model,
        projection,
        consistency_head,
        epoch=best_epoch,
        loss=best_val,
        lm_loss=best_state["lm_loss"],
        consistency_loss=best_state["sem_loss"],
        semantic_adapter_checkpoint=args.semantic_adapter,
        base_model=args.base_model,
        consistency_weight=args.consistency_weight,
        block_learning_rate=args.block_lr,
        projection_learning_rate=args.projection_lr,
        consistency_learning_rate=args.consistency_lr,
        lm_head_learning_rate=args.lm_head_lr,
    )

    print()
    print("Semantic-to-Global-Output Alignment v0.10.4 training completed.")
    print("Best epoch        :", best_epoch)
    print("Best val loss     :", f"{best_val:.6f}")
    print("Best val LM       :", f"{best_state['lm_loss']:.6f}")
    print("Best val semantic :", f"{best_state['sem_loss']:.6f}")
    print("Best val entity    :", f"{best_state['align_loss']:.6f}")
    print("Best val global    :", f"{best_state['global_loss']:.6f}")
    print("Best val continue  :", f"{best_state['cont_loss']:.6f}")
    print("Checkpoint saved  :", args.output)


if __name__ == "__main__":
    main()
