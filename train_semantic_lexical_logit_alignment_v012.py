# train_semantic_lexical_logit_alignment_v012.py
#
# v0.12 Semantic / Lexical -> Logit Alignment
#
# v0.11 generation model/projection are frozen.
# Only a small 345 -> rank -> vocab adapter trains.
# The direct bias affects only the first assistant token.

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augment_sft_v07 import augment_pairs
from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from semantic_encoder_adapter_v08 import SEMANTIC_HIERARCHY_LABELS
from semantic_encoder_adapter_v01 import CONCEPT_LABELS, ATTRIBUTE_LABELS
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
from semantic_generation_integration_v09 import (
    forward_semantic_conditioned,
    infer_semantic_condition,
)
from semantic_lexical_logit_alignment_v012 import (
    SemanticLexicalLogitAdapter,
    load_frozen_v011,
    save_checkpoint,
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
DEFAULT_NAME_BINDING = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
DEFAULT_V011 = "model/model-gpu-v0.9.1-lexical-generation-v011.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-semantic-lexical-logit-v012.pt"
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
    p.add_argument("--name-binding", default=DEFAULT_NAME_BINDING)
    p.add_argument("--v011-checkpoint", default=DEFAULT_V011)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.10)
    p.add_argument("--l2-weight", type=float, default=1e-4)
    return p.parse_args()


def condition_parts(
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    input_ids,
    prompt_index,
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
        lexical_identity = name_binding(adapted)
        condition = torch.cat(
            [
                adapted,
                concept_prob,
                attribute_prob,
                hierarchy_prob,
                lexical_identity,
            ],
            dim=-1,
        )
    return (
        condition,
        adapted,
        concept_prob,
        attribute_prob,
        hierarchy_prob,
        lexical_identity,
    )


def first_answer_targets(targets, lm_mask):
    # Find the first supervised assistant position in every row.
    active = lm_mask > 0
    first = active & (active.cumsum(dim=1) == 1)
    if not torch.all(first.sum(dim=1) == 1):
        raise RuntimeError("Every row must contain exactly one first answer token.")
    positions = first.float().argmax(dim=1)
    batch = torch.arange(targets.size(0), device=targets.device)
    return targets[batch, positions]


def batch_loss(
    adapter,
    generation_model,
    v011_projection,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    input_ids,
    targets,
    lm_mask,
    prompt_index,
    l2_weight,
):
    (
        condition,
        adapted,
        concept_prob,
        attribute_prob,
        hierarchy_prob,
        lexical_identity,
    ) = condition_parts(
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        name_binding,
        input_ids,
        prompt_index,
    )

    with torch.no_grad():
        semantic_bias = v011_projection(
            adapted,
            concept_prob,
            attribute_prob,
            hierarchy_prob,
            lexical_identity,
        )
        hidden = forward_semantic_conditioned(
            generation_model,
            input_ids,
            semantic_bias,
            v011_projection.inject_after,
        )
        base_logits = generation_model.lm_head(hidden)

    active = lm_mask > 0
    first = active & (active.cumsum(dim=1) == 1)
    if not torch.all(first.sum(dim=1) == 1):
        raise RuntimeError("Every row must contain exactly one first answer token.")
    positions = first.float().argmax(dim=1)
    batch = torch.arange(targets.size(0), device=targets.device)
    target = targets[batch, positions]
    frozen_first_logits = base_logits[batch, positions, :]

    bias = adapter(condition)
    combined_logits = frozen_first_logits + bias
    ce = F.cross_entropy(combined_logits, target)
    reg = bias.pow(2).mean()
    total = ce + l2_weight * reg
    return total, ce, reg


@torch.no_grad()
def evaluate(
    adapter,
    generation_model,
    v011_projection,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    loader,
    device,
    l2_weight,
):
    adapter.eval()
    total = ce_total = reg_total = 0.0
    batches = 0
    for input_ids, targets, lm_mask, prompt_index in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        lm_mask = lm_mask.to(device)
        prompt_index = prompt_index.to(device)
        loss, ce, reg = batch_loss(
            adapter,
            generation_model,
            v011_projection,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            input_ids,
            targets,
            lm_mask,
            prompt_index,
            l2_weight,
        )
        total += float(loss.item())
        ce_total += float(ce.item())
        reg_total += float(reg.item())
        batches += 1

    n = max(1, batches)
    return total/n, ce_total/n, reg_total/n


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
        args.name_binding,
        args.v011_checkpoint,
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
        args.base_model,
        args.semantic_adapter,
        device,
    )
    name_binding, name_checkpoint = load_cpu_name_binding_checkpoint(
        args.name_binding,
        device,
    )
    name_binding.eval()
    for p in name_binding.parameters():
        p.requires_grad_(False)

    generation_model, v011_projection, v011_checkpoint = load_frozen_v011(
        args.v011_checkpoint,
        device,
    )

    expected_dim = (
        generation_model.d_model
        + len(CONCEPT_LABELS)
        + len(ATTRIBUTE_LABELS)
        + len(SEMANTIC_HIERARCHY_LABELS)
        + name_binding.binding_dim
    )
    if expected_dim != v011_projection.input_dim:
        raise RuntimeError(
            f"Condition dim mismatch: expected {expected_dim}, "
            f"v0.11 has {v011_projection.input_dim}"
        )

    adapter = SemanticLexicalLogitAdapter(
        input_dim=expected_dim,
        vocab_size=generation_model.vocab_size,
        rank=args.rank,
        beta=args.beta,
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
    train_rows, val_rows = stratified_split(rows, args.validation_ratio, SEED)

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

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    print()
    print("====================================================")
    print(" Semantic/Lexical -> Logit Alignment v0.12")
    print("====================================================")
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Semantic adapter loss  :", semantic_checkpoint.get("loss"))
    print("Name binding loss      :", name_checkpoint.get("loss"))
    print("v0.11 integration loss :", v011_checkpoint.get("loss"))
    print("v0.11 model/projection : frozen")
    print("Semantic path          : frozen")
    print("Name binding           : frozen")
    print("Condition dim          :", expected_dim)
    print("Logit adapter          :", f"{expected_dim} -> {args.rank} -> {generation_model.vocab_size}")
    print("Beta                   :", args.beta)
    print("Application            : first assistant token only")
    print("Training logits        : frozen v0.11 logits + direct bias")
    print("LM Head                : frozen")
    print("Train rows             :", len(train_rows))
    print("Validation rows        :", len(val_rows))
    print()

    best = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        total = ce_total = reg_total = 0.0
        batches = 0

        for input_ids, targets, lm_mask, prompt_index in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            lm_mask = lm_mask.to(device)
            prompt_index = prompt_index.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, ce, reg = batch_loss(
                adapter,
                generation_model,
                v011_projection,
                semantic_model,
                semantic_adapter,
                semantic_heads,
                hierarchy_head,
                name_binding,
                input_ids,
                targets,
                lm_mask,
                prompt_index,
                args.l2_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            total += float(loss.item())
            ce_total += float(ce.item())
            reg_total += float(reg.item())
            batches += 1

        n = max(1, batches)
        val, val_ce, val_reg = evaluate(
            adapter,
            generation_model,
            v011_projection,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            val_loader,
            device,
            args.l2_weight,
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| train={total/n:.4f} ce={ce_total/n:.4f} reg={reg_total/n:.6f} "
            f"| val={val:.4f} ce={val_ce:.4f} reg={val_reg:.6f}"
        )

        if val < best - 1e-5:
            best = val
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in adapter.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    adapter.load_state_dict(best_state)
    save_checkpoint(
        args.output,
        adapter,
        epoch=best_epoch,
        loss=best,
        v011_checkpoint=args.v011_checkpoint,
        semantic_adapter_checkpoint=args.semantic_adapter,
        name_binding_checkpoint=args.name_binding,
        learning_rate=args.lr,
        l2_weight=args.l2_weight,
    )

    print()
    print("Semantic/Lexical Logit Alignment v0.12 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
