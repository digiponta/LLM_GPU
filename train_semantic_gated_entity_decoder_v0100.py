# train_semantic_gated_entity_decoder_v0100.py
#
# Train only the v0.10.0 answer-mode gate + entity decoder.
# v0.9.9 LM and semantic head remain frozen.

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from augment_sft_v07 import augment_pairs
from dynamic_top_competitor_v099 import load_checkpoint as load_v099
from entity_logit_alignment_v098 import ENTITY_BY_TAG
from evaluate_partial_intent_v09 import CASES
from semantic_gated_entity_decoder_v0100 import (
    ENTITY_TAGS,
    SemanticGatedEntityDecoder,
    save_checkpoint,
)
from tokenizer_bpe import Tokenizer
from train_sft_v09 import (
    AI_PREFIX,
    USER_PREFIX,
    deduplicate_pairs,
    parse_dialogues,
    parse_instruction_pairs,
    stratified_split,
)


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_SOURCE = "model/model-gpu-v0.9.9-dynamic-top-competitor.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.10.0-semantic-gated-entity-decoder.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION = "data/instruction-ja.txt"
SEED = 42


def single_entity_tag(tags):
    found = [tag for tag in ENTITY_TAGS if tag in set(tags)]
    return found[0] if len(found) == 1 else None


def entity_answer_target(answer, tags, tokenizer):
    tag = single_entity_tag(tags)
    if tag is None:
        return 0, -1

    entity_ids = tokenizer.encode(ENTITY_BY_TAG[tag])
    answer_ids = tokenizer.encode(answer)
    if not entity_ids or not answer_ids:
        return 0, -1
    if answer_ids[0] != entity_ids[0]:
        return 0, -1

    return 1, ENTITY_TAGS.index(tag)


class GateDataset(Dataset):
    def __init__(self, rows, tokenizer, model):
        self.rows = []
        self.entity_count = 0

        device = next(model.parameters()).device
        model.eval()

        with torch.no_grad():
            for prompt_text, answer_text, tags in rows:
                prompt = f"{USER_PREFIX}{prompt_text}\n{AI_PREFIX}"
                ids = tokenizer.encode(prompt, add_bos=True)
                ids = ids[-model.context_length:]
                x = torch.tensor([ids], dtype=torch.long, device=device)
                hidden = model.forward_hidden(x)[:, -1, :][0].cpu()

                gate_target, entity_target = entity_answer_target(
                    answer_text, tags, tokenizer
                )
                self.entity_count += gate_target
                self.rows.append((
                    hidden,
                    torch.tensor(float(gate_target), dtype=torch.float32),
                    torch.tensor(entity_target, dtype=torch.long),
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
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--entity-weight", type=float, default=1.0)
    p.add_argument("--gate-threshold", type=float, default=0.50)
    p.add_argument("--variants-per-intent", type=int, default=24)
    return p.parse_args()


def loss_fn(decoder, hidden, gate_target, entity_target, entity_weight):
    gate_logit, entity_logits = decoder(hidden)
    gate_loss = F.binary_cross_entropy_with_logits(gate_logit, gate_target)

    mask = entity_target >= 0
    if bool(mask.any()):
        entity_loss = F.cross_entropy(
            entity_logits[mask], entity_target[mask]
        )
        entity_acc = (
            entity_logits[mask].argmax(dim=-1) == entity_target[mask]
        ).float().mean()
    else:
        entity_loss = entity_logits.sum() * 0.0
        entity_acc = entity_logits.sum() * 0.0

    total = gate_loss + entity_weight * entity_loss
    gate_pred = (torch.sigmoid(gate_logit) >= 0.5).float()
    gate_acc = (gate_pred == gate_target).float().mean()
    return total, gate_loss, entity_loss, gate_acc, entity_acc


@torch.no_grad()
def evaluate(decoder, loader, device, entity_weight):
    decoder.eval()
    sums = [0.0] * 5
    batches = 0
    for hidden, gate_target, entity_target in loader:
        values = loss_fn(
            decoder,
            hidden.to(device),
            gate_target.to(device),
            entity_target.to(device),
            entity_weight,
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
        args.tokenizer, args.source, args.data, args.instruction_data
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, semantic_head, labels, source_ckpt = load_v099(args.source, device)

    model.eval()
    semantic_head.eval()
    for module in (model, semantic_head):
        for p in module.parameters():
            p.requires_grad_(False)

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

    train_rows, val_rows = stratified_split(
        rows, args.validation_ratio, SEED
    )

    train_set = GateDataset(train_rows, tokenizer, model)
    val_set = GateDataset(val_rows, tokenizer, model)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    decoder = SemanticGatedEntityDecoder(
        d_model=model.d_model,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    print()
    print("======================================")
    print(" v0.10.0 Semantic-Gated Entity Decoder")
    print("======================================")
    print("Device              :", device)
    print("Source              :", args.source)
    print("Source loss         :", source_ckpt.get("loss"))
    print("LM                  : frozen")
    print("Semantic head       : frozen")
    print("Decoder             : trainable")
    print("Hidden path         :", f"{model.d_model} -> {args.hidden_dim}")
    print("Entity candidates   :", len(ENTITY_TAGS))
    print("Exact fixed prompts : excluded")
    print("Excluded rows       :", excluded)
    print("Train rows          :", len(train_set))
    print("Train entity rows   :", train_set.entity_count)
    print("Validation rows     :", len(val_set))
    print("Validation entity   :", val_set.entity_count)
    print("Gate threshold      :", args.gate_threshold)
    print()

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        sums = [0.0] * 5
        batches = 0
        started = time.perf_counter()

        for hidden, gate_target, entity_target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            values = loss_fn(
                decoder,
                hidden.to(device),
                gate_target.to(device),
                entity_target.to(device),
                args.entity_weight,
            )
            values[0].backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            for i, value in enumerate(values):
                sums[i] += float(value.item())
            batches += 1

        n = max(1, batches)
        train_values = [v / n for v in sums]
        val_values = evaluate(
            decoder, val_loader, device, args.entity_weight
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train={train_values[0]:.4f} "
            f"gate={train_values[1]:.4f} ent={train_values[2]:.4f} "
            f"gate_acc={train_values[3]:.3f} ent_acc={train_values[4]:.3f} "
            f"| val={val_values[0]:.4f} "
            f"gate={val_values[1]:.4f} ent={val_values[2]:.4f} "
            f"gate_acc={val_values[3]:.3f} ent_acc={val_values[4]:.3f} "
            f"| {time.perf_counter()-started:.2f}s"
        )

        if val_values[0] < best - 1e-5:
            best = val_values[0]
            best_epoch = epoch
            bad = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in decoder.state_dict().items()
            }
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("No valid v0.10.0 decoder checkpoint.")

    decoder.load_state_dict(best_state)
    save_checkpoint(
        args.output,
        decoder,
        epoch=best_epoch,
        loss=best,
        source_checkpoint=args.source,
        gate_threshold=args.gate_threshold,
    )
    print()
    print("Completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
