# train_semantic_encoder_adapter_v01.py
#
# Train Semantic Encoder Adapter v0.1 on frozen v0.8 prompt hidden states.
#
# Loss:
#   concept CE
# + attribute BCE
# + preservation cosine loss
#
# No base-model parameters are updated.

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from augment_sft_v07 import classify_tags
from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v01 import (
    ATTRIBUTE_LABELS,
    CONCEPT_LABELS,
    SemanticEncoderAdapter,
    SemanticSupervisionHeads,
    save_semantic_adapter_checkpoint,
)
from semantic_probe_v09 import collect_reference_prompts
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9-semantic-adapter-v01.pt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Semantic Encoder Adapter v0.1."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--adapter-hidden", type=int, default=64)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--validation-ratio", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--concept-weight", type=float, default=1.0)
    p.add_argument("--attribute-weight", type=float, default=0.75)
    p.add_argument("--preservation-weight", type=float, default=0.25)
    return p.parse_args()


def attribute_target(label: str, prompt: str) -> torch.Tensor:
    target = torch.zeros(len(ATTRIBUTE_LABELS), dtype=torch.float32)
    tags = set(classify_tags(prompt))

    if label == "tech_gpu" or "property_parallel" in tags:
        target[ATTRIBUTE_LABELS.index("property_parallel")] = 1.0

    if label == "tech_cpu" or "property_general" in tags:
        target[ATTRIBUTE_LABELS.index("property_general")] = 1.0

    if label in {"tech_llm", "tech_python"} or "property_language" in tags:
        target[ATTRIBUTE_LABELS.index("property_language")] = 1.0

    if label == "tech_transformer":
        target[
            ATTRIBUTE_LABELS.index("property_attention_structure")
        ] = 1.0

    return target


def build_rows():
    refs = collect_reference_prompts()
    rows = []
    for label in CONCEPT_LABELS:
        for prompt in refs[label]:
            rows.append((
                prompt,
                CONCEPT_LABELS.index(label),
                attribute_target(label, prompt),
            ))
    return rows


def exact_overlap_with_dev(rows) -> List[str]:
    dev_prompts = {str(case["prompt"]) for case in CASES}
    return sorted({prompt for prompt, _concept, _attrs in rows if prompt in dev_prompts})


@torch.no_grad()
def encode_hidden(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
) -> torch.Tensor:
    device = next(model.parameters()).device
    text = f"{USER_PREFIX}{prompt}\n{AI_PREFIX}"
    ids = tokenizer.encode(text, add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = model.forward_hidden(x)
    return hidden[:, -1, :][0].detach().cpu()


class AdapterDataset(Dataset):
    def __init__(self, rows, model, tokenizer):
        self.items = []
        for prompt, concept, attrs in rows:
            hidden = encode_hidden(model, tokenizer, prompt)
            self.items.append((
                hidden,
                torch.tensor(concept, dtype=torch.long),
                attrs,
                prompt,
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        hidden, concept, attrs, _prompt = self.items[index]
        return hidden, concept, attrs


def stratified_split(rows, ratio: float, seed: int):
    groups: Dict[int, List[tuple]] = {}
    for row in rows:
        groups.setdefault(row[1], []).append(row)

    rng = random.Random(seed)
    train, val = [], []

    for concept in sorted(groups):
        items = list(groups[concept])
        rng.shuffle(items)
        count = max(1, int(round(len(items) * ratio)))
        count = min(count, len(items) - 2)
        val.extend(items[:count])
        train.extend(items[count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def batch_loss(
    adapter,
    heads,
    hidden,
    concept,
    attrs,
    concept_weight,
    attribute_weight,
    preservation_weight,
):
    adapted = adapter(hidden)
    concept_logits, attribute_logits = heads(adapted)

    concept_loss = F.cross_entropy(concept_logits, concept)
    attribute_loss = F.binary_cross_entropy_with_logits(
        attribute_logits,
        attrs,
    )

    preservation = 1.0 - F.cosine_similarity(
        adapted,
        hidden,
        dim=-1,
    ).mean()

    total = (
        concept_weight * concept_loss
        + attribute_weight * attribute_loss
        + preservation_weight * preservation
    )

    return total, concept_loss, attribute_loss, preservation


@torch.no_grad()
def evaluate(
    adapter,
    heads,
    loader,
    device,
    args,
):
    adapter.eval()
    heads.eval()

    total = concept_total = attr_total = preserve_total = 0.0
    concept_correct = 0
    concept_count = 0
    batches = 0

    for hidden, concept, attrs in loader:
        hidden = hidden.to(device)
        concept = concept.to(device)
        attrs = attrs.to(device)

        loss, c_loss, a_loss, p_loss = batch_loss(
            adapter,
            heads,
            hidden,
            concept,
            attrs,
            args.concept_weight,
            args.attribute_weight,
            args.preservation_weight,
        )

        adapted = adapter(hidden)
        concept_logits, attribute_logits = heads(adapted)

        total += float(loss.item())
        concept_total += float(c_loss.item())
        attr_total += float(a_loss.item())
        preserve_total += float(p_loss.item())
        concept_correct += int(
            (concept_logits.argmax(dim=-1) == concept).sum().item()
        )
        concept_count += int(concept.numel())
        batches += 1

    return {
        "loss": total / max(1, batches),
        "concept_loss": concept_total / max(1, batches),
        "attribute_loss": attr_total / max(1, batches),
        "preservation_loss": preserve_total / max(1, batches),
        "concept_acc": concept_correct / max(1, concept_count),
    }


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    base_model, checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    rows = build_rows()
    overlaps = exact_overlap_with_dev(rows)
    if overlaps:
        raise RuntimeError(
            "Exact overlap with fixed development prompts: "
            + repr(overlaps)
        )

    train_rows, val_rows = stratified_split(
        rows,
        args.validation_ratio,
        SEED,
    )

    train_set = AdapterDataset(train_rows, base_model, tokenizer)
    val_set = AdapterDataset(val_rows, base_model, tokenizer)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    adapter = SemanticEncoderAdapter(
        d_model=base_model.d_model,
        hidden_dim=args.adapter_hidden,
        residual_scale=args.residual_scale,
    ).to(device)
    heads = SemanticSupervisionHeads(
        d_model=base_model.d_model,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(heads.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.1 Training")
    print("====================================================")
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss   :", checkpoint.get("loss"))
    print("Base encoder           : frozen")
    print("d_model                :", base_model.d_model)
    print("Adapter                :", f"{base_model.d_model}->{args.adapter_hidden}->{base_model.d_model}")
    print("Residual scale         :", args.residual_scale)
    print("Concept classes        :", len(CONCEPT_LABELS))
    print("Attribute targets      :", len(ATTRIBUTE_LABELS))
    print("Train rows             :", len(train_rows))
    print("Validation rows        :", len(val_rows))
    print("Exact DEV overlap      :", len(overlaps))
    print("Learning rate          :", args.lr)
    print("Concept weight         :", args.concept_weight)
    print("Attribute weight       :", args.attribute_weight)
    print("Preservation weight    :", args.preservation_weight)
    print(
        "Adapter params         :",
        sum(p.numel() for p in adapter.parameters()),
    )
    print(
        "Supervision head params:",
        sum(p.numel() for p in heads.parameters()),
    )
    print()

    best_val = float("inf")
    best_epoch = 0
    best_adapter = None
    best_heads = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        heads.train()

        running = 0.0
        batches = 0

        for hidden, concept, attrs in train_loader:
            hidden = hidden.to(device)
            concept = concept.to(device)
            attrs = attrs.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, _c, _a, _p = batch_loss(
                adapter,
                heads,
                hidden,
                concept,
                attrs,
                args.concept_weight,
                args.attribute_weight,
                args.preservation_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters()) + list(heads.parameters()),
                1.0,
            )
            optimizer.step()

            running += float(loss.item())
            batches += 1

        train_loss = running / max(1, batches)
        metrics = evaluate(
            adapter,
            heads,
            val_loader,
            device,
            args,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| train={train_loss:.4f} "
            f"val={metrics['loss']:.4f} "
            f"concept_acc={metrics['concept_acc']:.1%} "
            f"attr={metrics['attribute_loss']:.4f} "
            f"preserve={metrics['preservation_loss']:.4f}"
        )

        if metrics["loss"] < best_val - 1e-5:
            best_val = metrics["loss"]
            best_epoch = epoch
            bad_epochs = 0
            best_adapter = {
                k: v.detach().cpu().clone()
                for k, v in adapter.state_dict().items()
            }
            best_heads = {
                k: v.detach().cpu().clone()
                for k, v in heads.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_adapter is None or best_heads is None:
        raise RuntimeError("No valid semantic adapter checkpoint.")

    adapter.load_state_dict(best_adapter)
    heads.load_state_dict(best_heads)

    save_semantic_adapter_checkpoint(
        args.output,
        adapter,
        heads,
        epoch=best_epoch,
        loss=best_val,
        base_model=args.model,
        learning_rate=args.lr,
        concept_weight=args.concept_weight,
        attribute_weight=args.attribute_weight,
        preservation_weight=args.preservation_weight,
    )

    print()
    print("Semantic Encoder Adapter v0.1 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best_val:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
