# train_semantic_encoder_adapter_v02.py
#
# Semantic Encoder Adapter v0.2: Margin-Aware Contrastive Binding.
#
# Continues from the v0.1 adapter by default and adds a centroid margin loss:
#
#   sim(sample, positive centroid)
#     >= max sim(sample, negative centroids) + margin
#
# The base v0.8 encoder remains frozen.

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import LanguageModel
from semantic_encoder_adapter_v01 import load_semantic_adapter_checkpoint
from semantic_encoder_adapter_v02 import (
    CONCEPT_LABELS,
    save_semantic_adapter_v02_checkpoint,
)
from tokenizer_bpe import Tokenizer
from train_semantic_encoder_adapter_v01 import (
    AdapterDataset,
    build_rows,
    exact_overlap_with_dev,
    stratified_split,
)


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INIT = "model/model-gpu-v0.9-semantic-adapter-v01.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9-semantic-adapter-v02.pt"
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Semantic Encoder Adapter v0.2."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--init-adapter", default=DEFAULT_INIT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--validation-ratio", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--concept-weight", type=float, default=1.0)
    p.add_argument("--attribute-weight", type=float, default=0.75)
    p.add_argument("--margin-weight", type=float, default=0.50)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--preservation-weight", type=float, default=0.25)
    return p.parse_args()


def raw_centroids(dataset, device):
    sums = [None] * len(CONCEPT_LABELS)
    counts = [0] * len(CONCEPT_LABELS)

    for hidden, concept, _attrs, _prompt in dataset.items:
        idx = int(concept.item())
        h = hidden.to(device)
        sums[idx] = h.clone() if sums[idx] is None else sums[idx] + h
        counts[idx] += 1

    centroids = []
    for idx in range(len(CONCEPT_LABELS)):
        if counts[idx] == 0:
            raise RuntimeError(f"No training rows for {CONCEPT_LABELS[idx]}")
        centroids.append(sums[idx] / counts[idx])

    return torch.stack(centroids, dim=0)


def centroid_margin_loss(adapter, hidden, concept, centroid_hidden, margin):
    adapted = F.normalize(adapter(hidden), dim=-1)
    adapted_centroids = F.normalize(adapter(centroid_hidden), dim=-1)
    sim = adapted @ adapted_centroids.T

    positive = sim.gather(1, concept.unsqueeze(1)).squeeze(1)

    mask = F.one_hot(
        concept,
        num_classes=sim.size(1),
    ).bool()
    negative = sim.masked_fill(mask, float("-inf")).max(dim=1).values

    return F.relu(float(margin) + negative - positive).mean()


def batch_loss(
    adapter,
    heads,
    hidden,
    concept,
    attrs,
    centroid_hidden,
    args,
):
    adapted = adapter(hidden)
    concept_logits, attribute_logits = heads(adapted)

    concept_loss = F.cross_entropy(concept_logits, concept)
    attribute_loss = F.binary_cross_entropy_with_logits(
        attribute_logits,
        attrs,
    )
    margin_loss = centroid_margin_loss(
        adapter,
        hidden,
        concept,
        centroid_hidden,
        args.margin,
    )
    preservation = 1.0 - F.cosine_similarity(
        adapted,
        hidden,
        dim=-1,
    ).mean()

    total = (
        args.concept_weight * concept_loss
        + args.attribute_weight * attribute_loss
        + args.margin_weight * margin_loss
        + args.preservation_weight * preservation
    )

    return total, concept_loss, attribute_loss, margin_loss, preservation


@torch.no_grad()
def evaluate(adapter, heads, loader, centroid_hidden, device, args):
    adapter.eval()
    heads.eval()

    totals = {
        "loss": 0.0,
        "concept": 0.0,
        "attribute": 0.0,
        "margin": 0.0,
        "preservation": 0.0,
    }
    correct = count = batches = 0

    for hidden, concept, attrs in loader:
        hidden = hidden.to(device)
        concept = concept.to(device)
        attrs = attrs.to(device)

        loss, c, a, m, p = batch_loss(
            adapter,
            heads,
            hidden,
            concept,
            attrs,
            centroid_hidden,
            args,
        )

        logits, _ = heads(adapter(hidden))
        correct += int((logits.argmax(dim=-1) == concept).sum().item())
        count += int(concept.numel())

        totals["loss"] += float(loss.item())
        totals["concept"] += float(c.item())
        totals["attribute"] += float(a.item())
        totals["margin"] += float(m.item())
        totals["preservation"] += float(p.item())
        batches += 1

    return {
        key: value / max(1, batches)
        for key, value in totals.items()
    } | {"concept_acc": correct / max(1, count)}


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (
        args.tokenizer,
        args.model,
        args.init_adapter,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    base_model, base_checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    adapter, heads, init_checkpoint = load_semantic_adapter_checkpoint(
        args.init_adapter,
        device,
    )
    adapter.train()
    heads.train()

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
    centroid_hidden = raw_centroids(train_set, device)

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

    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(heads.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.2 Training")
    print("====================================================")
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Initial adapter        :", args.init_adapter)
    print("Initial adapter loss   :", init_checkpoint.get("loss"))
    print("Base encoder           : frozen")
    print("Adapter                :", f"{adapter.d_model}->{adapter.hidden_dim}->{adapter.d_model}")
    print("Residual scale         :", adapter.residual_scale)
    print("Train rows             :", len(train_rows))
    print("Validation rows        :", len(val_rows))
    print("Exact DEV overlap      :", len(overlaps))
    print("Learning rate          :", args.lr)
    print("Concept weight         :", args.concept_weight)
    print("Attribute weight       :", args.attribute_weight)
    print("Margin weight          :", args.margin_weight)
    print("Centroid margin        :", args.margin)
    print("Preservation weight    :", args.preservation_weight)
    print()

    best_val = float("inf")
    best_epoch = 0
    best_adapter = None
    best_heads = None
    bad_epochs = 0

    parameters = list(adapter.parameters()) + list(heads.parameters())

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
            loss, _c, _a, _m, _p = batch_loss(
                adapter,
                heads,
                hidden,
                concept,
                attrs,
                centroid_hidden,
                args,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            running += float(loss.item())
            batches += 1

        metrics = evaluate(
            adapter,
            heads,
            val_loader,
            centroid_hidden,
            device,
            args,
        )
        train_loss = running / max(1, batches)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| train={train_loss:.4f} "
            f"val={metrics['loss']:.4f} "
            f"concept_acc={metrics['concept_acc']:.1%} "
            f"margin={metrics['margin']:.4f} "
            f"attr={metrics['attribute']:.4f} "
            f"preserve={metrics['preservation']:.4f}"
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
        raise RuntimeError("No valid v0.2 checkpoint.")

    adapter.load_state_dict(best_adapter)
    heads.load_state_dict(best_heads)

    save_semantic_adapter_v02_checkpoint(
        args.output,
        adapter,
        heads,
        epoch=best_epoch,
        loss=best_val,
        base_model=args.model,
        learning_rate=args.lr,
        concept_weight=args.concept_weight,
        attribute_weight=args.attribute_weight,
        margin_weight=args.margin_weight,
        margin=args.margin,
        preservation_weight=args.preservation_weight,
    )

    print()
    print("Semantic Encoder Adapter v0.2 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best_val:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
