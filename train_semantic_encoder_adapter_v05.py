# train_semantic_encoder_adapter_v05.py
#
# Semantic Encoder Adapter v0.5:
#   Hierarchical Processor Semantics
#
# Parent concept:
#   processor
#
# CPU branch:
#   general-purpose / control-oriented
#
# GPU branch:
#   throughput-oriented / data-parallel
#
# Continues from v0.4. Base v0.8 encoder remains frozen.

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v04 import load_semantic_adapter_v04_checkpoint
from semantic_encoder_adapter_v05 import (
    HIERARCHY_LABELS,
    HierarchicalProcessorHead,
    save_semantic_adapter_v05_checkpoint,
)
from tokenizer_bpe import Tokenizer
from train_semantic_encoder_adapter_v01 import (
    AdapterDataset,
    stratified_split,
)
from train_semantic_encoder_adapter_v02 import (
    centroid_margin_loss,
    raw_centroids,
)
from train_semantic_encoder_adapter_v03 import (
    build_rows,
    build_pairwise_hidden,
    explicit_pairwise_loss,
)


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INIT = "model/model-gpu-v0.9-semantic-adapter-v04.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9-semantic-adapter-v05.pt"
SEED = 42


HIERARCHY_ROWS: List[Tuple[str, Tuple[int, ...]]] = [
    # CPU / general-purpose / control-oriented.
    ("CPUは汎用プロセッサとして多様な命令を処理します。", (1, 1, 1, 0, 0)),
    ("CPUは分岐や制御を含む幅広い処理を担当するプロセッサです。", (1, 1, 1, 0, 0)),
    ("多様な命令を順序立てて実行する汎用プロセッサはCPUです。", (1, 1, 1, 0, 0)),
    ("OS制御や複雑な分岐を担当する中心的なプロセッサはCPUです。", (1, 1, 1, 0, 0)),
    ("幅広い仕事をこなすgeneral-purpose processorとしてCPUを使います。", (1, 1, 1, 0, 0)),
    ("低レイテンシで多様な命令を扱うcontrol-oriented processorはCPUです。", (1, 1, 1, 0, 0)),

    # GPU / throughput-oriented / data-parallel.
    ("GPUは高スループットの並列計算に最適化されたプロセッサです。", (1, 0, 0, 1, 1)),
    ("GPUは大量の同種演算を同時に処理するdata-parallel processorです。", (1, 0, 0, 1, 1)),
    ("多数の演算を並列化して処理するthroughput-oriented processorはGPUです。", (1, 0, 0, 1, 1)),
    ("GPUは分岐中心よりも大量並列演算を得意とするプロセッサです。", (1, 0, 0, 1, 1)),
    ("機械学習の行列計算を高スループットで処理するプロセッサはGPUです。", (1, 0, 0, 1, 1)),
    ("GPUは汎用命令制御よりdata-parallel throughputを重視した設計です。", (1, 0, 0, 1, 1)),

    # Parent-only processor statements to prevent CPU/GPU from becoming unrelated.
    ("CPUとGPUはいずれも計算を行うプロセッサです。", (1, 0, 0, 0, 0)),
    ("CPUもGPUもprocessorという上位概念を共有します。", (1, 0, 0, 0, 0)),
]


class HierarchyDataset(Dataset):
    def __init__(self, rows, model, tokenizer):
        self.items = []
        for prompt, target in rows:
            hidden = encode_prompt_hidden(model, tokenizer, prompt).cpu()
            self.items.append((
                hidden,
                torch.tensor(target, dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Semantic Encoder Adapter v0.5."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--init-adapter", default=DEFAULT_INIT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--validation-ratio", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--concept-weight", type=float, default=1.0)
    p.add_argument("--attribute-weight", type=float, default=0.75)
    p.add_argument("--hierarchy-weight", type=float, default=1.0)
    p.add_argument("--hierarchy-contrast-weight", type=float, default=0.75)
    p.add_argument("--centroid-margin-weight", type=float, default=0.20)
    p.add_argument("--centroid-margin", type=float, default=0.05)
    p.add_argument("--pairwise-weight", type=float, default=0.50)
    p.add_argument("--pairwise-margin", type=float, default=0.08)
    p.add_argument("--preservation-weight", type=float, default=0.25)
    return p.parse_args()


def exact_overlap_with_dev(rows):
    dev = {str(case["prompt"]) for case in CASES}
    base_overlap = sorted({
        prompt for prompt, _concept, _attrs in rows if prompt in dev
    })
    hierarchy_overlap = sorted({
        prompt for prompt, _target in HIERARCHY_ROWS if prompt in dev
    })
    return sorted(set(base_overlap + hierarchy_overlap))


@torch.no_grad()
def encode_prompt_hidden(model, tokenizer, prompt):
    device = next(model.parameters()).device
    text = f"人: {prompt}\nAI: "
    ids = tokenizer.encode(text, add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = model.forward_hidden(x)
    return hidden[:, -1, :][0].detach()


def hierarchy_contrast_loss(hierarchy_logits, hierarchy_targets):
    probs = torch.sigmoid(hierarchy_logits)

    general_idx = HIERARCHY_LABELS.index("general_purpose")
    control_idx = HIERARCHY_LABELS.index("control_oriented")
    throughput_idx = HIERARCHY_LABELS.index("throughput_oriented")
    parallel_idx = HIERARCHY_LABELS.index("data_parallel")

    cpu_mask = (
        (hierarchy_targets[:, general_idx] > 0.5)
        & (hierarchy_targets[:, control_idx] > 0.5)
    )
    gpu_mask = (
        (hierarchy_targets[:, throughput_idx] > 0.5)
        & (hierarchy_targets[:, parallel_idx] > 0.5)
    )

    losses = []

    if cpu_mask.any():
        cpu_positive = 0.5 * (
            probs[cpu_mask, general_idx] + probs[cpu_mask, control_idx]
        )
        cpu_negative = 0.5 * (
            probs[cpu_mask, throughput_idx] + probs[cpu_mask, parallel_idx]
        )
        losses.append(F.relu(0.20 + cpu_negative - cpu_positive).mean())

    if gpu_mask.any():
        gpu_positive = 0.5 * (
            probs[gpu_mask, throughput_idx] + probs[gpu_mask, parallel_idx]
        )
        gpu_negative = 0.5 * (
            probs[gpu_mask, general_idx] + probs[gpu_mask, control_idx]
        )
        losses.append(F.relu(0.20 + gpu_negative - gpu_positive).mean())

    if not losses:
        return hierarchy_logits.new_tensor(0.0)

    return torch.stack(losses).mean()


def hierarchy_batch_loss(adapter, hierarchy_head, hidden, targets):
    adapted = adapter(hidden)
    logits = hierarchy_head(adapted)

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    contrast = hierarchy_contrast_loss(logits, targets)

    return bce, contrast


def combined_loss(
    adapter,
    heads,
    hierarchy_head,
    hidden,
    concept,
    attrs,
    centroid_hidden,
    pair_hidden,
    pair_positive,
    pair_negative,
    hierarchy_hidden,
    hierarchy_targets,
    args,
):
    adapted = adapter(hidden)
    concept_logits, attribute_logits = heads(adapted)

    concept_loss = F.cross_entropy(concept_logits, concept)
    attribute_loss = F.binary_cross_entropy_with_logits(
        attribute_logits,
        attrs,
    )
    centroid_loss = centroid_margin_loss(
        adapter,
        hidden,
        concept,
        centroid_hidden,
        args.centroid_margin,
    )
    pairwise_loss = explicit_pairwise_loss(
        adapter,
        pair_hidden,
        pair_positive,
        pair_negative,
        centroid_hidden,
        args.pairwise_margin,
    )
    hierarchy_loss, hierarchy_contrast = hierarchy_batch_loss(
        adapter,
        hierarchy_head,
        hierarchy_hidden,
        hierarchy_targets,
    )
    preservation = 1.0 - F.cosine_similarity(
        adapted,
        hidden,
        dim=-1,
    ).mean()

    total = (
        args.concept_weight * concept_loss
        + args.attribute_weight * attribute_loss
        + args.hierarchy_weight * hierarchy_loss
        + args.hierarchy_contrast_weight * hierarchy_contrast
        + args.centroid_margin_weight * centroid_loss
        + args.pairwise_weight * pairwise_loss
        + args.preservation_weight * preservation
    )

    return (
        total,
        concept_loss,
        attribute_loss,
        hierarchy_loss,
        hierarchy_contrast,
        centroid_loss,
        pairwise_loss,
        preservation,
    )


@torch.no_grad()
def evaluate(
    adapter,
    heads,
    hierarchy_head,
    loader,
    centroid_hidden,
    pair_hidden,
    pair_positive,
    pair_negative,
    hierarchy_hidden,
    hierarchy_targets,
    device,
    args,
):
    adapter.eval()
    heads.eval()
    hierarchy_head.eval()

    totals = {
        "loss": 0.0,
        "concept": 0.0,
        "attribute": 0.0,
        "hierarchy": 0.0,
        "hierarchy_contrast": 0.0,
        "centroid": 0.0,
        "pairwise": 0.0,
        "preservation": 0.0,
    }
    correct = count = batches = 0

    for hidden, concept, attrs in loader:
        hidden = hidden.to(device)
        concept = concept.to(device)
        attrs = attrs.to(device)

        values = combined_loss(
            adapter,
            heads,
            hierarchy_head,
            hidden,
            concept,
            attrs,
            centroid_hidden,
            pair_hidden,
            pair_positive,
            pair_negative,
            hierarchy_hidden,
            hierarchy_targets,
            args,
        )
        loss, c, a, h, hc, cm, pw, p = values

        logits, _ = heads(adapter(hidden))
        correct += int((logits.argmax(dim=-1) == concept).sum().item())
        count += int(concept.numel())

        totals["loss"] += float(loss.item())
        totals["concept"] += float(c.item())
        totals["attribute"] += float(a.item())
        totals["hierarchy"] += float(h.item())
        totals["hierarchy_contrast"] += float(hc.item())
        totals["centroid"] += float(cm.item())
        totals["pairwise"] += float(pw.item())
        totals["preservation"] += float(p.item())
        batches += 1

    metrics = {
        key: value / max(1, batches)
        for key, value in totals.items()
    }
    metrics["concept_acc"] = correct / max(1, count)
    return metrics


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

    adapter, heads, init_checkpoint = load_semantic_adapter_v04_checkpoint(
        args.init_adapter,
        device,
    )
    adapter.train()
    heads.train()

    hierarchy_head = HierarchicalProcessorHead(
        d_model=adapter.d_model,
    ).to(device)

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

    pair_hidden, pair_positive, pair_negative = build_pairwise_hidden(
        base_model,
        tokenizer,
        device,
    )

    hierarchy_set = HierarchyDataset(
        HIERARCHY_ROWS,
        base_model,
        tokenizer,
    )
    hierarchy_hidden = torch.stack(
        [item[0] for item in hierarchy_set.items],
        dim=0,
    ).to(device)
    hierarchy_targets = torch.stack(
        [item[1] for item in hierarchy_set.items],
        dim=0,
    ).to(device)

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

    parameters = (
        list(adapter.parameters())
        + list(heads.parameters())
        + list(hierarchy_head.parameters())
    )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=0.01,
    )

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.5 Training")
    print("====================================================")
    print("Device                    :", device)
    if device.type == "cuda":
        print("GPU                       :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss      :", base_checkpoint.get("loss"))
    print("Initial adapter           :", args.init_adapter)
    print("Initial adapter loss      :", init_checkpoint.get("loss"))
    print("Base encoder              : frozen")
    print("Adapter                   :", f"{adapter.d_model}->{adapter.hidden_dim}->{adapter.d_model}")
    print("Hierarchy labels          :", ", ".join(HIERARCHY_LABELS))
    print("Hierarchy rows            :", len(HIERARCHY_ROWS))
    print("Train rows                :", len(train_rows))
    print("Validation rows           :", len(val_rows))
    print("Exact DEV overlap         :", len(overlaps))
    print("Learning rate             :", args.lr)
    print("Concept weight            :", args.concept_weight)
    print("Attribute weight          :", args.attribute_weight)
    print("Hierarchy weight          :", args.hierarchy_weight)
    print("Hierarchy contrast weight :", args.hierarchy_contrast_weight)
    print("Centroid margin weight    :", args.centroid_margin_weight)
    print("Pairwise weight           :", args.pairwise_weight)
    print("Preservation weight       :", args.preservation_weight)
    print()

    best_val = float("inf")
    best_epoch = 0
    best_adapter = None
    best_heads = None
    best_hierarchy = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        heads.train()
        hierarchy_head.train()
        running = 0.0
        batches = 0

        for hidden, concept, attrs in train_loader:
            hidden = hidden.to(device)
            concept = concept.to(device)
            attrs = attrs.to(device)

            optimizer.zero_grad(set_to_none=True)
            values = combined_loss(
                adapter,
                heads,
                hierarchy_head,
                hidden,
                concept,
                attrs,
                centroid_hidden,
                pair_hidden,
                pair_positive,
                pair_negative,
                hierarchy_hidden,
                hierarchy_targets,
                args,
            )
            loss = values[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            running += float(loss.item())
            batches += 1

        metrics = evaluate(
            adapter,
            heads,
            hierarchy_head,
            val_loader,
            centroid_hidden,
            pair_hidden,
            pair_positive,
            pair_negative,
            hierarchy_hidden,
            hierarchy_targets,
            device,
            args,
        )
        train_loss = running / max(1, batches)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| train={train_loss:.4f} "
            f"val={metrics['loss']:.4f} "
            f"concept_acc={metrics['concept_acc']:.1%} "
            f"hier={metrics['hierarchy']:.4f} "
            f"hcontrast={metrics['hierarchy_contrast']:.4f} "
            f"centroid={metrics['centroid']:.4f} "
            f"pair={metrics['pairwise']:.4f} "
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
            best_hierarchy = {
                k: v.detach().cpu().clone()
                for k, v in hierarchy_head.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_adapter is None or best_heads is None or best_hierarchy is None:
        raise RuntimeError("No valid v0.5 checkpoint.")

    adapter.load_state_dict(best_adapter)
    heads.load_state_dict(best_heads)
    hierarchy_head.load_state_dict(best_hierarchy)

    save_semantic_adapter_v05_checkpoint(
        args.output,
        adapter,
        heads,
        hierarchy_head,
        epoch=best_epoch,
        loss=best_val,
        base_model=args.model,
        learning_rate=args.lr,
        concept_weight=args.concept_weight,
        attribute_weight=args.attribute_weight,
        hierarchy_weight=args.hierarchy_weight,
        hierarchy_contrast_weight=args.hierarchy_contrast_weight,
        centroid_margin_weight=args.centroid_margin_weight,
        centroid_margin=args.centroid_margin,
        pairwise_weight=args.pairwise_weight,
        pairwise_margin=args.pairwise_margin,
        preservation_weight=args.preservation_weight,
    )

    print()
    print("Semantic Encoder Adapter v0.5 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best_val:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
