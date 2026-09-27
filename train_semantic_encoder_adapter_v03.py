# train_semantic_encoder_adapter_v03.py
#
# Semantic Encoder Adapter v0.3:
#   Hard Negative Pair Supervision
#   + Acronym / Full-name Binding
#
# Continues from v0.2.
# Base v0.8 encoder remains frozen.

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v02 import load_semantic_adapter_v02_checkpoint
from semantic_encoder_adapter_v03 import (
    CONCEPT_LABELS,
    save_semantic_adapter_v03_checkpoint,
)
from tokenizer_bpe import Tokenizer
from train_semantic_encoder_adapter_v01 import (
    AdapterDataset,
    attribute_target,
    build_rows as build_base_rows,
    stratified_split,
)
from train_semantic_encoder_adapter_v02 import (
    centroid_margin_loss,
    raw_centroids,
)


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INIT = "model/model-gpu-v0.9-semantic-adapter-v02.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9-semantic-adapter-v03.pt"
SEED = 42


# Explicit acronym/full-name binding. These are intentionally distinct from
# the fixed 30 development prompts.
ACRONYM_BINDING_ROWS: List[Tuple[str, str]] = [
    ("CPUはCentral Processing Unitの略です。何を担当する装置ですか。", "tech_cpu"),
    ("Central Processing Unitを略すと何ですか。", "tech_cpu"),
    ("Central Processing Unitはコンピュータの中心的な処理装置です。名称を答えてください。", "tech_cpu"),
    ("Central Processing Unitは多様な命令を実行し汎用処理と制御を担当します。略称は何ですか。", "tech_cpu"),
    ("中央の処理を担うCentral Processing Unitの略称を答えてください。", "tech_cpu"),
    ("CPUという略称の正式名称はCentral Processing Unitです。どのような処理が得意ですか。", "tech_cpu"),

    ("GPUはGraphics Processing Unitの略です。どのような計算が得意ですか。", "tech_gpu"),
    ("Graphics Processing Unitを略すと何ですか。", "tech_gpu"),
    ("Graphics Processing Unitは多数の演算を並列に処理する装置です。名称を答えてください。", "tech_gpu"),
    ("Graphics Processing Unitは画像処理だけでなく大量並列計算にも使われます。略称は何ですか。", "tech_gpu"),
    ("GPUという略称の正式名称はGraphics Processing Unitです。強みを説明してください。", "tech_gpu"),
    ("Graphics Processing Unitは高スループットの並列計算を得意とします。略称を答えてください。", "tech_gpu"),
]


# Hard-negative prompt supervision. Each row specifies the expected concept
# and the particular confusing concept that must be ranked lower.
HARD_NEGATIVE_ROWS: List[Tuple[str, str, str]] = [
    ("中心的な演算装置として多様な命令を実行するのは何ですか。", "tech_cpu", "tech_gpu"),
    ("幅広い命令と制御を担当する汎用的な処理装置は何ですか。", "tech_cpu", "tech_gpu"),
    ("順序依存の強い処理やOS制御を主に担当する装置は何ですか。", "tech_cpu", "tech_gpu"),
    ("多数の同種演算を同時並列に処理する装置は何ですか。", "tech_gpu", "tech_cpu"),
    ("高スループットの並列演算を得意とする装置は何ですか。", "tech_gpu", "tech_cpu"),
    ("機械学習の行列演算を大量並列化する装置は何ですか。", "tech_gpu", "tech_cpu"),

    ("NVIDIA GPUを汎用計算に利用するための技術は何ですか。", "tech_cuda", "tech_python"),
    ("GPU計算を実行するNVIDIAの計算基盤は何ですか。", "tech_cuda", "tech_python"),
    ("NVIDIA製GPU向けの汎用計算プラットフォームは何ですか。", "tech_cuda", "tech_python"),
    ("読みやすい文法を持つ汎用プログラミング言語は何ですか。", "tech_python", "tech_cuda"),
    ("スクリプトや機械学習で広く使われるプログラミング言語は何ですか。", "tech_python", "tech_cuda"),

    ("Attentionを中心に系列を処理するモデル構造は何ですか。", "tech_transformer", "tech_python"),
    ("Self-Attentionを主要機構とするニューラルネットワーク構造は何ですか。", "tech_transformer", "tech_cuda"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Semantic Encoder Adapter v0.3."
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
    p.add_argument("--centroid-margin-weight", type=float, default=0.25)
    p.add_argument("--centroid-margin", type=float, default=0.05)
    p.add_argument("--pairwise-weight", type=float, default=0.75)
    p.add_argument("--pairwise-margin", type=float, default=0.08)
    p.add_argument("--preservation-weight", type=float, default=0.25)
    return p.parse_args()


def build_rows():
    rows = list(build_base_rows())
    for prompt, label in ACRONYM_BINDING_ROWS:
        rows.append((
            prompt,
            CONCEPT_LABELS.index(label),
            attribute_target(label, prompt),
        ))
    for prompt, positive, _negative in HARD_NEGATIVE_ROWS:
        rows.append((
            prompt,
            CONCEPT_LABELS.index(positive),
            attribute_target(positive, prompt),
        ))
    return rows


def exact_overlap_with_dev(rows):
    dev = {str(case["prompt"]) for case in CASES}
    overlap = sorted({
        prompt
        for prompt, _concept, _attrs in rows
        if prompt in dev
    })
    hard_overlap = sorted({
        prompt
        for prompt, _positive, _negative in HARD_NEGATIVE_ROWS
        if prompt in dev
    })
    return sorted(set(overlap + hard_overlap))


@torch.no_grad()
def encode_prompt_hidden(model, tokenizer, prompt):
    device = next(model.parameters()).device
    text = f"人: {prompt}\nAI: "
    ids = tokenizer.encode(text, add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = model.forward_hidden(x)
    return hidden[:, -1, :][0].detach()


def build_pairwise_hidden(model, tokenizer, device):
    hidden = []
    positive = []
    negative = []

    for prompt, pos_label, neg_label in HARD_NEGATIVE_ROWS:
        hidden.append(encode_prompt_hidden(model, tokenizer, prompt))
        positive.append(CONCEPT_LABELS.index(pos_label))
        negative.append(CONCEPT_LABELS.index(neg_label))

    return (
        torch.stack(hidden, dim=0).to(device),
        torch.tensor(positive, dtype=torch.long, device=device),
        torch.tensor(negative, dtype=torch.long, device=device),
    )


def explicit_pairwise_loss(
    adapter,
    pair_hidden,
    positive,
    negative,
    centroid_hidden,
    margin,
):
    adapted = F.normalize(adapter(pair_hidden), dim=-1)
    centroids = F.normalize(adapter(centroid_hidden), dim=-1)

    positive_sim = (adapted * centroids[positive]).sum(dim=-1)
    negative_sim = (adapted * centroids[negative]).sum(dim=-1)

    return F.relu(
        float(margin) + negative_sim - positive_sim
    ).mean()


def batch_loss(
    adapter,
    heads,
    hidden,
    concept,
    attrs,
    centroid_hidden,
    pair_hidden,
    pair_positive,
    pair_negative,
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
    preservation = 1.0 - F.cosine_similarity(
        adapted,
        hidden,
        dim=-1,
    ).mean()

    total = (
        args.concept_weight * concept_loss
        + args.attribute_weight * attribute_loss
        + args.centroid_margin_weight * centroid_loss
        + args.pairwise_weight * pairwise_loss
        + args.preservation_weight * preservation
    )

    return (
        total,
        concept_loss,
        attribute_loss,
        centroid_loss,
        pairwise_loss,
        preservation,
    )


@torch.no_grad()
def evaluate(
    adapter,
    heads,
    loader,
    centroid_hidden,
    pair_hidden,
    pair_positive,
    pair_negative,
    device,
    args,
):
    adapter.eval()
    heads.eval()

    totals = {
        "loss": 0.0,
        "concept": 0.0,
        "attribute": 0.0,
        "centroid": 0.0,
        "pairwise": 0.0,
        "preservation": 0.0,
    }
    correct = count = batches = 0

    for hidden, concept, attrs in loader:
        hidden = hidden.to(device)
        concept = concept.to(device)
        attrs = attrs.to(device)

        loss, c, a, cm, pw, p = batch_loss(
            adapter,
            heads,
            hidden,
            concept,
            attrs,
            centroid_hidden,
            pair_hidden,
            pair_positive,
            pair_negative,
            args,
        )

        logits, _ = heads(adapter(hidden))
        correct += int((logits.argmax(dim=-1) == concept).sum().item())
        count += int(concept.numel())

        totals["loss"] += float(loss.item())
        totals["concept"] += float(c.item())
        totals["attribute"] += float(a.item())
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

    adapter, heads, init_checkpoint = load_semantic_adapter_v02_checkpoint(
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

    pair_hidden, pair_positive, pair_negative = build_pairwise_hidden(
        base_model,
        tokenizer,
        device,
    )

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
    print(" Semantic Encoder Adapter v0.3 Training")
    print("====================================================")
    print("Device                    :", device)
    if device.type == "cuda":
        print("GPU                       :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss      :", base_checkpoint.get("loss"))
    print("Initial adapter           :", args.init_adapter)
    print("Initial adapter loss      :", init_checkpoint.get("loss"))
    print("Base encoder              : frozen")
    print("Adapter                   :", f"{adapter.d_model}->{adapter.hidden_dim}->{adapter.d_model}")
    print("Residual scale            :", adapter.residual_scale)
    print("Train rows                :", len(train_rows))
    print("Validation rows           :", len(val_rows))
    print("Acronym binding rows      :", len(ACRONYM_BINDING_ROWS))
    print("Hard negative rows        :", len(HARD_NEGATIVE_ROWS))
    print("Exact DEV overlap         :", len(overlaps))
    print("Learning rate             :", args.lr)
    print("Concept weight            :", args.concept_weight)
    print("Attribute weight          :", args.attribute_weight)
    print("Centroid margin weight    :", args.centroid_margin_weight)
    print("Centroid margin           :", args.centroid_margin)
    print("Pairwise weight           :", args.pairwise_weight)
    print("Pairwise margin           :", args.pairwise_margin)
    print("Preservation weight       :", args.preservation_weight)
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
            loss, _c, _a, _cm, _pw, _p = batch_loss(
                adapter,
                heads,
                hidden,
                concept,
                attrs,
                centroid_hidden,
                pair_hidden,
                pair_positive,
                pair_negative,
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
            pair_hidden,
            pair_positive,
            pair_negative,
            device,
            args,
        )
        train_loss = running / max(1, batches)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| train={train_loss:.4f} "
            f"val={metrics['loss']:.4f} "
            f"concept_acc={metrics['concept_acc']:.1%} "
            f"centroid={metrics['centroid']:.4f} "
            f"pair={metrics['pairwise']:.4f} "
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
        raise RuntimeError("No valid v0.3 checkpoint.")

    adapter.load_state_dict(best_adapter)
    heads.load_state_dict(best_heads)

    save_semantic_adapter_v03_checkpoint(
        args.output,
        adapter,
        heads,
        epoch=best_epoch,
        loss=best_val,
        base_model=args.model,
        learning_rate=args.lr,
        concept_weight=args.concept_weight,
        attribute_weight=args.attribute_weight,
        centroid_margin_weight=args.centroid_margin_weight,
        centroid_margin=args.centroid_margin,
        pairwise_weight=args.pairwise_weight,
        pairwise_margin=args.pairwise_margin,
        preservation_weight=args.preservation_weight,
    )

    print()
    print("Semantic Encoder Adapter v0.3 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best val loss    :", f"{best_val:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
