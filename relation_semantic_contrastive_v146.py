# relation_semantic_contrastive_v146.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_explicit_role_span_v143 import DATASET as CONCEPT_DATASET, CLASSES
from semantic_relation_paraphrase_v145 import RELATION_SAMPLES, FAMILIES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer
from relation_paraphrase_generalization_v145 import (
    encode_text_stages,
    concept_cv_features,
)

CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
STAGE = "block5"


class RelationProjection(nn.Module):
    def __init__(self, d_in: int, hidden: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class RelationClassifier(nn.Module):
    def __init__(self, proj_dim: int):
        super().__init__()
        self.fc = nn.Linear(proj_dim, 2)

    def forward(self, z):
        return self.fc(z)


def standardize_fit(X):
    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return mean, std


def standardize_apply(X, mean, std):
    return (X - mean) / std


def supervised_contrastive_loss(z, y, temperature=0.1):
    n = z.shape[0]
    sim = torch.matmul(z, z.T) / temperature

    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same = y[:, None].eq(y[None, :]) & ~eye

    logits = sim.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    positive_count = same.sum(dim=1).clamp_min(1)
    loss_i = -(log_prob.masked_fill(~same, 0.0).sum(dim=1) / positive_count)
    return loss_i.mean()


def train_contrastive_model(Xtr, ytr, args, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    proj = RelationProjection(Xtr.shape[1], args.hidden, args.proj_dim).to(Xtr.device)
    clf = RelationClassifier(args.proj_dim).to(Xtr.device)

    opt = torch.optim.AdamW(
        list(proj.parameters()) + list(clf.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for _ in range(args.epochs):
        proj.train()
        clf.train()
        opt.zero_grad()

        z = proj(Xtr)
        logits = clf(z)
        ce = F.cross_entropy(logits, ytr)
        supcon = supervised_contrastive_loss(z, ytr, args.temperature)
        loss = ce + args.supcon_weight * supcon

        loss.backward()
        opt.step()

    proj.eval()
    clf.eval()
    return proj, clf


def predict_family_split(X, y, train_idx, test_idx, args, seeds, seed_offset):
    Xtr_raw = X[train_idx]
    Xte_raw = X[test_idx]
    mean, std = standardize_fit(Xtr_raw)
    Xtr = standardize_apply(Xtr_raw, mean, std)
    Xte = standardize_apply(Xte_raw, mean, std)

    probs = []
    train_metrics = []

    for seed in seeds:
        proj, clf = train_contrastive_model(
            Xtr, y[train_idx], args, seed + seed_offset
        )
        with torch.no_grad():
            ztr = proj(Xtr)
            zte = proj(Xte)
            probs.append(torch.softmax(clf(zte), dim=-1).cpu())

            first = ztr[y[train_idx] == 0]
            second = ztr[y[train_idx] == 1]
            c0 = F.normalize(first.mean(0, keepdim=True), dim=-1)
            c1 = F.normalize(second.mean(0, keepdim=True), dim=-1)
            centroid_cos = float((c0 * c1).sum())
            within0 = float((first @ c0.T).mean())
            within1 = float((second @ c1.T).mean())
            train_metrics.append((centroid_cos, (within0 + within1) / 2.0))

    pred = torch.stack(probs).mean(0).argmax(-1)
    centroid_cos = sum(m[0] for m in train_metrics) / len(train_metrics)
    within = sum(m[1] for m in train_metrics) / len(train_metrics)
    return pred, centroid_cos, within


def reconstruct_target_accuracy(pred_rel, p1, p2):
    hits = []
    for i, s in enumerate(CONCEPT_DATASET):
        desired_pos = 0 if s["target"] == s["first_class"] else 1
        candidates = [
            j for j, r in enumerate(RELATION_SAMPLES)
            if r["target_position"] == desired_pos
        ]
        for j in candidates:
            pos = int(pred_rel[j])
            pred_target = int(p1[i]) if pos == 0 else int(p2[i])
            hits.append(pred_target == CLASS_TO_ID[s["target"]])
    return sum(hits) / len(hits)


def oracle_target_accuracy(p1, p2):
    hits = []
    for i, s in enumerate(CONCEPT_DATASET):
        pos = 0 if s["target"] == s["first_class"] else 1
        pred_target = int(p1[i]) if pos == 0 else int(p2[i])
        hits.append(pred_target == CLASS_TO_ID[s["target"]])
    return sum(hits) / len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--proj-dim", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.10)
    ap.add_argument("--supcon-weight", type=float, default=1.0)
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument(
        "--output-dir",
        default="results/relation_semantic_contrastive_v146",
    )
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("=" * 118)
    print(" LLM_GPU v1.4.6 Relation Semantic Contrastive Learning")
    print("=" * 118)
    print("Device          :", device)
    print("Base            :", args.model)
    print("Stage           :", STAGE)
    print("Relation data   : 20 phrases = 5 families x 2 labels x 2 paraphrases")
    print("Families        :", ", ".join(FAMILIES))
    print("Evaluation      : leave-one-relation-family-out")
    print("Projection      :", f"{model.d_model} -> {args.hidden} -> {args.proj_dim}")
    print("Loss            : CrossEntropy + "
          f"{args.supcon_weight:g} * SupervisedContrastive")
    print("Temperature     :", args.temperature)
    print("Base model      : completely frozen")
    print()

    print("Encoding relation phrases...")
    relation_cache = {
        s["text"]: encode_text_stages(model, tok, s["text"])
        for s in RELATION_SAMPLES
    }
    X = torch.stack(
        [relation_cache[s["text"]][STAGE] for s in RELATION_SAMPLES]
    ).to(device)
    y = torch.tensor(
        [s["target_position"] for s in RELATION_SAMPLES],
        dtype=torch.long,
        device=device,
    )

    pred_rel = torch.zeros(len(RELATION_SAMPLES), dtype=torch.long)
    family_scores = {}
    family_diag = {}

    for fi, fam in enumerate(FAMILIES):
        test = [i for i, s in enumerate(RELATION_SAMPLES) if s["family"] == fam]
        train = [i for i, s in enumerate(RELATION_SAMPLES) if s["family"] != fam]

        pred, centroid_cos, within = predict_family_split(
            X, y, train, test, args, seeds, 30000 + 1000 * fi
        )
        pred_rel[test] = pred
        family_scores[fam] = float(
            (pred == y[test].cpu()).float().mean()
        )
        family_diag[fam] = (centroid_cos, within)

    relation_acc = float((pred_rel == y.cpu()).float().mean())

    p1, p2 = concept_cv_features(model, tok, STAGE, args, seeds)
    y1 = torch.tensor(
        [CLASS_TO_ID[s["first_class"]] for s in CONCEPT_DATASET],
        dtype=torch.long,
    )
    y2 = torch.tensor(
        [CLASS_TO_ID[s["second_class"]] for s in CONCEPT_DATASET],
        dtype=torch.long,
    )
    concept_acc = (
        float((p1 == y1).float().mean())
        + float((p2 == y2).float().mean())
    ) / 2.0

    target_acc = reconstruct_target_accuracy(pred_rel, p1, p2)
    oracle_target = oracle_target_accuracy(p1, p2)

    print("BLOCK5 CONTRASTIVE RESULT")
    print("-" * 88)
    print(f"unseen relation phrase accuracy      : {relation_acc:.1%}")
    for fam in FAMILIES:
        centroid_cos, within = family_diag[fam]
        print(
            f"  {fam:10s} {family_scores[fam]:.1%}"
            f"   train centroid cos={centroid_cos:+.3f}"
            f"   within={within:.3f}"
        )
    print(f"mean concept accuracy                : {concept_acc:.1%}")
    print(f"reconstructed target (unseen phrase): {target_acc:.1%}")
    print(f"oracle relation target upper bound   : {oracle_target:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.5 best unseen relation accuracy : 60.0%")
    print("v1.4.5 best reconstructed Target     : 58.6%")
    print("Binary relation chance               : 50.0%")
    print()

    delta_rel = relation_acc - 0.60
    delta_target = target_acc - 0.586
    print("Delta vs v1.4.5")
    print("----------------")
    print(f"unseen relation accuracy             : {delta_rel:+.1%}")
    print(f"reconstructed target                 : {delta_target:+.1%}")
    print()

    print("Interpretation")
    print("--------------")
    print("The base Transformer remains frozen. Improvement therefore comes only from")
    print("learning a small semantic projection in which paraphrases with the same")
    print("FIRST/SECOND meaning are pulled together across relation families.")
    print("Leave-one-family-out evaluation tests whether that semantic abstraction")
    print("transfers to an unseen wording family rather than memorizing phrases.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "stage",
            "unseen_relation_accuracy",
            "mean_concept_accuracy",
            "reconstructed_target_accuracy",
            "oracle_relation_target_accuracy",
            "delta_relation_vs_v145",
            "delta_target_vs_v145",
            *[f"family_{fam}" for fam in FAMILIES],
        ])
        w.writerow([
            STAGE,
            relation_acc,
            concept_acc,
            target_acc,
            oracle_target,
            delta_rel,
            delta_target,
            *[family_scores[fam] for fam in FAMILIES],
        ])

    with (outdir / "family_diagnostics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "held_out_family",
            "accuracy",
            "train_centroid_cosine",
            "train_mean_within_class_cosine",
        ])
        for fam in FAMILIES:
            centroid_cos, within = family_diag[fam]
            w.writerow([fam, family_scores[fam], centroid_cos, within])

    print()
    print("Output dir:", outdir)


if __name__ == "__main__":
    main()
