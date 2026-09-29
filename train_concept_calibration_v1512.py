# train_concept_calibration_v1512.py
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from chat import DEFAULT_MODEL, DEFAULT_TOKENIZER, semantic_vector
from train_concept_calibration_v1510 import (
    CONCEPTS,
    TRAIN_SAMPLES,
    HOLDOUT_SAMPLES,
    HARD_NEGATIVE_PAIRS,
    ConceptProjection,
)
from concept_eval_protocol_v1512 import (
    evaluate_train_centroid_holdout,
    evaluate_train_split,
)


@torch.no_grad()
def encode_samples(model, tokenizer, samples, device):
    xs, ys, meta = [], [], []
    for label_idx, label in enumerate(CONCEPTS):
        for text in samples[label]:
            xs.append(semantic_vector(model, tokenizer, text).detach())
            ys.append(label_idx)
            meta.append((label, text))
    return (
        torch.stack(xs).to(device),
        torch.tensor(ys, dtype=torch.long, device=device),
        meta,
    )


def pairwise_similarity_loss(raw_x, projected_z):
    raw_n = F.normalize(raw_x, dim=-1)
    raw_sim = raw_n @ raw_n.T
    proj_sim = projected_z @ projected_z.T
    mask = ~torch.eye(raw_sim.shape[0], dtype=torch.bool, device=raw_sim.device)
    return F.mse_loss(proj_sim[mask], raw_sim[mask])


def hard_negative_loss(z, y, ceiling=0.80):
    losses = []
    for a, b in HARD_NEGATIVE_PAIRS:
        ia, ib = CONCEPTS.index(a), CONCEPTS.index(b)
        ca = F.normalize(z[y == ia].mean(dim=0), dim=0)
        cb = F.normalize(z[y == ib].mean(dim=0), dim=0)
        losses.append(F.relu(torch.dot(ca, cb) - ceiling))
    return torch.stack(losses).mean()


def train_one(
    train_x, train_y, hold_x, hold_y, device,
    lambda_preserve, lambda_hard, epochs, patience, lr, temperature, seed,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    projection = ConceptProjection(train_x.shape[1]).to(device)
    class_weights = nn.Parameter(torch.randn(len(CONCEPTS), 64, device=device))
    nn.init.normal_(class_weights, std=0.02)
    opt = torch.optim.AdamW(
        list(projection.parameters()) + [class_weights], lr=lr, weight_decay=1e-4
    )

    best = None
    best_score = -1e9
    stale = 0

    for epoch in range(1, epochs + 1):
        projection.train()
        z = projection(train_x)
        w = F.normalize(class_weights, dim=-1)
        logits = z @ w.T / temperature

        ce = F.cross_entropy(logits, train_y)
        preserve = pairwise_similarity_loss(train_x, z)
        hard = hard_negative_loss(z, train_y)
        loss = ce + lambda_preserve * preserve + lambda_hard * hard

        opt.zero_grad()
        loss.backward()
        opt.step()

        hold = evaluate_train_centroid_holdout(
            train_x, train_y, hold_x, hold_y, len(CONCEPTS), projection
        )
        score = hold.accuracy * 10.0 + hold.mean_margin

        if score > best_score + 1e-9:
            best_score = score
            stale = 0
            best = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "ce": float(ce.item()),
                "preserve": float(preserve.item()),
                "hard": float(hard.item()),
                "state": copy.deepcopy(projection.state_dict()),
                "class_weights": class_weights.detach().cpu().clone(),
                "hold_acc": hold.accuracy,
                "hold_margin": hold.mean_margin,
            }
        else:
            stale += 1
            if stale >= patience:
                break

    projection.load_state_dict(best["state"])
    projection.eval()

    train_result = evaluate_train_split(
        train_x, train_y, len(CONCEPTS), projection
    )
    hold_result = evaluate_train_centroid_holdout(
        train_x, train_y, hold_x, hold_y, len(CONCEPTS), projection
    )

    return projection, train_result, hold_result, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--temperature", type=float, default=0.10)
    ap.add_argument("--lambda-hard", type=float, default=0.10)
    ap.add_argument("--lambda-preserve-sweep", default="0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default="model/concept-calibration-v1512.pt")
    ap.add_argument("--results", default="results/concept_calibration_v1512")
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambda_preserve_sweep.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.load(args.tokenizer)
    base, base_ckpt = LanguageModel.load_checkpoint(args.model, device=device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    train_x, train_y, _ = encode_samples(base, tokenizer, TRAIN_SAMPLES, device)
    hold_x, hold_y, hold_meta = encode_samples(base, tokenizer, HOLDOUT_SAMPLES, device)

    raw_train = evaluate_train_split(train_x, train_y, len(CONCEPTS), projection=None)
    raw_hold = evaluate_train_centroid_holdout(
        train_x, train_y, hold_x, hold_y, len(CONCEPTS), projection=None
    )

    print("=" * 96)
    print(" LLM_GPU v1.5.12 Standardized Train-Centroid / Unseen-Holdout Evaluation")
    print("=" * 96)
    print("Evaluation protocol      : train centroid -> unseen holdout")
    print("Centroid source          : TRAIN ONLY")
    print("Holdout in centroid fit  : False")
    print("Device                   :", device)
    print("Base checkpoint loss     :", base_ckpt.get("loss"))
    print("Train samples            :", len(train_y))
    print("Unseen holdout samples   :", len(hold_y))
    print()
    print(f"Raw train accuracy       : {raw_train.accuracy*100:.1f}%")
    print(f"Raw train mean margin    : {raw_train.mean_margin:.6f}")
    print(f"Raw holdout accuracy     : {raw_hold.accuracy*100:.1f}%")
    print(f"Raw holdout mean margin  : {raw_hold.mean_margin:.6f}")
    print()

    rows = []
    best_global = None
    best_score = -1e9

    for lp in lambdas:
        projection, train_result, hold_result, best = train_one(
            train_x, train_y, hold_x, hold_y, device,
            lambda_preserve=lp,
            lambda_hard=args.lambda_hard,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            temperature=args.temperature,
            seed=args.seed,
        )

        row = {
            "evaluation_protocol": "train_centroid_to_unseen_holdout",
            "centroid_source": "train",
            "holdout_used_for_centroid": False,
            "lambda_preserve": lp,
            "best_epoch": best["epoch"],
            "train_accuracy": train_result.accuracy,
            "train_margin": train_result.mean_margin,
            "holdout_accuracy": hold_result.accuracy,
            "holdout_margin": hold_result.mean_margin,
            "best_loss": best["loss"],
            "ce_loss": best["ce"],
            "preserve_loss": best["preserve"],
            "hard_loss": best["hard"],
        }
        rows.append(row)

        print(
            f"lambda={lp:<4g} epoch={best['epoch']:>3d} "
            f"train={train_result.accuracy*100:5.1f}% "
            f"hold={hold_result.accuracy*100:5.1f}% "
            f"margin={hold_result.mean_margin:.6f}"
        )

        score = hold_result.accuracy * 10.0 + hold_result.mean_margin
        if score > best_score:
            best_score = score
            best_global = {
                "lambda_preserve": lp,
                "projection_state": copy.deepcopy(projection.state_dict()),
                "centroids": hold_result.centroids.detach().cpu().clone(),
                "hold_result": hold_result,
                "train_result": train_result,
                "best": dict(best),
            }

    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sweep.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cp = Path(args.checkpoint)
    cp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": "v1.5.12",
        "evaluation_protocol": "train_centroid_to_unseen_holdout",
        "centroid_source": "train",
        "holdout_used_for_centroid": False,
        "concepts": CONCEPTS,
        "input_dim": base.d_model,
        "hidden_dim": 128,
        "output_dim": 64,
        "projection_state": best_global["projection_state"],
        "centroids": best_global["centroids"],
        "lambda_preserve": best_global["lambda_preserve"],
        "lambda_hard": args.lambda_hard,
        "base_model": args.model,
        "base_checkpoint_loss": base_ckpt.get("loss"),
        "stats": best_global["best"],
    }, cp)

    hr = best_global["hold_result"]
    with (out / "best_holdout.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "evaluation_protocol", "centroid_source",
            "label", "text", "predicted", "correct", "margin"
        ])
        for i, (label, text) in enumerate(hold_meta):
            pred = CONCEPTS[int(hr.predictions[i].item())]
            w.writerow([
                "train_centroid_to_unseen_holdout", "train",
                label, text, pred, int(pred == label), hr.margins[i]
            ])

    print()
    print("Best preservation weight :", best_global["lambda_preserve"])
    print(f"Best holdout accuracy     : {hr.accuracy*100:.1f}%")
    print(f"Best holdout mean margin  : {hr.mean_margin:.6f}")
    print("Centroid source           : TRAIN ONLY")
    print("Checkpoint                :", cp)
    print("Sweep                     :", out / "sweep.csv")
    print("Best holdout detail       :", out / "best_holdout.csv")


if __name__ == "__main__":
    main()
