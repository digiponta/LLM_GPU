# train_concept_calibration_v1511.py
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


def class_centroids(z, y):
    cs = []
    for i in range(len(CONCEPTS)):
        cs.append(F.normalize(z[y == i].mean(dim=0), dim=0))
    return torch.stack(cs)


@torch.no_grad()
def eval_with_train_centroids(projection, train_x, train_y, eval_x, eval_y):
    z_train = projection(train_x)
    centroids = class_centroids(z_train, train_y)
    z_eval = projection(eval_x)
    sims = z_eval @ centroids.T
    pred = sims.argmax(dim=1)
    acc = (pred == eval_y).float().mean().item()
    margins = []
    for i in range(len(eval_y)):
        gold = eval_y[i]
        own = sims[i, gold]
        others = torch.cat((sims[i, :gold], sims[i, gold+1:]))
        margins.append(float((own - others.max()).item()))
    return acc, sum(margins) / len(margins), centroids, pred, margins


def raw_eval(train_x, train_y, eval_x, eval_y):
    train_z = F.normalize(train_x, dim=-1)
    eval_z = F.normalize(eval_x, dim=-1)
    centroids = class_centroids(train_z, train_y)
    sims = eval_z @ centroids.T
    pred = sims.argmax(dim=1)
    acc = (pred == eval_y).float().mean().item()
    margins = []
    for i in range(len(eval_y)):
        gold = eval_y[i]
        own = sims[i, gold]
        others = torch.cat((sims[i, :gold], sims[i, gold+1:]))
        margins.append(float((own - others.max()).item()))
    return acc, sum(margins) / len(margins)


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

        projection.eval()
        hold_acc, hold_margin, _, _, _ = eval_with_train_centroids(
            projection, train_x, train_y, hold_x, hold_y
        )

        # Accuracy dominates; margin breaks ties. This prevents selecting
        # an over-separated model with a larger margin but worse generalization.
        score = hold_acc * 10.0 + hold_margin

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
                "hold_acc": hold_acc,
                "hold_margin": hold_margin,
            }
        else:
            stale += 1
            if stale >= patience:
                break

    projection.load_state_dict(best["state"])
    projection.eval()

    train_acc, train_margin, centroids, _, _ = eval_with_train_centroids(
        projection, train_x, train_y, train_x, train_y
    )
    hold_acc, hold_margin, centroids, hold_pred, hold_margins = eval_with_train_centroids(
        projection, train_x, train_y, hold_x, hold_y
    )

    return projection, centroids, hold_pred, hold_margins, {
        **best,
        "train_acc": train_acc,
        "train_margin": train_margin,
        "hold_acc": hold_acc,
        "hold_margin": hold_margin,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--temperature", type=float, default=0.10)
    ap.add_argument("--lambda-hard", type=float, default=0.10)
    ap.add_argument(
        "--lambda-preserve-sweep",
        default="0.25,0.5,1.0,2.0,4.0",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default="model/concept-calibration-v1511.pt")
    ap.add_argument("--results", default="results/concept_calibration_v1511")
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

    raw_train_acc, raw_train_margin = raw_eval(train_x, train_y, train_x, train_y)
    raw_hold_acc, raw_hold_margin = raw_eval(train_x, train_y, hold_x, hold_y)

    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print(" LLM_GPU v1.5.11 Preservation-Regularized Concept Calibration")
    print("=" * 92)
    print("Device                  :", device)
    print("Base checkpoint loss    :", base_ckpt.get("loss"))
    print("Train samples           :", len(train_y))
    print("Holdout samples         :", len(hold_y))
    print("Preservation sweep      :", ", ".join(str(x) for x in lambdas))
    print("Hard-negative weight    :", args.lambda_hard)
    print("Early-stop patience     :", args.patience)
    print()
    print(f"Raw train accuracy      : {raw_train_acc*100:.1f}%")
    print(f"Raw train mean margin   : {raw_train_margin:.6f}")
    print(f"Raw holdout accuracy    : {raw_hold_acc*100:.1f}%")
    print(f"Raw holdout mean margin : {raw_hold_margin:.6f}")
    print()

    rows = []
    best_global = None
    best_global_score = -1e9

    for lp in lambdas:
        projection, centroids, hold_pred, hold_margins, stats = train_one(
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
            "lambda_preserve": lp,
            "best_epoch": stats["epoch"],
            "train_accuracy": stats["train_acc"],
            "train_margin": stats["train_margin"],
            "holdout_accuracy": stats["hold_acc"],
            "holdout_margin": stats["hold_margin"],
            "best_loss": stats["loss"],
            "ce_loss": stats["ce"],
            "preserve_loss": stats["preserve"],
            "hard_loss": stats["hard"],
        }
        rows.append(row)

        print(
            f"lambda={lp:<4g} "
            f"epoch={stats['epoch']:>3d} "
            f"train={stats['train_acc']*100:5.1f}% "
            f"hold={stats['hold_acc']*100:5.1f}% "
            f"margin={stats['hold_margin']:.6f} "
            f"preserve={stats['preserve']:.6f}"
        )

        # Prefer full holdout accuracy, then largest margin.
        score = stats["hold_acc"] * 10.0 + stats["hold_margin"]
        if score > best_global_score:
            best_global_score = score
            best_global = {
                "lambda_preserve": lp,
                "projection_state": copy.deepcopy(projection.state_dict()),
                "centroids": centroids.detach().cpu().clone(),
                "stats": dict(stats),
                "pred": hold_pred.detach().cpu().clone(),
                "margins": list(hold_margins),
            }

    with (out / "sweep.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cp = Path(args.checkpoint)
    cp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": "v1.5.11",
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
        "stats": best_global["stats"],
    }, cp)

    with (out / "best_holdout.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "text", "predicted", "correct", "margin"])
        for i, (label, text) in enumerate(hold_meta):
            pred = CONCEPTS[int(best_global["pred"][i].item())]
            w.writerow([
                label, text, pred, int(pred == label), best_global["margins"][i]
            ])

    print()
    print("Best preservation weight :", best_global["lambda_preserve"])
    print(f"Best holdout accuracy     : {best_global['stats']['hold_acc']*100:.1f}%")
    print(f"Best holdout mean margin  : {best_global['stats']['hold_margin']:.6f}")
    print("Checkpoint                :", cp)
    print("Sweep                     :", out / "sweep.csv")
    print("Best holdout detail       :", out / "best_holdout.csv")


if __name__ == "__main__":
    main()
