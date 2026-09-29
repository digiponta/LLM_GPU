# semantic_reliability_coverage_expansion_v151.py
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from semantic_contrast_decomposition_v1412 import CONTRAST_SAMPLES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer
from learned_semantic_span_detector_v1414 import (
    encode_token_states, token_labels, build_folds
)
from learned_semantic_risk_predictor_v1429 import (
    make_inner_oof,
    fit_risk_predictor,
    predict_risk,
    train_router,
    router_record,
    auc_roc,
    mean_std,
)

ENSEMBLE_SIZE = 5
TARGET_ACCURACY = 0.95
COVERAGE_GRID = [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60]


def entropy_binary(ps):
    vals = []
    for p in ps:
        p = min(max(float(p), 1e-8), 1-1e-8)
        vals.append(-(p*math.log(p) + (1-p)*math.log(1-p)))
    return statistics.mean(vals) / math.log(2.0)


def pairwise_disagreement(preds):
    if len(preds) < 2:
        return 0.0
    pairs = 0
    diff = 0
    for i in range(len(preds)):
        for j in range(i+1, len(preds)):
            pairs += 1
            diff += int(preds[i] != preds[j])
    return diff / pairs if pairs else 0.0


def augment_features(base_features, member_risks, member_preds):
    risk_mean = statistics.mean(member_risks)
    risk_std = statistics.pstdev(member_risks) if len(member_risks) > 1 else 0.0
    risk_range = max(member_risks) - min(member_risks)
    safe_vote_fraction = statistics.mean(1.0 if r < 0.5 else 0.0 for r in member_risks)
    vote_disagreement = pairwise_disagreement(member_preds)
    risk_entropy = entropy_binary(member_risks)
    return list(base_features) + [
        risk_mean,
        risk_std,
        risk_range,
        safe_vote_fraction,
        vote_disagreement,
        risk_entropy,
    ]


class ReliabilityRiskPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_reliability_predictor(records, seed, device):
    X = torch.tensor([r["reliability_features"] for r in records], dtype=torch.float32, device=device)
    y = torch.tensor([1-r["correct"] for r in records], dtype=torch.float32, device=device)

    mean = X.mean(0)
    std = X.std(0, unbiased=False).clamp_min(1e-6)
    Xn = (X-mean)/std

    if float(y.min()) == float(y.max()):
        return None, mean, std, float(y[0])

    torch.manual_seed(seed)
    m = ReliabilityRiskPredictor(X.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=1e-2)

    pos = float(y.sum().item())
    neg = float(len(y)-pos)
    pos_weight = torch.tensor(max(1.0, neg/max(pos,1.0)), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(600):
        opt.zero_grad(set_to_none=True)
        logits = m(Xn)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

    return m, mean, std, None


def predict_rel_risk(model, mean, std, constant, feats, device):
    if constant is not None:
        return constant
    x = torch.tensor(feats, dtype=torch.float32, device=device)
    x = (x-mean)/std
    with torch.no_grad():
        return float(torch.sigmoid(model(x)).cpu())


def build_member_ensemble(train_idx, states_all, labels_all, args, seed, device):
    ensemble = []
    for member in range(args.ensemble_size):
        member_seed = seed + 100000*member
        risk_train = make_inner_oof(
            train_idx, states_all, labels_all, args, member_seed, device
        )
        ensemble.append(
            fit_risk_predictor(risk_train, member_seed+777, device)
        )
    return ensemble


def record_with_reliability(i, states_all, router, ensemble, device):
    base = router_record(i, states_all, router, device)
    member_risks = []
    member_preds = []

    for model_r, rmean, rstd, constant in ensemble:
        risk = predict_risk(
            model_r, rmean, rstd, constant, base["features"], device
        )
        member_risks.append(risk)
        member_preds.append(base["prediction"])

    base["member_risks"] = member_risks
    base["reliability_features"] = augment_features(
        base["features"], member_risks, member_preds
    )
    return base


def make_inner_reliability_oof(train_idx, states_all, labels_all, args, seed, device):
    buckets = [[] for _ in range(4)]
    for j, idx in enumerate(train_idx):
        buckets[j % 4].append(idx)

    out = []
    for fold in range(4):
        val = buckets[fold]
        trn = [x for k,b in enumerate(buckets) if k != fold for x in b]

        router = train_router(
            states_all, labels_all, trn, args, seed + 1000000 + 1000*fold, device
        )
        ensemble = build_member_ensemble(
            trn, states_all, labels_all, args, seed + 2000000 + 1000*fold, device
        )
        for i in val:
            out.append(record_with_reliability(
                i, states_all, router, ensemble, device
            ))
    return out


def evaluate_coverage(records, target):
    ranked = sorted(records, key=lambda r: r["risk"])
    n = len(ranked)
    k = max(1, min(n, int(math.ceil(target*n))))
    acc = sum(r["correct"] for r in ranked[:k]) / k
    wrong_acc = sum(1-r["correct"] for r in ranked[:k]) / n
    total_wrong = sum(1-r["correct"] for r in ranked)
    rejected_wrong = sum(1-r["correct"] for r in ranked[k:])
    return {
        "target_coverage": target,
        "coverage": k/n,
        "accepted_accuracy": acc,
        "wrong_accepted_rate": wrong_acc,
        "error_capture_rate": rejected_wrong/total_wrong if total_wrong else 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--seeds", default="1-20")
    ap.add_argument("--span-hidden", type=int, default=64)
    ap.add_argument("--position-hidden", type=int, default=32)
    ap.add_argument("--explicit-hidden", type=int, default=32)
    ap.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE)
    ap.add_argument("--target-accuracy", type=float, default=TARGET_ACCURACY)
    ap.add_argument(
        "--output-dir",
        default="results/semantic_reliability_coverage_expansion_v151",
    )
    args = ap.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        a,b = args.seeds.split("-",1)
        seeds = list(range(int(a), int(b)+1))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("="*118)
    print(" LLM_GPU v1.5.1 Semantic Reliability Features / Coverage Expansion")
    print("="*118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Risk features    : v1.4.29 semantic-quality + ensemble reliability")
    print("Reliability      : mean/std/range risk + safe votes + disagreement + entropy")
    print("Primary metric   : Max Coverage @ Accepted Accuracy >= 95%")
    print("Validation       : nested CV")
    print("Fallback         : 未学習です")
    print()

    print("Encoding block5 token states...")
    states_all, labels_all = [], []
    for sample in CONTRAST_SAMPLES:
        states, offsets = encode_token_states(model, tok, sample["text"])
        states_all.append(states)
        labels_all.append(token_labels(sample, offsets))

    per_cov = {c: [] for c in COVERAGE_GRID}
    aucs = []
    all_rows = []

    print()
    print("Per-seed coverage expansion")
    print("-"*118)

    for seed in seeds:
        seed_records = []
        for outer_fold, test in enumerate(build_folds()):
            test_set = set(test)
            train_idx = [i for i in range(len(CONTRAST_SAMPLES)) if i not in test_set]

            rel_train = make_inner_reliability_oof(
                train_idx, states_all, labels_all, args,
                seed + 10000000*outer_fold, device
            )
            rel_model, rel_mean, rel_std, rel_const = fit_reliability_predictor(
                rel_train, seed + 20000000*outer_fold, device
            )

            router = train_router(
                states_all, labels_all, train_idx, args,
                seed + 30000000*outer_fold, device
            )
            ensemble = build_member_ensemble(
                train_idx, states_all, labels_all, args,
                seed + 40000000*outer_fold, device
            )

            for i in test:
                rec = record_with_reliability(
                    i, states_all, router, ensemble, device
                )
                rec["risk"] = predict_rel_risk(
                    rel_model, rel_mean, rel_std, rel_const,
                    rec["reliability_features"], device
                )
                rec["seed"] = seed
                rec["outer_fold"] = outer_fold
                seed_records.append(rec)
                all_rows.append(rec)

        labels = [1-r["correct"] for r in seed_records]
        scores = [r["risk"] for r in seed_records]
        auc = auc_roc(labels, scores)
        aucs.append(auc)

        print(
            f"seed={seed:>3d} baseline={statistics.mean(r['correct'] for r in seed_records):.1%} "
            f"AUC={auc:.3f}",
            end=""
        )

        for cov in COVERAGE_GRID:
            m = evaluate_coverage(seed_records, cov)
            per_cov[cov].append(m)
            print(f" | {cov:.0%}:{m['accepted_accuracy']:.1%}", end="")
        print()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for cov in COVERAGE_GRID:
        row = {"target_coverage": cov}
        for key in ["coverage","accepted_accuracy","wrong_accepted_rate","error_capture_rate"]:
            m,s = mean_std([x[key] for x in per_cov[cov]])
            row[key+"_mean"] = m
            row[key+"_std"] = s
        summary.append(row)

    with (out/"coverage_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(summary[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    with (out/"per_example.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "seed","outer_fold","sample_index","text","gold","prediction",
            "correct","risk"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k:r[k] for k in fields})

    feasible = [
        r for r in summary if r["accepted_accuracy_mean"] >= args.target_accuracy
    ]
    if feasible:
        best = max(feasible, key=lambda r:r["coverage_mean"])
    else:
        best = max(summary, key=lambda r:r["accepted_accuracy_mean"])

    auc_m, auc_s = mean_std(aucs)

    print()
    print("Coverage expansion summary")
    print("-"*118)
    print(
        f"{'target':>7s} {'coverage':>12s} {'accepted acc':>14s} "
        f"{'wrong accepted':>16s} {'error capture':>14s}"
    )
    for r in summary:
        print(
            f"{r['target_coverage']:>6.0%} "
            f"{r['coverage_mean']:>8.1%}±{r['coverage_std']:<5.1%} "
            f"{r['accepted_accuracy_mean']:>9.1%}±{r['accepted_accuracy_std']:<5.1%} "
            f"{r['wrong_accepted_rate_mean']:>11.1%}±{r['wrong_accepted_rate_std']:<5.1%} "
            f"{r['error_capture_rate_mean']:>9.1%}±{r['error_capture_rate_std']:<5.1%}"
        )

    print()
    print("Risk discrimination")
    print("-"*72)
    print(f"Outer-holdout error AUROC          : {auc_m:.3f} ± {auc_s:.3f}")
    print()
    print("Primary result")
    print("-"*72)
    print(f"Max Coverage @ >= {args.target_accuracy:.0%} accepted accuracy : {best['coverage_mean']:.1%}")
    print(f"Accepted accuracy                 : {best['accepted_accuracy_mean']:.1%} ± {best['accepted_accuracy_std']:.1%}")
    print(f"Wrong-accepted rate               : {best['wrong_accepted_rate_mean']:.1%} ± {best['wrong_accepted_rate_std']:.1%}")
    print(f"Error capture rate                : {best['error_capture_rate_mean']:.1%} ± {best['error_capture_rate_std']:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.32 train-calibrated deployment : 95.6% at 7.5% realized coverage")
    print("v1.4.31 test-ranked proof-of-concept : 95.6% at 20% coverage")
    print("Goal                              : expand coverage while keeping accepted accuracy >=95%")
    print("Output dir:", out)


if __name__ == "__main__":
    main()
