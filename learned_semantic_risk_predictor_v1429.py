# learned_semantic_risk_predictor_v1429.py
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
    encode_token_states,
    token_labels,
    pooled_gold_role_vector,
    train_position_head,
    build_folds,
    NEGATED,
    SELECTED,
)
from explicit_implicit_semantic_decomposition_v1415 import (
    train_binary_span_tagger,
    train_explicitness_head,
    predict_explicit,
)
from adaptive_semantic_window_v1420 import role_scores, position_probs, mass_window

MASS = 0.80
RISK_THRESHOLDS = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]
FEATURE_NAMES = [
    "route_margin",
    "sel_max",
    "sel_top_gap",
    "sel_entropy",
    "sel_mass_width_ratio",
    "neg_max",
    "neg_top_gap",
    "neg_entropy",
    "neg_mass_width_ratio",
    "explicit_confidence",
]


def mean_std(xs):
    if len(xs) <= 1:
        return (float(xs[0]) if xs else 0.0), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def norm_entropy(scores):
    p = scores.detach().float().clamp_min(1e-8)
    p = p / p.sum().clamp_min(1e-8)
    h = float((-(p * p.log()).sum()).cpu())
    den = math.log(max(2, p.numel()))
    return h / den


def top_gap(scores):
    s = scores.detach().float()
    if s.numel() < 2:
        return float(s.max().cpu()) if s.numel() else 0.0
    v = torch.topk(s, 2).values
    return float((v[0] - v[1]).cpu())


def route_margin(fused):
    q = fused.detach().float()
    den = float(q.sum().cpu())
    if den <= 1e-12:
        return 0.0
    return float(torch.abs(q[1] - q[0]).cpu()) / den


def train_router(states_all, labels_all, train_idx, args, seed, device):
    sel = train_binary_span_tagger(
        [states_all[i] for i in train_idx],
        [labels_all[i] for i in train_idx],
        SELECTED, args, seed + 101, device,
    )
    neg = train_binary_span_tagger(
        [states_all[i] for i in train_idx],
        [labels_all[i] for i in train_idx],
        NEGATED, args, seed + 202, device,
    )
    explicit, emean, estd = train_explicitness_head(
        [states_all[i] for i in train_idx],
        [labels_all[i] for i in train_idx],
        args, seed + 303, device,
    )

    pv, py = [], []
    for i in train_idx:
        sv = pooled_gold_role_vector(states_all[i], labels_all[i], SELECTED)
        nv = pooled_gold_role_vector(states_all[i], labels_all[i], NEGATED)
        if sv is not None:
            pv.append(sv)
            py.append(CONTRAST_SAMPLES[i]["selected_position"])
        if nv is not None:
            pv.append(nv)
            py.append(CONTRAST_SAMPLES[i]["negated_position"])

    pos, pmean, pstd = train_position_head(
        pv, py, args, seed + 404, device
    )
    return sel, neg, explicit, emean, estd, pos, pmean, pstd


def router_record(i, states_all, router, device):
    sel_tagger, neg_tagger, explicit_head, emean, estd, pos_head, pmean, pstd = router
    states = states_all[i]
    sample = CONTRAST_SAMPLES[i]

    ss = role_scores(states, sel_tagger, device)
    ns = role_scores(states, neg_tagger, device)

    exp_hat, exp_conf = predict_explicit(
        states, explicit_head, emean, estd, device
    )

    sv, s0, s1 = mass_window(states, ss, MASS)
    fused = position_probs(sv, pos_head, pmean, pstd).clone()

    neg_width = 0
    if exp_hat == 1:
        nv, n0, n1 = mass_window(states, ns, MASS)
        neg_width = max(1, n1 - n0 + 1)
        np = position_probs(nv, pos_head, pmean, pstd)
        fused[0] += np[1]
        fused[1] += np[0]

    pred = int(fused.argmax())
    gold = int(sample["selected_position"])
    n_tok = max(1, len(states))

    feats = [
        route_margin(fused),
        float(ss.max().detach().cpu()),
        top_gap(ss),
        norm_entropy(ss),
        max(1, s1 - s0 + 1) / n_tok,
        float(ns.max().detach().cpu()),
        top_gap(ns),
        norm_entropy(ns),
        (neg_width / n_tok) if exp_hat == 1 else 0.0,
        float(exp_conf),
    ]

    return {
        "sample_index": i,
        "text": sample["text"],
        "gold": gold,
        "prediction": pred,
        "correct": int(pred == gold),
        "features": feats,
    }


def make_inner_oof(outer_train, states_all, labels_all, args, seed, device):
    # 4-fold OOF risk-training data, fully contained inside the outer train split.
    buckets = [[] for _ in range(4)]
    for j, idx in enumerate(outer_train):
        buckets[j % 4].append(idx)

    records = []
    for inner_fold in range(4):
        val = buckets[inner_fold]
        trn = [x for k,b in enumerate(buckets) if k != inner_fold for x in b]
        router = train_router(
            states_all, labels_all, trn, args,
            seed + 100000 + 1000 * inner_fold, device
        )
        for i in val:
            records.append(router_record(i, states_all, router, device))
    return records


class RiskPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def fit_risk_predictor(records, seed, device):
    X = torch.tensor([r["features"] for r in records], dtype=torch.float32, device=device)
    y = torch.tensor([1 - r["correct"] for r in records], dtype=torch.float32, device=device)

    mean = X.mean(0)
    std = X.std(0, unbiased=False).clamp_min(1e-6)
    Xn = (X - mean) / std

    # Handle degenerate inner data without pretending to learn a classifier.
    if float(y.min()) == float(y.max()):
        return None, mean, std, float(y[0])

    torch.manual_seed(seed)
    model = RiskPredictor(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-2)

    pos = float(y.sum().item())
    neg = float(len(y) - pos)
    pos_weight = torch.tensor(
        max(1.0, neg / max(pos, 1.0)), device=device
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(500):
        opt.zero_grad(set_to_none=True)
        logits = model(Xn)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

    return model, mean, std, None


def predict_risk(model, mean, std, constant, features, device):
    if constant is not None:
        return constant
    x = torch.tensor(features, dtype=torch.float32, device=device)
    x = (x - mean) / std
    with torch.no_grad():
        return float(torch.sigmoid(model(x)).cpu())


def auc_roc(labels, scores):
    pos = [s for y,s in zip(labels,scores) if y == 1]
    neg = [s for y,s in zip(labels,scores) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def selective_metrics(records, thr):
    accepted = [r for r in records if r["risk"] < thr]
    rejected = [r for r in records if r["risk"] >= thr]
    n = len(records)
    wrong_total = sum(1-r["correct"] for r in records)
    wrong_acc = sum(1-r["correct"] for r in accepted)
    wrong_rej = sum(1-r["correct"] for r in rejected)
    return {
        "threshold": thr,
        "coverage": len(accepted)/n if n else 0.0,
        "accepted_accuracy": (
            sum(r["correct"] for r in accepted)/len(accepted)
            if accepted else 1.0
        ),
        "reject_rate": len(rejected)/n if n else 0.0,
        "wrong_accepted_rate": wrong_acc/n if n else 0.0,
        "error_capture_rate": wrong_rej/wrong_total if wrong_total else 1.0,
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
    ap.add_argument("--target-accuracy", type=float, default=0.95)
    ap.add_argument(
        "--output-dir",
        default="results/learned_semantic_risk_predictor_v1429",
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
    print(" LLM_GPU v1.4.29 Learned Semantic Risk Predictor / Unknown Rejection")
    print("="*118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Router           : adaptive 80% semantic mass")
    print("Risk model       : linear logistic predictor")
    print("Risk features    :", ", ".join(FEATURE_NAMES))
    print("Validation       : nested CV (outer 5-fold, inner 4-fold OOF)")
    print("Fallback         : 未学習です")
    print("Target accuracy  :", f"{args.target_accuracy:.1%}")
    print("Base model       : completely frozen")
    print()

    print("Encoding block5 token states...")
    states_all, labels_all = [], []
    for sample in CONTRAST_SAMPLES:
        states, offsets = encode_token_states(model, tok, sample["text"])
        states_all.append(states)
        labels_all.append(token_labels(sample, offsets))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_outer = []
    per_seed_auc = []
    per_thr = {t: [] for t in RISK_THRESHOLDS}

    print()
    print("Per-seed nested-CV results")
    print("-"*118)

    for seed in seeds:
        seed_records = []
        for outer_fold, test in enumerate(build_folds()):
            test_set = set(test)
            outer_train = [i for i in range(len(CONTRAST_SAMPLES)) if i not in test_set]

            risk_train = make_inner_oof(
                outer_train, states_all, labels_all, args,
                seed + 1000000 * outer_fold, device
            )
            risk_model, rmean, rstd, constant = fit_risk_predictor(
                risk_train, seed + 2000000 * outer_fold, device
            )

            outer_router = train_router(
                states_all, labels_all, outer_train, args,
                seed + 3000000 * outer_fold, device
            )

            for i in test:
                r = router_record(i, states_all, outer_router, device)
                r["risk"] = predict_risk(
                    risk_model, rmean, rstd, constant, r["features"], device
                )
                r["seed"] = seed
                r["outer_fold"] = outer_fold
                seed_records.append(r)

        labels = [1-r["correct"] for r in seed_records]
        scores = [r["risk"] for r in seed_records]
        auc = auc_roc(labels, scores)
        per_seed_auc.append(auc)

        base_acc = statistics.mean(r["correct"] for r in seed_records)

        best = None
        for t in RISK_THRESHOLDS:
            m = selective_metrics(seed_records, t)
            per_thr[t].append(m)
            if m["accepted_accuracy"] >= args.target_accuracy and m["coverage"] > 0:
                if best is None or m["coverage"] > best["coverage"]:
                    best = m

        if best is None:
            print(
                f"seed={seed:>3d} baseline={base_acc:.1%} "
                f"riskAUC={auc:.3f} target NOT REACHED"
            )
        else:
            print(
                f"seed={seed:>3d} baseline={base_acc:.1%} riskAUC={auc:.3f} "
                f"thr={best['threshold']:.2f} coverage={best['coverage']:.1%} "
                f"accepted={best['accepted_accuracy']:.1%}"
            )

        all_outer.extend(seed_records)

    # Persist per-example outer-holdout predictions.
    with (out/"per_example.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "seed","outer_fold","sample_index","text","gold","prediction",
            "correct","risk"
        ] + FEATURE_NAMES
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_outer:
            row = {k:r[k] for k in [
                "seed","outer_fold","sample_index","text","gold",
                "prediction","correct","risk"
            ]}
            row.update(dict(zip(FEATURE_NAMES, r["features"])))
            w.writerow(row)

    summary = []
    for t in RISK_THRESHOLDS:
        row = {"threshold": t}
        for key in [
            "coverage","accepted_accuracy","reject_rate",
            "wrong_accepted_rate","error_capture_rate"
        ]:
            m,s = mean_std([x[key] for x in per_thr[t]])
            row[key+"_mean"] = m
            row[key+"_std"] = s
        summary.append(row)

    with (out/"threshold_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(summary[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    auc_m, auc_s = mean_std(per_seed_auc)
    candidates = [
        r for r in summary
        if r["accepted_accuracy_mean"] >= args.target_accuracy
        and r["coverage_mean"] > 0
    ]
    if candidates:
        rec = max(candidates, key=lambda r:r["coverage_mean"])
    else:
        rec = max(summary, key=lambda r:r["accepted_accuracy_mean"])

    baseline_acc = statistics.mean(r["correct"] for r in all_outer)

    print()
    print("Risk-threshold sweep")
    print("-"*118)
    print(
        f"{'thr':>5s} {'coverage':>12s} {'accepted acc':>14s} "
        f"{'reject':>12s} {'wrong accepted':>16s} {'error capture':>14s}"
    )
    for r in summary:
        print(
            f"{r['threshold']:5.2f} "
            f"{r['coverage_mean']:>8.1%}±{r['coverage_std']:<5.1%} "
            f"{r['accepted_accuracy_mean']:>9.1%}±{r['accepted_accuracy_std']:<5.1%} "
            f"{r['reject_rate_mean']:>8.1%}±{r['reject_rate_std']:<5.1%} "
            f"{r['wrong_accepted_rate_mean']:>11.1%}±{r['wrong_accepted_rate_std']:<5.1%} "
            f"{r['error_capture_rate_mean']:>9.1%}±{r['error_capture_rate_std']:<5.1%}"
        )

    print()
    print("Risk discrimination")
    print("-"*72)
    print(f"Nested outer-holdout error AUROC : {auc_m:.3f} ± {auc_s:.3f}")
    print(f"Baseline router accuracy         : {baseline_acc:.1%}")
    print()
    print("Recommended operating point")
    print("-"*72)
    print(f"Risk threshold                   : {rec['threshold']:.2f}")
    print(f"Coverage                         : {rec['coverage_mean']:.1%} ± {rec['coverage_std']:.1%}")
    print(f"Accepted accuracy                : {rec['accepted_accuracy_mean']:.1%} ± {rec['accepted_accuracy_std']:.1%}")
    print(f"Reject rate ('未学習です')       : {rec['reject_rate_mean']:.1%} ± {rec['reject_rate_std']:.1%}")
    print(f"Wrong-accepted rate              : {rec['wrong_accepted_rate_mean']:.1%} ± {rec['wrong_accepted_rate_std']:.1%}")
    print(f"Error capture rate               : {rec['error_capture_rate_mean']:.1%} ± {rec['error_capture_rate_std']:.1%}")
    print()
    if rec["accepted_accuracy_mean"] >= args.target_accuracy:
        print("Conclusion: semantic-quality-aware rejection reaches the target accepted accuracy.")
    else:
        print("Conclusion: semantic-quality-aware rejection does not reach the target on this benchmark.")
    print()
    print("Runtime rule")
    print("------------")
    print(f"if predicted_error_risk >= {rec['threshold']:.2f}: return '未学習です'")
    print("else: route to FIRST/SECOND")
    print()
    print("Reference")
    print("---------")
    print("v1.4.28 margin-only rejection best accepted accuracy : 84.1%")
    print("v1.4.26 mass-80 baseline                            : 83.9% ± 1.7%")
    print("v1.4.26 gold-mask upper bound                       : 99.9% ± 0.6%")
    print("Output dir:", out)


if __name__ == "__main__":
    main()
