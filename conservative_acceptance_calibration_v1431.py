# conservative_acceptance_calibration_v1431.py
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch

from semantic_contrast_decomposition_v1412 import CONTRAST_SAMPLES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer
from learned_semantic_span_detector_v1414 import encode_token_states, token_labels, build_folds
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
COVERAGE_TARGETS = [0.10, 0.20, 0.30, 0.40, 0.50]
TARGET_ACCURACY = 0.95


def evaluate_coverage(records, target):
    ranked = sorted(records, key=lambda r: (r["risk_mean"], r["risk_std"]))
    n = len(ranked)
    k = max(1, min(n, int(math.ceil(target * n))))
    accepted = ranked[:k]
    rejected = ranked[k:]

    correct = sum(r["correct"] for r in accepted)
    wrong = k - correct
    total_wrong = sum(1-r["correct"] for r in ranked)
    wrong_rejected = sum(1-r["correct"] for r in rejected)

    cutoff = accepted[-1]["risk_mean"]

    return {
        "target_coverage": target,
        "actual_coverage": k / n,
        "accepted_accuracy": correct / k,
        "reject_rate": len(rejected) / n,
        "wrong_accepted_rate": wrong / n,
        "error_capture_rate": (
            wrong_rejected / total_wrong if total_wrong else 1.0
        ),
        "risk_cutoff": cutoff,
        "accepted_risk_mean": statistics.mean(r["risk_mean"] for r in accepted),
        "accepted_risk_std_mean": statistics.mean(r["risk_std"] for r in accepted),
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
        default="results/conservative_acceptance_calibration_v1431",
    )
    args = ap.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("=" * 118)
    print(" LLM_GPU v1.4.31 Conservative Acceptance Calibration / Coverage Targeting")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Router           : adaptive 80% semantic mass")
    print("Risk ensemble    :", args.ensemble_size, "nested-CV logistic predictors")
    print("Coverage targets :", ", ".join(f"{x:.0%}" for x in COVERAGE_TARGETS))
    print("Selection        : lowest mean-risk cases first")
    print("Target accuracy  :", f"{args.target_accuracy:.1%}")
    print("Fallback         : 未学習です")
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
    per_cov = {c: [] for c in COVERAGE_TARGETS}

    print()
    print("Per-seed coverage calibration")
    print("-" * 118)

    for seed in seeds:
        seed_records = []

        for outer_fold, test in enumerate(build_folds()):
            test_set = set(test)
            outer_train = [
                i for i in range(len(CONTRAST_SAMPLES))
                if i not in test_set
            ]

            ensemble = []
            for member in range(args.ensemble_size):
                member_seed = (
                    seed
                    + 100000 * member
                    + 1000000 * outer_fold
                )
                risk_train = make_inner_oof(
                    outer_train,
                    states_all,
                    labels_all,
                    args,
                    member_seed,
                    device,
                )
                ensemble.append(
                    fit_risk_predictor(
                        risk_train,
                        member_seed + 777,
                        device,
                    )
                )

            outer_router = train_router(
                states_all,
                labels_all,
                outer_train,
                args,
                seed + 9000000 * outer_fold,
                device,
            )

            for i in test:
                r = router_record(i, states_all, outer_router, device)
                risks = []
                for model_r, rmean, rstd, constant in ensemble:
                    risks.append(
                        predict_risk(
                            model_r,
                            rmean,
                            rstd,
                            constant,
                            r["features"],
                            device,
                        )
                    )

                r["member_risks"] = risks
                r["risk_mean"] = statistics.mean(risks)
                r["risk_std"] = (
                    statistics.pstdev(risks)
                    if len(risks) > 1 else 0.0
                )
                r["seed"] = seed
                r["outer_fold"] = outer_fold
                seed_records.append(r)

        labels = [1-r["correct"] for r in seed_records]
        scores = [r["risk_mean"] for r in seed_records]
        auc = auc_roc(labels, scores)
        per_seed_auc.append(auc)

        print(
            f"seed={seed:>3d} baseline="
            f"{statistics.mean(r['correct'] for r in seed_records):.1%} "
            f"AUC={auc:.3f}",
            end=""
        )

        for cov in COVERAGE_TARGETS:
            m = evaluate_coverage(seed_records, cov)
            per_cov[cov].append(m)
            print(
                f" | {cov:.0%}:{m['accepted_accuracy']:.1%}",
                end=""
            )
        print()

        all_outer.extend(seed_records)

    with (out / "per_example.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = [
            "seed","outer_fold","sample_index","text","gold",
            "prediction","correct","risk_mean","risk_std"
        ] + [f"risk_member_{i+1}" for i in range(args.ensemble_size)]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_outer:
            row = {
                k: r[k]
                for k in [
                    "seed","outer_fold","sample_index","text","gold",
                    "prediction","correct","risk_mean","risk_std"
                ]
            }
            for j, x in enumerate(r["member_risks"]):
                row[f"risk_member_{j+1}"] = x
            w.writerow(row)

    summary = []
    for cov in COVERAGE_TARGETS:
        row = {"target_coverage": cov}
        for key in [
            "actual_coverage",
            "accepted_accuracy",
            "reject_rate",
            "wrong_accepted_rate",
            "error_capture_rate",
            "risk_cutoff",
            "accepted_risk_mean",
            "accepted_risk_std_mean",
        ]:
            m, s = mean_std([x[key] for x in per_cov[cov]])
            row[key + "_mean"] = m
            row[key + "_std"] = s
        summary.append(row)

    with (out / "coverage_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = list(summary[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    auc_m, auc_s = mean_std(per_seed_auc)

    feasible = [
        r for r in summary
        if r["accepted_accuracy_mean"] >= args.target_accuracy
    ]
    if feasible:
        recommended = max(
            feasible,
            key=lambda r: r["actual_coverage_mean"]
        )
    else:
        recommended = max(
            summary,
            key=lambda r: (
                r["accepted_accuracy_mean"],
                -r["actual_coverage_mean"],
            )
        )

    baseline_acc = statistics.mean(r["correct"] for r in all_outer)

    print()
    print("Coverage targeting summary")
    print("-" * 118)
    print(
        f"{'target':>7s} {'actual':>10s} {'accepted acc':>14s} "
        f"{'reject':>12s} {'wrong accepted':>16s} "
        f"{'error capture':>14s} {'risk cutoff':>12s}"
    )
    for r in summary:
        print(
            f"{r['target_coverage']:>6.0%} "
            f"{r['actual_coverage_mean']:>7.1%}±{r['actual_coverage_std']:<5.1%} "
            f"{r['accepted_accuracy_mean']:>9.1%}±{r['accepted_accuracy_std']:<5.1%} "
            f"{r['reject_rate_mean']:>8.1%}±{r['reject_rate_std']:<5.1%} "
            f"{r['wrong_accepted_rate_mean']:>11.1%}±{r['wrong_accepted_rate_std']:<5.1%} "
            f"{r['error_capture_rate_mean']:>9.1%}±{r['error_capture_rate_std']:<5.1%} "
            f"{r['risk_cutoff_mean']:.3f}"
        )

    print()
    print("Risk discrimination")
    print("-" * 72)
    print(
        f"Ensemble outer-holdout error AUROC : "
        f"{auc_m:.3f} ± {auc_s:.3f}"
    )
    print(f"Baseline router accuracy           : {baseline_acc:.1%}")

    print()
    print("Recommended coverage operating point")
    print("-" * 72)
    print(
        f"Target coverage                    : "
        f"{recommended['target_coverage']:.0%}"
    )
    print(
        f"Actual coverage                    : "
        f"{recommended['actual_coverage_mean']:.1%} ± "
        f"{recommended['actual_coverage_std']:.1%}"
    )
    print(
        f"Accepted accuracy                  : "
        f"{recommended['accepted_accuracy_mean']:.1%} ± "
        f"{recommended['accepted_accuracy_std']:.1%}"
    )
    print(
        f"Reject rate ('未学習です')         : "
        f"{recommended['reject_rate_mean']:.1%} ± "
        f"{recommended['reject_rate_std']:.1%}"
    )
    print(
        f"Wrong-accepted rate                : "
        f"{recommended['wrong_accepted_rate_mean']:.1%} ± "
        f"{recommended['wrong_accepted_rate_std']:.1%}"
    )
    print(
        f"Error capture rate                 : "
        f"{recommended['error_capture_rate_mean']:.1%} ± "
        f"{recommended['error_capture_rate_std']:.1%}"
    )
    print(
        f"Mean empirical risk cutoff         : "
        f"{recommended['risk_cutoff_mean']:.3f} ± "
        f"{recommended['risk_cutoff_std']:.3f}"
    )

    print()
    if recommended["accepted_accuracy_mean"] >= args.target_accuracy:
        print(
            "Conclusion: the target accepted accuracy is reached at the "
            "recommended conservative coverage level."
        )
    else:
        print(
            "Conclusion: even the most conservative tested coverage target "
            "does not reach the target accepted accuracy on average."
        )

    print()
    print("Runtime policy")
    print("--------------")
    print(
        "Rank candidate inputs by ensemble mean risk and answer only within "
        f"the lowest-risk ~{recommended['target_coverage']:.0%} region."
    )
    print("All other inputs return: '未学習です'")

    print()
    print("Reference")
    print("---------")
    print(
        "v1.4.30 stable ensemble best: "
        "94.5% accepted accuracy at 30.8% coverage"
    )
    print(
        "v1.4.29 single risk predictor: "
        "92.9% accepted accuracy at 43.5% coverage"
    )
    print("v1.4.26 gold-mask upper bound: 99.9% ± 0.6%")
    print("Output dir:", out)


if __name__ == "__main__":
    main()
