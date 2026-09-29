# train_calibrated_unknown_threshold_v1432.py
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
CALIBRATION_FRACTION = 0.25
COVERAGE_TARGETS = [0.10, 0.20, 0.30, 0.40]
TARGET_ACCURACY = 0.95


def split_risk_train_calibration(indices, fold, seed):
    # Deterministic rotation; calibration examples never train the router/risk model.
    ordered = list(indices)
    shift = (seed + 3 * fold) % len(ordered)
    ordered = ordered[shift:] + ordered[:shift]
    n_cal = max(4, int(round(len(ordered) * CALIBRATION_FRACTION)))
    calibration = ordered[:n_cal]
    risk_train = ordered[n_cal:]
    return risk_train, calibration


def ensemble_risk_for_index(i, states_all, router, ensemble, device):
    rec = router_record(i, states_all, router, device)
    risks = []
    for model_r, rmean, rstd, constant in ensemble:
        risks.append(
            predict_risk(
                model_r, rmean, rstd, constant,
                rec["features"], device
            )
        )
    rec["member_risks"] = risks
    rec["risk_mean"] = statistics.mean(risks)
    rec["risk_std"] = statistics.pstdev(risks) if len(risks) > 1 else 0.0
    return rec


def cutoff_for_target(calibration_records, target):
    vals = sorted(r["risk_mean"] for r in calibration_records)
    k = max(1, min(len(vals), int(math.ceil(target * len(vals)))))
    return vals[k - 1]


def evaluate_threshold(records, cutoff):
    accepted = [r for r in records if r["risk_mean"] <= cutoff]
    rejected = [r for r in records if r["risk_mean"] > cutoff]
    n = len(records)
    wrong_total = sum(1-r["correct"] for r in records)
    wrong_acc = sum(1-r["correct"] for r in accepted)
    wrong_rej = sum(1-r["correct"] for r in rejected)

    return {
        "cutoff": cutoff,
        "coverage": len(accepted)/n if n else 0.0,
        "accepted_accuracy": (
            sum(r["correct"] for r in accepted)/len(accepted)
            if accepted else 1.0
        ),
        "reject_rate": len(rejected)/n if n else 0.0,
        "wrong_accepted_rate": wrong_acc/n if n else 0.0,
        "error_capture_rate": wrong_rej/wrong_total if wrong_total else 1.0,
        "accepted_count": len(accepted),
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
        default="results/train_calibrated_unknown_threshold_v1432",
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
    print(" LLM_GPU v1.4.32 Train-Calibrated Unknown Threshold / Deployment Simulation")
    print("="*118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Router           : adaptive 80% semantic mass")
    print("Risk ensemble    :", args.ensemble_size, "logistic predictors")
    print("Calibration      : held-out subset of outer-train only")
    print("Calibration frac :", f"{CALIBRATION_FRACTION:.0%}")
    print("Coverage targets :", ", ".join(f"{x:.0%}" for x in COVERAGE_TARGETS))
    print("Target accuracy  :", f"{args.target_accuracy:.1%}")
    print("Fallback         : 未学習です")
    print("Outer test used for calibration: NEVER")
    print()

    print("Encoding block5 token states...")
    states_all, labels_all = [], []
    for sample in CONTRAST_SAMPLES:
        states, offsets = encode_token_states(model, tok, sample["text"])
        states_all.append(states)
        labels_all.append(token_labels(sample, offsets))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_target = {c: [] for c in COVERAGE_TARGETS}
    all_test_rows = []
    per_seed_auc = []

    print()
    print("Per-seed deployment simulation")
    print("-"*118)

    for seed in seeds:
        seed_test = []
        seed_policy = {c: [] for c in COVERAGE_TARGETS}

        for outer_fold, test in enumerate(build_folds()):
            test_set = set(test)
            outer_train = [
                i for i in range(len(CONTRAST_SAMPLES))
                if i not in test_set
            ]

            risk_train, calibration = split_risk_train_calibration(
                outer_train, outer_fold, seed
            )

            # Router and risk models never see calibration or outer test examples.
            router = train_router(
                states_all, labels_all, risk_train, args,
                seed + 5000000 * outer_fold, device
            )

            ensemble = []
            for member in range(args.ensemble_size):
                member_seed = (
                    seed
                    + 100000 * member
                    + 1000000 * outer_fold
                )
                risk_oof = make_inner_oof(
                    risk_train,
                    states_all,
                    labels_all,
                    args,
                    member_seed,
                    device,
                )
                ensemble.append(
                    fit_risk_predictor(
                        risk_oof,
                        member_seed + 777,
                        device,
                    )
                )

            calibration_records = [
                ensemble_risk_for_index(
                    i, states_all, router, ensemble, device
                )
                for i in calibration
            ]
            test_records = [
                ensemble_risk_for_index(
                    i, states_all, router, ensemble, device
                )
                for i in test
            ]

            for r in test_records:
                r["seed"] = seed
                r["outer_fold"] = outer_fold
                seed_test.append(r)
                all_test_rows.append(r)

            for target in COVERAGE_TARGETS:
                cutoff = cutoff_for_target(calibration_records, target)
                m = evaluate_threshold(test_records, cutoff)
                m["target_coverage"] = target
                m["seed"] = seed
                m["outer_fold"] = outer_fold
                seed_policy[target].append(m)

        labels = [1-r["correct"] for r in seed_test]
        scores = [r["risk_mean"] for r in seed_test]
        auc = auc_roc(labels, scores)
        per_seed_auc.append(auc)

        base_acc = statistics.mean(r["correct"] for r in seed_test)
        print(
            f"seed={seed:>3d} baseline={base_acc:.1%} AUC={auc:.3f}",
            end=""
        )

        for target in COVERAGE_TARGETS:
            folds = seed_policy[target]
            agg = {
                "coverage": statistics.mean(x["coverage"] for x in folds),
                "accepted_accuracy": statistics.mean(x["accepted_accuracy"] for x in folds),
                "reject_rate": statistics.mean(x["reject_rate"] for x in folds),
                "wrong_accepted_rate": statistics.mean(x["wrong_accepted_rate"] for x in folds),
                "error_capture_rate": statistics.mean(x["error_capture_rate"] for x in folds),
                "cutoff": statistics.mean(x["cutoff"] for x in folds),
            }
            per_target[target].append(agg)
            print(
                f" | {target:.0%}:cov={agg['coverage']:.1%}/acc={agg['accepted_accuracy']:.1%}",
                end=""
            )
        print()

    with (out/"per_example.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "seed","outer_fold","sample_index","text","gold","prediction",
            "correct","risk_mean","risk_std"
        ] + [f"risk_member_{i+1}" for i in range(args.ensemble_size)]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_test_rows:
            row = {k:r[k] for k in [
                "seed","outer_fold","sample_index","text","gold",
                "prediction","correct","risk_mean","risk_std"
            ]}
            for j,x in enumerate(r["member_risks"]):
                row[f"risk_member_{j+1}"] = x
            w.writerow(row)

    summary = []
    for target in COVERAGE_TARGETS:
        row = {"target_coverage": target}
        for key in [
            "coverage","accepted_accuracy","reject_rate",
            "wrong_accepted_rate","error_capture_rate","cutoff"
        ]:
            m,s = mean_std([x[key] for x in per_target[target]])
            row[key+"_mean"] = m
            row[key+"_std"] = s
        summary.append(row)

    with (out/"deployment_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(summary[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    auc_m, auc_s = mean_std(per_seed_auc)
    baseline = statistics.mean(r["correct"] for r in all_test_rows)

    feasible = [
        r for r in summary
        if r["accepted_accuracy_mean"] >= args.target_accuracy
        and r["coverage_mean"] > 0
    ]
    if feasible:
        rec = max(feasible, key=lambda r:r["coverage_mean"])
    else:
        rec = max(
            summary,
            key=lambda r:(r["accepted_accuracy_mean"], r["coverage_mean"])
        )

    print()
    print("Train-calibrated deployment summary")
    print("-"*118)
    print(
        f"{'target':>7s} {'realized cov':>14s} {'accepted acc':>14s} "
        f"{'reject':>12s} {'wrong accepted':>16s} "
        f"{'error capture':>14s} {'cutoff':>10s}"
    )
    for r in summary:
        print(
            f"{r['target_coverage']:>6.0%} "
            f"{r['coverage_mean']:>9.1%}±{r['coverage_std']:<5.1%} "
            f"{r['accepted_accuracy_mean']:>9.1%}±{r['accepted_accuracy_std']:<5.1%} "
            f"{r['reject_rate_mean']:>8.1%}±{r['reject_rate_std']:<5.1%} "
            f"{r['wrong_accepted_rate_mean']:>11.1%}±{r['wrong_accepted_rate_std']:<5.1%} "
            f"{r['error_capture_rate_mean']:>9.1%}±{r['error_capture_rate_std']:<5.1%} "
            f"{r['cutoff_mean']:.4f}"
        )

    print()
    print("Risk discrimination")
    print("-"*72)
    print(f"Outer-test error AUROC            : {auc_m:.3f} ± {auc_s:.3f}")
    print(f"Baseline router accuracy          : {baseline:.1%}")

    print()
    print("Recommended deployable operating point")
    print("-"*72)
    print(f"Calibration target coverage       : {rec['target_coverage']:.0%}")
    print(f"Realized outer-test coverage      : {rec['coverage_mean']:.1%} ± {rec['coverage_std']:.1%}")
    print(f"Accepted accuracy                 : {rec['accepted_accuracy_mean']:.1%} ± {rec['accepted_accuracy_std']:.1%}")
    print(f"Reject rate ('未学習です')        : {rec['reject_rate_mean']:.1%} ± {rec['reject_rate_std']:.1%}")
    print(f"Wrong-accepted rate               : {rec['wrong_accepted_rate_mean']:.1%} ± {rec['wrong_accepted_rate_std']:.1%}")
    print(f"Error capture rate                : {rec['error_capture_rate_mean']:.1%} ± {rec['error_capture_rate_std']:.1%}")
    print(f"Train-calibrated cutoff           : {rec['cutoff_mean']:.4f} ± {rec['cutoff_std']:.4f}")

    print()
    if rec["accepted_accuracy_mean"] >= args.target_accuracy:
        print("Conclusion: train-only calibration reaches the target accepted accuracy on unseen outer test data.")
    else:
        print("Conclusion: train-only calibration does not reach the target accepted accuracy on unseen outer test data.")

    print()
    print("Deployment rule")
    print("---------------")
    print(
        "For each trained deployment instance, compute the cutoff from its "
        "held-out calibration set only."
    )
    print("if ensemble_mean_risk <= calibrated_cutoff: route to FIRST/SECOND")
    print("else: return '未学習です'")

    print()
    print("Reference")
    print("---------")
    print("v1.4.31 test-ranked proof-of-concept : 95.6% at 20% coverage")
    print("v1.4.30 stable ensemble             : 94.5% at 30.8% coverage")
    print("Output dir:", out)


if __name__ == "__main__":
    main()
