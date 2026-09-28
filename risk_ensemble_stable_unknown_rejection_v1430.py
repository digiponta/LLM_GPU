# risk_ensemble_stable_unknown_rejection_v1430.py
from __future__ import annotations

import argparse
import csv
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
RISK_THRESHOLDS = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]
AGREEMENTS = [0.60,0.80,1.00]


def selective_metrics(records, risk_thr, agreement_thr):
    accepted, rejected = [], []
    for r in records:
        safe_votes = sum(x < risk_thr for x in r["member_risks"])
        agreement = safe_votes / len(r["member_risks"])
        safe = (r["risk_mean"] < risk_thr) and (agreement >= agreement_thr)
        (accepted if safe else rejected).append(r)

    n = len(records)
    wrong_total = sum(1-r["correct"] for r in records)
    wrong_acc = sum(1-r["correct"] for r in accepted)
    wrong_rej = sum(1-r["correct"] for r in rejected)

    return {
        "risk_threshold": risk_thr,
        "agreement_threshold": agreement_thr,
        "coverage": len(accepted)/n if n else 0.0,
        "accepted_accuracy": (
            sum(r["correct"] for r in accepted)/len(accepted)
            if accepted else 1.0
        ),
        "reject_rate": len(rejected)/n if n else 0.0,
        "wrong_accepted_rate": wrong_acc/n if n else 0.0,
        "error_capture_rate": wrong_rej/wrong_total if wrong_total else 1.0,
        "accepted_risk_std": (
            statistics.mean(r["risk_std"] for r in accepted)
            if accepted else 0.0
        ),
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
    ap.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE)
    ap.add_argument(
        "--output-dir",
        default="results/risk_ensemble_stable_unknown_rejection_v1430",
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
    print(" LLM_GPU v1.4.30 Risk Ensemble / Stable Unknown Rejection")
    print("="*118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Router           : adaptive 80% semantic mass")
    print("Risk ensemble    :", args.ensemble_size, "nested-CV logistic predictors")
    print("Decision         : mean risk + safe-vote agreement")
    print("Agreements       :", ", ".join(f"{x:.0%}" for x in AGREEMENTS))
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

    all_outer = []
    per_seed_auc = []
    per_policy = {(t,a): [] for t in RISK_THRESHOLDS for a in AGREEMENTS}

    print()
    print("Per-seed ensemble results")
    print("-"*118)

    for seed in seeds:
        seed_records = []

        for outer_fold, test in enumerate(build_folds()):
            test_set = set(test)
            outer_train = [i for i in range(len(CONTRAST_SAMPLES)) if i not in test_set]

            ensemble = []
            for member in range(args.ensemble_size):
                member_seed = seed + 100000 * member + 1000000 * outer_fold
                risk_train = make_inner_oof(
                    outer_train, states_all, labels_all, args,
                    member_seed, device
                )
                ensemble.append(
                    fit_risk_predictor(
                        risk_train,
                        member_seed + 777,
                        device
                    )
                )

            outer_router = train_router(
                states_all, labels_all, outer_train, args,
                seed + 9000000 * outer_fold, device
            )

            for i in test:
                r = router_record(i, states_all, outer_router, device)
                risks = []
                for model_r, rmean, rstd, constant in ensemble:
                    risks.append(
                        predict_risk(
                            model_r, rmean, rstd, constant,
                            r["features"], device
                        )
                    )

                r["member_risks"] = risks
                r["risk_mean"] = statistics.mean(risks)
                r["risk_std"] = statistics.pstdev(risks) if len(risks) > 1 else 0.0
                r["seed"] = seed
                r["outer_fold"] = outer_fold
                seed_records.append(r)

        labels = [1-r["correct"] for r in seed_records]
        scores = [r["risk_mean"] for r in seed_records]
        auc = auc_roc(labels, scores)
        per_seed_auc.append(auc)

        base_acc = statistics.mean(r["correct"] for r in seed_records)
        best = None

        for t in RISK_THRESHOLDS:
            for a in AGREEMENTS:
                m = selective_metrics(seed_records, t, a)
                per_policy[(t,a)].append(m)
                if m["accepted_accuracy"] >= args.target_accuracy and m["coverage"] > 0:
                    if best is None or m["coverage"] > best["coverage"]:
                        best = m

        if best is None:
            print(
                f"seed={seed:>3d} baseline={base_acc:.1%} "
                f"ensembleAUC={auc:.3f} target NOT REACHED"
            )
        else:
            print(
                f"seed={seed:>3d} baseline={base_acc:.1%} ensembleAUC={auc:.3f} "
                f"risk<{best['risk_threshold']:.2f} agree>={best['agreement_threshold']:.0%} "
                f"coverage={best['coverage']:.1%} accepted={best['accepted_accuracy']:.1%}"
            )

        all_outer.extend(seed_records)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out/"per_example.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "seed","outer_fold","sample_index","text","gold","prediction",
            "correct","risk_mean","risk_std"
        ] + [f"risk_member_{i+1}" for i in range(args.ensemble_size)]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_outer:
            row = {k:r[k] for k in [
                "seed","outer_fold","sample_index","text","gold",
                "prediction","correct","risk_mean","risk_std"
            ]}
            for j,x in enumerate(r["member_risks"]):
                row[f"risk_member_{j+1}"] = x
            w.writerow(row)

    summary = []
    for t in RISK_THRESHOLDS:
        for a in AGREEMENTS:
            row = {"risk_threshold":t, "agreement_threshold":a}
            for key in [
                "coverage","accepted_accuracy","reject_rate",
                "wrong_accepted_rate","error_capture_rate","accepted_risk_std"
            ]:
                m,s = mean_std([x[key] for x in per_policy[(t,a)]])
                row[key+"_mean"] = m
                row[key+"_std"] = s
            summary.append(row)

    with (out/"policy_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
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
        rec = max(
            candidates,
            key=lambda r: (r["coverage_mean"], -r["accepted_risk_std_mean"])
        )
    else:
        rec = max(
            summary,
            key=lambda r: (
                r["accepted_accuracy_mean"],
                r["coverage_mean"],
                -r["accepted_risk_std_mean"],
            )
        )

    baseline = statistics.mean(r["correct"] for r in all_outer)

    print()
    print("Policy sweep")
    print("-"*118)
    print(
        f"{'risk':>5s} {'agree':>7s} {'coverage':>12s} {'accepted acc':>14s} "
        f"{'reject':>12s} {'wrong accepted':>16s} {'error capture':>14s} {'risk std':>10s}"
    )
    for r in summary:
        print(
            f"{r['risk_threshold']:5.2f} {r['agreement_threshold']:>6.0%} "
            f"{r['coverage_mean']:>8.1%}±{r['coverage_std']:<5.1%} "
            f"{r['accepted_accuracy_mean']:>9.1%}±{r['accepted_accuracy_std']:<5.1%} "
            f"{r['reject_rate_mean']:>8.1%}±{r['reject_rate_std']:<5.1%} "
            f"{r['wrong_accepted_rate_mean']:>11.1%}±{r['wrong_accepted_rate_std']:<5.1%} "
            f"{r['error_capture_rate_mean']:>9.1%}±{r['error_capture_rate_std']:<5.1%} "
            f"{r['accepted_risk_std_mean']:.3f}"
        )

    print()
    print("Risk discrimination")
    print("-"*72)
    print(f"Ensemble outer-holdout error AUROC : {auc_m:.3f} ± {auc_s:.3f}")
    print(f"Baseline router accuracy           : {baseline:.1%}")
    print()
    print("Recommended stable operating point")
    print("-"*72)
    print(f"Mean risk threshold                : {rec['risk_threshold']:.2f}")
    print(f"Safe-vote agreement                : {rec['agreement_threshold']:.0%}")
    print(f"Coverage                           : {rec['coverage_mean']:.1%} ± {rec['coverage_std']:.1%}")
    print(f"Accepted accuracy                  : {rec['accepted_accuracy_mean']:.1%} ± {rec['accepted_accuracy_std']:.1%}")
    print(f"Reject rate ('未学習です')         : {rec['reject_rate_mean']:.1%} ± {rec['reject_rate_std']:.1%}")
    print(f"Wrong-accepted rate                : {rec['wrong_accepted_rate_mean']:.1%} ± {rec['wrong_accepted_rate_std']:.1%}")
    print(f"Error capture rate                 : {rec['error_capture_rate_mean']:.1%} ± {rec['error_capture_rate_std']:.1%}")
    print(f"Accepted ensemble risk std         : {rec['accepted_risk_std_mean']:.3f} ± {rec['accepted_risk_std_std']:.3f}")
    print()
    if rec["accepted_accuracy_mean"] >= args.target_accuracy:
        print("Conclusion: ensemble agreement reaches the target accepted accuracy.")
    else:
        print("Conclusion: ensemble agreement improves stability but does not reach the target on this benchmark.")
    print()
    print("Runtime rule")
    print("------------")
    print(
        f"accept only if mean_risk < {rec['risk_threshold']:.2f} "
        f"and safe_vote_agreement >= {rec['agreement_threshold']:.0%}"
    )
    print("otherwise: return '未学習です'")
    print()
    print("Reference")
    print("---------")
    print("v1.4.29 single risk predictor AUROC                  : 0.562 ± 0.072")
    print("v1.4.29 accepted accuracy / coverage at risk<0.10   : 92.9% / 43.5%")
    print("v1.4.26 gold-mask upper bound                       : 99.9% ± 0.6%")
    print("Output dir:", out)


if __name__ == "__main__":
    main()
