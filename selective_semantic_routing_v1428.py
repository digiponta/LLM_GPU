# selective_semantic_routing_v1428.py
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
from adaptive_semantic_window_v1420 import (
    role_scores,
    position_probs,
    mass_window,
)

MASS = 0.80
CONF_THRESHOLDS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def confidence_from_fused(fused):
    total = float(fused.sum().detach())
    if total <= 1e-12:
        return 0.0
    margin = float(torch.abs(fused[1] - fused[0]).detach())
    return margin / total


def eval_seed(states_all, labels_all, args, seed, device):
    records = []

    for fold, test in enumerate(build_folds()):
        test_set = set(test)
        train = [i for i in range(len(CONTRAST_SAMPLES)) if i not in test_set]

        sel_tagger = train_binary_span_tagger(
            [states_all[i] for i in train],
            [labels_all[i] for i in train],
            SELECTED, args, seed + 1000 * fold, device,
        )
        neg_tagger = train_binary_span_tagger(
            [states_all[i] for i in train],
            [labels_all[i] for i in train],
            NEGATED, args, seed + 10000 + 1000 * fold, device,
        )
        explicit_head, emean, estd = train_explicitness_head(
            [states_all[i] for i in train],
            [labels_all[i] for i in train],
            args, seed + 20000 + 1000 * fold, device,
        )

        position_vectors, position_labels = [], []
        for i in train:
            sel = pooled_gold_role_vector(states_all[i], labels_all[i], SELECTED)
            neg = pooled_gold_role_vector(states_all[i], labels_all[i], NEGATED)
            if sel is not None:
                position_vectors.append(sel)
                position_labels.append(CONTRAST_SAMPLES[i]["selected_position"])
            if neg is not None:
                position_vectors.append(neg)
                position_labels.append(CONTRAST_SAMPLES[i]["negated_position"])

        pos_head, pmean, pstd = train_position_head(
            position_vectors, position_labels, args,
            seed + 30000 + 1000 * fold, device,
        )

        for i in test:
            states = states_all[i]
            sample = CONTRAST_SAMPLES[i]

            sel_score = role_scores(states, sel_tagger, device)
            neg_score = role_scores(states, neg_tagger, device)
            exp_hat, exp_conf = predict_explicit(
                states, explicit_head, emean, estd, device
            )

            sel_vec, _, _ = mass_window(states, sel_score, MASS)
            fused = position_probs(
                sel_vec, pos_head, pmean, pstd
            ).clone()

            if exp_hat == 1:
                neg_vec, _, _ = mass_window(states, neg_score, MASS)
                np = position_probs(
                    neg_vec, pos_head, pmean, pstd
                )
                fused[0] += np[1]
                fused[1] += np[0]

            pred = int(fused.argmax())
            gold = int(sample["selected_position"])
            conf = confidence_from_fused(fused)

            records.append({
                "seed": seed,
                "fold": fold,
                "sample_index": i,
                "text": sample["text"],
                "gold": gold,
                "prediction": pred,
                "correct": int(pred == gold),
                "confidence": conf,
                "explicit_hat": int(exp_hat),
                "explicit_confidence": float(exp_conf),
            })

    return records


def metrics_at_threshold(records, threshold):
    accepted = [r for r in records if r["confidence"] >= threshold]
    rejected = [r for r in records if r["confidence"] < threshold]

    n = len(records)
    a = len(accepted)
    correct_accepted = sum(r["correct"] for r in accepted)
    wrong_accepted = a - correct_accepted

    coverage = a / n if n else 0.0
    reject_rate = len(rejected) / n if n else 0.0
    accepted_accuracy = correct_accepted / a if a else 1.0
    wrong_accepted_rate = wrong_accepted / n if n else 0.0
    selective_risk = wrong_accepted / a if a else 0.0

    wrong_rejected = sum(1 - r["correct"] for r in rejected)
    total_wrong = sum(1 - r["correct"] for r in records)
    error_capture = wrong_rejected / total_wrong if total_wrong else 1.0

    return {
        "threshold": threshold,
        "coverage": coverage,
        "accepted_accuracy": accepted_accuracy,
        "reject_rate": reject_rate,
        "wrong_accepted_rate": wrong_accepted_rate,
        "selective_risk": selective_risk,
        "error_capture_rate": error_capture,
        "accepted_count": a,
        "rejected_count": len(rejected),
        "wrong_accepted_count": wrong_accepted,
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
        default="results/selective_semantic_routing_v1428",
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
    print(" LLM_GPU v1.4.28 Selective Semantic Routing / Unknown Rejection")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Semantic field   : adaptive 80% role-score mass")
    print("Confidence       : normalized FIRST/SECOND fused probability margin")
    print("Fallback         : 未学習です")
    print("Evaluation       : 5-fold unseen-paraphrase holdout")
    print("Relation seeds   :", f"{seeds[0]}..{seeds[-1]}")
    print("Target accuracy  :", f"{args.target_accuracy:.1%}")
    print("Base model       : completely frozen")
    print()

    print("Encoding block5 token states...")
    states_all, labels_all = [], []
    for sample in CONTRAST_SAMPLES:
        states, offsets = encode_token_states(model, tok, sample["text"])
        labels = token_labels(sample, offsets)
        states_all.append(states)
        labels_all.append(labels)

    all_records = []
    per_seed_metrics = {t: [] for t in CONF_THRESHOLDS}

    print()
    print("Per-seed baseline / selective routing")
    print("-" * 118)

    for seed in seeds:
        records = eval_seed(states_all, labels_all, args, seed, device)
        all_records.extend(records)

        base_acc = statistics.mean(r["correct"] for r in records)
        seed_best = None
        for t in CONF_THRESHOLDS:
            m = metrics_at_threshold(records, t)
            per_seed_metrics[t].append(m)
            if m["accepted_accuracy"] >= args.target_accuracy and m["accepted_count"] > 0:
                if seed_best is None or m["coverage"] > seed_best["coverage"]:
                    seed_best = m

        if seed_best is None:
            print(
                f"seed={seed:>3d} baseline={base_acc:.1%} "
                f"target={args.target_accuracy:.1%} NOT REACHED"
            )
        else:
            print(
                f"seed={seed:>3d} baseline={base_acc:.1%} "
                f"thr={seed_best['threshold']:.2f} "
                f"coverage={seed_best['coverage']:.1%} "
                f"accepted_acc={seed_best['accepted_accuracy']:.1%} "
                f"reject={seed_best['reject_rate']:.1%}"
            )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "per_example.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = list(all_records[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_records)

    summary_rows = []
    for t in CONF_THRESHOLDS:
        ms = per_seed_metrics[t]
        row = {"threshold": t}
        for key in [
            "coverage",
            "accepted_accuracy",
            "reject_rate",
            "wrong_accepted_rate",
            "selective_risk",
            "error_capture_rate",
        ]:
            vals = [m[key] for m in ms]
            mean, std = mean_std(vals)
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        summary_rows.append(row)

    with (out / "threshold_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    candidates = [
        row for row in summary_rows
        if row["accepted_accuracy_mean"] >= args.target_accuracy
        and row["coverage_mean"] > 0
    ]
    if candidates:
        recommended = max(candidates, key=lambda r: r["coverage_mean"])
    else:
        recommended = max(summary_rows, key=lambda r: r["accepted_accuracy_mean"])

    baseline = summary_rows[0]

    print()
    print("Coverage / accuracy sweep")
    print("-" * 118)
    print(
        f"{'thr':>5s} {'coverage':>12s} {'accepted acc':>14s} "
        f"{'reject':>12s} {'wrong accepted':>16s} {'error capture':>14s}"
    )
    for row in summary_rows:
        print(
            f"{row['threshold']:5.2f} "
            f"{row['coverage_mean']:>8.1%}±{row['coverage_std']:<5.1%} "
            f"{row['accepted_accuracy_mean']:>9.1%}±{row['accepted_accuracy_std']:<5.1%} "
            f"{row['reject_rate_mean']:>8.1%}±{row['reject_rate_std']:<5.1%} "
            f"{row['wrong_accepted_rate_mean']:>11.1%}±{row['wrong_accepted_rate_std']:<5.1%} "
            f"{row['error_capture_rate_mean']:>9.1%}±{row['error_capture_rate_std']:<5.1%}"
        )

    print()
    print("Recommended operating point")
    print("-" * 72)
    print(f"Threshold                        : {recommended['threshold']:.2f}")
    print(f"Coverage                         : {recommended['coverage_mean']:.1%} ± {recommended['coverage_std']:.1%}")
    print(f"Accepted accuracy                : {recommended['accepted_accuracy_mean']:.1%} ± {recommended['accepted_accuracy_std']:.1%}")
    print(f"Reject rate ('未学習です')       : {recommended['reject_rate_mean']:.1%} ± {recommended['reject_rate_std']:.1%}")
    print(f"Wrong-accepted rate              : {recommended['wrong_accepted_rate_mean']:.1%} ± {recommended['wrong_accepted_rate_std']:.1%}")
    print(f"Error capture rate               : {recommended['error_capture_rate_mean']:.1%} ± {recommended['error_capture_rate_std']:.1%}")
    print()
    print(f"No-rejection baseline accuracy   : {baseline['accepted_accuracy_mean']:.1%} ± {baseline['accepted_accuracy_std']:.1%}")
    print()
    if recommended["accepted_accuracy_mean"] >= args.target_accuracy:
        print(
            "Conclusion: target accepted accuracy is reachable by rejecting "
            "low-confidence cases as '未学習です'."
        )
    else:
        print(
            "Conclusion: the requested target accepted accuracy is not reached "
            "by margin-only rejection on this benchmark."
        )
    print()
    print("Runtime rule")
    print("------------")
    print(
        f"if confidence < {recommended['threshold']:.2f}: return '未学習です'"
    )
    print("else: route to FIRST/SECOND")
    print()
    print("Reference")
    print("---------")
    print("v1.4.26 learned / mass-80             : 83.9% ± 1.7%")
    print("v1.4.26 gold-mask upper bound         : 99.9% ± 0.6%")
    print("Output dir:", out)


if __name__ == "__main__":
    main()
