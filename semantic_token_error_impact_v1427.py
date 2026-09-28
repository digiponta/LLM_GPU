# semantic_token_error_impact_v1427.py
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
from automatic_semantic_decomposition_v1413 import eval_direct, eval_oracle
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
    train_explicitness_head,
    predict_explicit,
)
from adaptive_semantic_window_v1420 import position_probs


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def masked_mean(states, mask, device):
    mask = mask.bool()
    if not bool(mask.any()):
        return None
    return states.to(device)[mask.to(device)].mean(0)


def signed_margin(probs, gold_position):
    # Positive means confidence in the correct class.
    other = 1 - int(gold_position)
    return float((probs[int(gold_position)] - probs[other]).detach())


def evaluate_mask(states, mask, pos_head, pmean, pstd, gold_position, device):
    vec = masked_mean(states, mask, device)
    if vec is None:
        return None, None
    probs = position_probs(vec, pos_head, pmean, pstd)
    pred = int(probs.argmax())
    margin = signed_margin(probs, gold_position)
    return pred, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--seeds", default="1-20")
    ap.add_argument("--position-hidden", type=int, default=32)
    ap.add_argument("--explicit-hidden", type=int, default=32)
    ap.add_argument(
        "--output-dir",
        default="results/semantic_token_error_impact_v1427",
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
    print(" LLM_GPU v1.4.27 Semantic Token Error Impact / Harmful False-Positive Analysis")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Analysis         : gold mask + one FP / gold mask - one TP")
    print("Evaluation       : 5-fold unseen-paraphrase holdout")
    print("Relation seeds   :", f"{seeds[0]}..{seeds[-1]}")
    print("Base model       : completely frozen")
    print()

    print("Encoding block5 token states...")
    states_all, labels_all, offsets_all = [], [], []
    for sample in CONTRAST_SAMPLES:
        states, offsets = encode_token_states(model, tok, sample["text"])
        labels = token_labels(sample, offsets)
        states_all.append(states)
        labels_all.append(labels)
        offsets_all.append(offsets)

    from automatic_semantic_decomposition_v1413 import encode_cached
    cache = {}
    Xfull = torch.stack([
        encode_cached(model, tok, cache, s["text"])
        for s in CONTRAST_SAMPLES
    ]).to(device)
    Xneg = torch.stack([
        encode_cached(model, tok, cache, s["negated_text"])
        for s in CONTRAST_SAMPLES
    ]).to(device)
    Xsel = torch.stack([
        encode_cached(model, tok, cache, s["selected_text"])
        for s in CONTRAST_SAMPLES
    ]).to(device)
    ysel = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long, device=device,
    )
    yneg = torch.tensor(
        [s["negated_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long, device=device,
    )

    token_rows = []
    per_seed_rows = []

    for seed in seeds:
        fp_margin_deltas = []
        tp_margin_deltas = []
        harmful_fp_flags = []
        critical_tp_flags = []
        fp_flip_flags = []
        tp_flip_flags = []

        for fold, test in enumerate(build_folds()):
            test_set = set(test)
            train = [i for i in range(len(CONTRAST_SAMPLES)) if i not in test_set]

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
                labels = labels_all[i]
                sample = CONTRAST_SAMPLES[i]
                exp_hat, _ = predict_explicit(
                    states, explicit_head, emean, estd, device
                )

                for role, gold_position, role_name in [
                    (SELECTED, sample["selected_position"], "SELECTED"),
                    (NEGATED, sample["negated_position"], "NEGATED"),
                ]:
                    gold_mask = labels == role
                    if not bool(gold_mask.any()):
                        continue

                    # Keep NEGATED analysis aligned with the learned explicit route.
                    if role == NEGATED and exp_hat != 1:
                        continue

                    base_pred, base_margin = evaluate_mask(
                        states, gold_mask, pos_head, pmean, pstd,
                        gold_position, device
                    )
                    if base_margin is None:
                        continue

                    nongold_idx = torch.nonzero(
                        ~gold_mask, as_tuple=False
                    ).flatten().tolist()
                    gold_idx = torch.nonzero(
                        gold_mask, as_tuple=False
                    ).flatten().tolist()

                    # Add one false-positive token to the exact gold mask.
                    for token_idx in nongold_idx:
                        perturbed = gold_mask.clone()
                        perturbed[token_idx] = True
                        pred, margin = evaluate_mask(
                            states, perturbed, pos_head, pmean, pstd,
                            gold_position, device
                        )
                        delta = margin - base_margin
                        flip = int(pred != gold_position and base_pred == gold_position)
                        harmful = int(delta < 0)

                        fp_margin_deltas.append(delta)
                        fp_flip_flags.append(flip)
                        harmful_fp_flags.append(harmful)

                        start, end = offsets_all[i][token_idx]
                        token_rows.append({
                            "seed": seed,
                            "fold": fold,
                            "sample_index": i,
                            "role": role_name,
                            "operation": "add_false_positive",
                            "token_index": token_idx,
                            "char_start": start,
                            "char_end": end,
                            "token_text": sample["text"][start:end],
                            "base_margin": base_margin,
                            "new_margin": margin,
                            "margin_delta": delta,
                            "decision_flip": flip,
                            "harmful_or_critical": harmful,
                        })

                    # Remove one true-positive token from the exact gold mask.
                    if len(gold_idx) > 1:
                        for token_idx in gold_idx:
                            perturbed = gold_mask.clone()
                            perturbed[token_idx] = False
                            pred, margin = evaluate_mask(
                                states, perturbed, pos_head, pmean, pstd,
                                gold_position, device
                            )
                            delta = margin - base_margin
                            flip = int(pred != gold_position and base_pred == gold_position)
                            critical = int(delta < 0)

                            tp_margin_deltas.append(delta)
                            tp_flip_flags.append(flip)
                            critical_tp_flags.append(critical)

                            start, end = offsets_all[i][token_idx]
                            token_rows.append({
                                "seed": seed,
                                "fold": fold,
                                "sample_index": i,
                                "role": role_name,
                                "operation": "remove_true_positive",
                                "token_index": token_idx,
                                "char_start": start,
                                "char_end": end,
                                "token_text": sample["text"][start:end],
                                "base_margin": base_margin,
                                "new_margin": margin,
                                "margin_delta": delta,
                                "decision_flip": flip,
                                "harmful_or_critical": critical,
                            })

        fp_mean = statistics.mean(fp_margin_deltas) if fp_margin_deltas else 0.0
        fp_med = statistics.median(fp_margin_deltas) if fp_margin_deltas else 0.0
        tp_mean = statistics.mean(tp_margin_deltas) if tp_margin_deltas else 0.0
        tp_med = statistics.median(tp_margin_deltas) if tp_margin_deltas else 0.0

        per_seed_rows.append({
            "seed": seed,
            "fp_events": len(fp_margin_deltas),
            "fp_mean_margin_delta": fp_mean,
            "fp_median_margin_delta": fp_med,
            "fp_harmful_fraction": (
                statistics.mean(harmful_fp_flags) if harmful_fp_flags else 0.0
            ),
            "fp_flip_fraction": (
                statistics.mean(fp_flip_flags) if fp_flip_flags else 0.0
            ),
            "tp_events": len(tp_margin_deltas),
            "tp_mean_margin_delta": tp_mean,
            "tp_median_margin_delta": tp_med,
            "tp_critical_fraction": (
                statistics.mean(critical_tp_flags) if critical_tp_flags else 0.0
            ),
            "tp_flip_fraction": (
                statistics.mean(tp_flip_flags) if tp_flip_flags else 0.0
            ),
        })

        print(
            f"seed={seed:>3d} "
            f"FP meanΔ={fp_mean:+.4f} "
            f"harmful={per_seed_rows[-1]['fp_harmful_fraction']:.1%} "
            f"flip={per_seed_rows[-1]['fp_flip_fraction']:.1%} | "
            f"TP-remove meanΔ={tp_mean:+.4f} "
            f"critical={per_seed_rows[-1]['tp_critical_fraction']:.1%} "
            f"flip={per_seed_rows[-1]['tp_flip_fraction']:.1%}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "token_impacts.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = [
            "seed","fold","sample_index","role","operation",
            "token_index","char_start","char_end","token_text",
            "base_margin","new_margin","margin_delta",
            "decision_flip","harmful_or_critical",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(token_rows)

    with (out / "per_seed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = list(per_seed_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_seed_rows)

    # Aggregate the most harmful/critical token text patterns.
    groups = {}
    for row in token_rows:
        key = (row["role"], row["operation"], row["token_text"])
        g = groups.setdefault(key, {
            "count": 0,
            "sum_delta": 0.0,
            "flips": 0,
            "harmful": 0,
        })
        g["count"] += 1
        g["sum_delta"] += row["margin_delta"]
        g["flips"] += row["decision_flip"]
        g["harmful"] += row["harmful_or_critical"]

    pattern_rows = []
    for (role, operation, token_text), g in groups.items():
        pattern_rows.append({
            "role": role,
            "operation": operation,
            "token_text": token_text,
            "count": g["count"],
            "mean_margin_delta": g["sum_delta"] / g["count"],
            "decision_flip_rate": g["flips"] / g["count"],
            "harmful_or_critical_rate": g["harmful"] / g["count"],
        })

    pattern_rows.sort(key=lambda r: r["mean_margin_delta"])

    with (out / "token_pattern_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = [
            "role","operation","token_text","count",
            "mean_margin_delta","decision_flip_rate",
            "harmful_or_critical_rate",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pattern_rows)

    fp_mean_all, fp_std_all = mean_std([
        r["fp_mean_margin_delta"] for r in per_seed_rows
    ])
    fp_harm_all, fp_harm_std = mean_std([
        r["fp_harmful_fraction"] for r in per_seed_rows
    ])
    fp_flip_all, fp_flip_std = mean_std([
        r["fp_flip_fraction"] for r in per_seed_rows
    ])
    tp_mean_all, tp_std_all = mean_std([
        r["tp_mean_margin_delta"] for r in per_seed_rows
    ])
    tp_crit_all, tp_crit_std = mean_std([
        r["tp_critical_fraction"] for r in per_seed_rows
    ])
    tp_flip_all, tp_flip_std = mean_std([
        r["tp_flip_fraction"] for r in per_seed_rows
    ])

    direct_scores = [
        eval_direct(Xfull, ysel, args, seed) for seed in seeds
    ]
    oracle_scores = [
        eval_oracle(Xneg, Xsel, yneg, ysel, args, seed) for seed in seeds
    ]
    dm, ds = mean_std(direct_scores)
    om, os = mean_std(oracle_scores)

    print()
    print("Aggregate impact summary")
    print("-" * 96)
    print(f"FP mean margin delta            : {fp_mean_all:+.4f} ± {fp_std_all:.4f}")
    print(f"FP harmful fraction             : {fp_harm_all:.1%} ± {fp_harm_std:.1%}")
    print(f"FP decision-flip fraction       : {fp_flip_all:.1%} ± {fp_flip_std:.1%}")
    print(f"TP-removal mean margin delta    : {tp_mean_all:+.4f} ± {tp_std_all:.4f}")
    print(f"TP critical fraction            : {tp_crit_all:.1%} ± {tp_crit_std:.1%}")
    print(f"TP decision-flip fraction       : {tp_flip_all:.1%} ± {tp_flip_std:.1%}")
    print(f"Direct baseline                 : {dm:.1%} ± {ds:.1%}")
    print(f"Full oracle                     : {om:.1%} ± {os:.1%}")

    print()
    print("Most harmful token patterns")
    print("-" * 96)
    shown = 0
    for row in pattern_rows:
        if row["operation"] != "add_false_positive":
            continue
        print(
            f"{row['role']:8s} token={row['token_text']!r:12s} "
            f"n={row['count']:3d} meanΔ={row['mean_margin_delta']:+.4f} "
            f"flip={row['decision_flip_rate']:.1%}"
        )
        shown += 1
        if shown >= 12:
            break

    print()
    print("Reference")
    print("---------")
    print("v1.4.26 best threshold              : 83.0% ± 1.5%")
    print("v1.4.26 mass-80                     : 83.9% ± 1.7%")
    print("v1.4.26 gold mask                   : 99.9% ± 0.6%")
    print()
    print("Interpretation")
    print("--------------")
    print(
        "Negative margin delta after adding a non-gold token quantifies harmful "
        "false-positive contamination. Negative margin delta after removing a "
        "gold token quantifies true-positive importance."
    )
    print(
        "If only a subset of false positives causes large negative deltas or "
        "decision flips, the next selector should optimize downstream decision "
        "impact rather than token-level precision/recall alone."
    )
    print("Output dir:", out)


if __name__ == "__main__":
    main()
