# semantic_token_mask_calibration_v1426.py
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
    train_binary_span_tagger,
    train_explicitness_head,
    predict_explicit,
)
from adaptive_semantic_window_v1420 import (
    role_scores,
    position_probs,
    mass_window,
)

THRESHOLDS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
MASS = 0.80


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def threshold_pool(states, score, threshold):
    mask = score >= threshold
    if not bool(mask.any()):
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask[int(score.argmax())] = True
    X = states.to(score.device)[mask]
    w = score[mask]
    w = w / w.sum().clamp_min(1e-6)
    return (w[:, None] * X).sum(0), mask


def gold_mask_pool(states, labels, role, device):
    mask = labels == role
    if not bool(mask.any()):
        return None
    return states.to(device)[mask.to(device)].mean(0)


def prf(pred_mask, gold_mask):
    pred = pred_mask.bool().cpu()
    gold = gold_mask.bool().cpu()
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    return precision, recall, f1, int(pred.sum()), fp


def eval_seed(states_all, labels_all, args, seed, device):
    names = [f"thr_{int(t*10):02d}" for t in THRESHOLDS] + ["mass80", "gold_mask"]
    preds = {
        name: torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)
        for name in names
    }

    metrics = {
        t: {
            "sel_precision": [],
            "sel_recall": [],
            "sel_f1": [],
            "sel_count": [],
            "sel_fp": [],
            "neg_precision": [],
            "neg_recall": [],
            "neg_f1": [],
            "neg_count": [],
            "neg_fp": [],
        }
        for t in THRESHOLDS
    }

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
            labels = labels_all[i]
            sel_score = role_scores(states, sel_tagger, device)
            neg_score = role_scores(states, neg_tagger, device)
            exp_hat, _ = predict_explicit(
                states, explicit_head, emean, estd, device
            )

            for t in THRESHOLDS:
                name = f"thr_{int(t*10):02d}"
                sel_vec, sel_mask = threshold_pool(states, sel_score, t)
                fused = position_probs(
                    sel_vec, pos_head, pmean, pstd
                ).clone()

                p, r, f1, count, fp = prf(
                    sel_mask, labels == SELECTED
                )
                metrics[t]["sel_precision"].append(p)
                metrics[t]["sel_recall"].append(r)
                metrics[t]["sel_f1"].append(f1)
                metrics[t]["sel_count"].append(count)
                metrics[t]["sel_fp"].append(fp)

                if exp_hat == 1:
                    neg_vec, neg_mask = threshold_pool(states, neg_score, t)
                    np = position_probs(
                        neg_vec, pos_head, pmean, pstd
                    )
                    fused[0] += np[1]
                    fused[1] += np[0]

                    gold_neg = labels == NEGATED
                    if bool(gold_neg.any()):
                        p, r, f1, count, fp = prf(neg_mask, gold_neg)
                        metrics[t]["neg_precision"].append(p)
                        metrics[t]["neg_recall"].append(r)
                        metrics[t]["neg_f1"].append(f1)
                        metrics[t]["neg_count"].append(count)
                        metrics[t]["neg_fp"].append(fp)

                preds[name][i] = int(fused.argmax())

            # mass-80 reference
            sel_mass, _, _ = mass_window(states, sel_score, MASS)
            fused_mass = position_probs(
                sel_mass, pos_head, pmean, pstd
            ).clone()
            if exp_hat == 1:
                neg_mass, _, _ = mass_window(states, neg_score, MASS)
                np = position_probs(
                    neg_mass, pos_head, pmean, pstd
                )
                fused_mass[0] += np[1]
                fused_mass[1] += np[0]
            preds["mass80"][i] = int(fused_mass.argmax())

            # gold mask upper bound
            sel_gold = gold_mask_pool(states, labels, SELECTED, device)
            fused_gold = position_probs(
                sel_gold, pos_head, pmean, pstd
            ).clone()
            if exp_hat == 1:
                neg_gold = gold_mask_pool(states, labels, NEGATED, device)
                if neg_gold is not None:
                    np = position_probs(
                        neg_gold, pos_head, pmean, pstd
                    )
                    fused_gold[0] += np[1]
                    fused_gold[1] += np[0]
            preds["gold_mask"][i] = int(fused_gold.argmax())

    gold = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
    )

    accs = {
        name: float((pred == gold).float().mean())
        for name, pred in preds.items()
    }

    agg = {}
    for t in THRESHOLDS:
        agg[t] = {}
        for key, vals in metrics[t].items():
            agg[t][key] = statistics.mean(vals) if vals else 0.0

    return accs, agg


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
    ap.add_argument(
        "--output-dir",
        default="results/semantic_token_mask_calibration_v1426",
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
    print(" LLM_GPU v1.4.26 Semantic Token Mask Calibration / Precision-Recall Sweep")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Masking          : thresholded learned role score")
    print("Thresholds       :", ", ".join(f"{t:.1f}" for t in THRESHOLDS))
    print("References       : mass-80 contiguous + gold role mask")
    print("Evaluation       : 5-fold unseen-paraphrase holdout")
    print("Relation seeds   :", f"{seeds[0]}..{seeds[-1]}")
    print("Base model       : completely frozen")
    print()

    print("Encoding block5 token states...")
    states_all, labels_all = [], []
    for sample in CONTRAST_SAMPLES:
        states, offsets = encode_token_states(model, tok, sample["text"])
        labels = token_labels(sample, offsets)
        states_all.append(states)
        labels_all.append(labels)

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

    setting_names = [f"thr_{int(t*10):02d}" for t in THRESHOLDS] + ["mass80", "gold_mask"]
    per_setting = {name: [] for name in setting_names}
    per_metric = {
        t: {
            key: []
            for key in [
                "sel_precision","sel_recall","sel_f1","sel_count","sel_fp",
                "neg_precision","neg_recall","neg_f1","neg_count","neg_fp",
            ]
        }
        for t in THRESHOLDS
    }
    direct_scores, oracle_scores = [], []
    rows = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(Xneg, Xsel, yneg, ysel, args, seed)
        accs, agg = eval_seed(
            states_all, labels_all, args, seed, device
        )

        direct_scores.append(direct)
        oracle_scores.append(oracle)

        row = {
            "seed": seed,
            "direct_accuracy": direct,
            "oracle_accuracy": oracle,
        }

        text_parts = []
        for name in setting_names:
            per_setting[name].append(accs[name])
            row[f"{name}_accuracy"] = accs[name]
            text_parts.append(f"{name}={accs[name]:.1%}")

        for t in THRESHOLDS:
            for key, value in agg[t].items():
                per_metric[t][key].append(value)
                row[f"thr_{int(t*10):02d}_{key}"] = value

        rows.append(row)

        print(
            f"seed={seed:>3d} direct={direct:>6.1%} "
            + " ".join(text_parts)
            + f" oracle={oracle:>6.1%}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = ["seed", "direct_accuracy"]
    for name in setting_names:
        fields.append(f"{name}_accuracy")
    for t in THRESHOLDS:
        for key in per_metric[t]:
            fields.append(f"thr_{int(t*10):02d}_{key}")
    fields.append("oracle_accuracy")

    with (out / "per_seed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary_rows = []
    for name in setting_names:
        m, s = mean_std(per_setting[name])
        summary_rows.append(
            (name, m, s, min(per_setting[name]), max(per_setting[name]))
        )

    with (out / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "setting","accuracy_mean","accuracy_std",
            "accuracy_min","accuracy_max"
        ])
        w.writerows(summary_rows)

    pr_rows = []
    for t in THRESHOLDS:
        means = {
            key: mean_std(vals)[0]
            for key, vals in per_metric[t].items()
        }
        pr_rows.append((
            t,
            means["sel_precision"], means["sel_recall"], means["sel_f1"],
            means["sel_count"], means["sel_fp"],
            means["neg_precision"], means["neg_recall"], means["neg_f1"],
            means["neg_count"], means["neg_fp"],
            mean_std(per_setting[f"thr_{int(t*10):02d}"])[0],
        ))

    with (out / "precision_recall_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "threshold",
            "selected_precision","selected_recall","selected_f1",
            "selected_count","selected_false_positive_count",
            "negated_precision","negated_recall","negated_f1",
            "negated_count","negated_false_positive_count",
            "routing_accuracy",
        ])
        w.writerows(pr_rows)

    best_threshold = max(
        THRESHOLDS,
        key=lambda t: mean_std(
            per_setting[f"thr_{int(t*10):02d}"]
        )[0],
    )
    best_name = f"thr_{int(best_threshold*10):02d}"
    best_mean, best_std = mean_std(per_setting[best_name])

    dm, ds = mean_std(direct_scores)
    om, os = mean_std(oracle_scores)
    mass_m, mass_s = mean_std(per_setting["mass80"])
    gold_m, gold_s = mean_std(per_setting["gold_mask"])

    print()
    print("Threshold sweep summary")
    print("-" * 118)
    for t, sp, sr, sf1, sc, sfp, np, nr, nf1, nc, nfp, acc in pr_rows:
        print(
            f"thr={t:.1f} acc={acc:.1%}  "
            f"SEL P/R/F1={sp:.1%}/{sr:.1%}/{sf1:.1%} "
            f"count={sc:.2f} FP={sfp:.2f}  "
            f"NEG P/R/F1={np:.1%}/{nr:.1%}/{nf1:.1%} "
            f"count={nc:.2f} FP={nfp:.2f}"
        )

    print()
    print("Best threshold                 :", f"{best_threshold:.1f}")
    print(f"Best mean accuracy             : {best_mean:.1%} ± {best_std:.1%}")
    print(f"Mass-80 reference              : {mass_m:.1%} ± {mass_s:.1%}")
    print(f"Gold-mask upper bound          : {gold_m:.1%} ± {gold_s:.1%}")
    print(f"Direct baseline                : {dm:.1%} ± {ds:.1%}")
    print(f"Full oracle                    : {om:.1%} ± {os:.1%}")
    print(f"Best gain vs mass-80           : {best_mean-mass_m:+.1%}")
    print(f"Best gap vs gold mask          : {best_mean-gold_m:+.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.25 best Top-K                 : 82.1% ± 1.2%")
    print("v1.4.25 mass-80                    : 83.9% ± 1.7%")
    print("v1.4.25 gold mask                  : 99.9% ± 0.6%")
    print()
    print("Interpretation")
    print("--------------")
    print(
        "This sweep calibrates the binary semantic token mask and measures "
        "precision, recall, F1, selected-token count, false positives, and "
        "routing accuracy at the same operating points."
    )
    print(
        "If routing peaks at a threshold with higher precision and fewer "
        "false positives than Top-K, false-positive token contamination is "
        "confirmed as a major remaining bottleneck."
    )
    print("Output dir:", out)


if __name__ == "__main__":
    main()
