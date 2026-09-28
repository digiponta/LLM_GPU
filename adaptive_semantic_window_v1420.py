# adaptive_semantic_window_v1420.py
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

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

MASS_TARGETS = [0.70, 0.80, 0.90]
LENGTH_RATIOS = [0.25, 0.40]
FIXED_RADIUS = 7


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


@torch.no_grad()
def role_scores(states, tagger, device):
    return torch.sigmoid(tagger(states.to(device)))


@torch.no_grad()
def position_probs(vec, head, mean, std):
    return torch.softmax(head((vec[None, :] - mean) / std), dim=-1)[0]


def pool_range(states, score, lo, hi):
    X = states.to(score.device)
    local_x = X[lo:hi]
    local_score = score[lo:hi]
    w = local_score / local_score.sum().clamp_min(1e-6)
    return (w[:, None] * local_x).sum(0)


def fixed_window(states, score, radius):
    anchor = int(score.argmax())
    lo = max(0, anchor - radius)
    hi = min(len(states), anchor + radius + 1)
    return pool_range(states, score, lo, hi), lo, hi


def length_ratio_window(states, score, ratio):
    n = len(states)
    anchor = int(score.argmax())
    width = max(1, int(math.ceil(n * ratio)))
    radius = max(0, (width - 1) // 2)
    lo = max(0, anchor - radius)
    hi = min(n, anchor + radius + 1)
    while hi - lo < width:
        if lo > 0:
            lo -= 1
        elif hi < n:
            hi += 1
        else:
            break
    return pool_range(states, score, lo, hi), lo, hi


def mass_window(states, score, target_mass):
    n = len(states)
    anchor = int(score.argmax())
    total = float(score.sum())
    if total <= 1e-8:
        return fixed_window(states, score, 0)

    lo = hi = anchor
    current = float(score[anchor])

    while current / total < target_mass and (lo > 0 or hi < n - 1):
        left_score = float(score[lo - 1]) if lo > 0 else -1.0
        right_score = float(score[hi + 1]) if hi < n - 1 else -1.0

        if right_score > left_score:
            hi += 1
            current += float(score[hi])
        else:
            lo -= 1
            current += float(score[lo])

    return pool_range(states, score, lo, hi + 1), lo, hi + 1


def global_pool(states, score):
    return pool_range(states, score, 0, len(states)), 0, len(states)


def gold_anchor_indices(labels, role):
    idx = torch.nonzero(labels == role, as_tuple=False).flatten()
    return set(int(x) for x in idx.tolist())


def eval_seed(states_all, labels_all, args, seed, device):
    setting_names = (
        [f"mass_{int(m*100)}" for m in MASS_TARGETS]
        + [f"ratio_{int(r*100)}" for r in LENGTH_RATIOS]
        + [f"radius_{FIXED_RADIUS}", "global"]
    )

    preds = {
        name: torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)
        for name in setting_names
    }

    widths = {name: [] for name in setting_names}
    selected_anchor_hits = []
    negated_anchor_hits = []

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
            sel_score = role_scores(states, sel_tagger, device)
            neg_score = role_scores(states, neg_tagger, device)

            sel_anchor = int(sel_score.argmax())
            neg_anchor = int(neg_score.argmax())
            selected_anchor_hits.append(
                sel_anchor in gold_anchor_indices(labels_all[i], SELECTED)
            )
            gold_neg = gold_anchor_indices(labels_all[i], NEGATED)
            if gold_neg:
                negated_anchor_hits.append(neg_anchor in gold_neg)

            exp_hat, _ = predict_explicit(
                states, explicit_head, emean, estd, device
            )

            # Mass windows
            for mass in MASS_TARGETS:
                name = f"mass_{int(mass*100)}"
                sel_vec, slo, shi = mass_window(states, sel_score, mass)
                fused = position_probs(sel_vec, pos_head, pmean, pstd).clone()
                widths[name].append(shi - slo)

                if exp_hat == 1:
                    neg_vec, nlo, nhi = mass_window(states, neg_score, mass)
                    np = position_probs(neg_vec, pos_head, pmean, pstd)
                    fused[0] += args.negated_weight * np[1]
                    fused[1] += args.negated_weight * np[0]
                    widths[name].append(nhi - nlo)

                preds[name][i] = int(fused.argmax())

            # Length-ratio windows
            for ratio in LENGTH_RATIOS:
                name = f"ratio_{int(ratio*100)}"
                sel_vec, slo, shi = length_ratio_window(states, sel_score, ratio)
                fused = position_probs(sel_vec, pos_head, pmean, pstd).clone()
                widths[name].append(shi - slo)

                if exp_hat == 1:
                    neg_vec, nlo, nhi = length_ratio_window(states, neg_score, ratio)
                    np = position_probs(neg_vec, pos_head, pmean, pstd)
                    fused[0] += args.negated_weight * np[1]
                    fused[1] += args.negated_weight * np[0]
                    widths[name].append(nhi - nlo)

                preds[name][i] = int(fused.argmax())

            # Fixed radius reference
            name = f"radius_{FIXED_RADIUS}"
            sel_vec, slo, shi = fixed_window(states, sel_score, FIXED_RADIUS)
            fused = position_probs(sel_vec, pos_head, pmean, pstd).clone()
            widths[name].append(shi - slo)

            if exp_hat == 1:
                neg_vec, nlo, nhi = fixed_window(states, neg_score, FIXED_RADIUS)
                np = position_probs(neg_vec, pos_head, pmean, pstd)
                fused[0] += args.negated_weight * np[1]
                fused[1] += args.negated_weight * np[0]
                widths[name].append(nhi - nlo)

            preds[name][i] = int(fused.argmax())

            # Global
            name = "global"
            sel_vec, slo, shi = global_pool(states, sel_score)
            fused = position_probs(sel_vec, pos_head, pmean, pstd).clone()
            widths[name].append(shi - slo)

            if exp_hat == 1:
                neg_vec, nlo, nhi = global_pool(states, neg_score)
                np = position_probs(neg_vec, pos_head, pmean, pstd)
                fused[0] += args.negated_weight * np[1]
                fused[1] += args.negated_weight * np[0]
                widths[name].append(nhi - nlo)

            preds[name][i] = int(fused.argmax())

    gold = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
    )

    accs = {
        name: float((pred == gold).float().mean())
        for name, pred in preds.items()
    }
    mean_widths = {
        name: statistics.mean(vals) if vals else 0.0
        for name, vals in widths.items()
    }

    return (
        accs,
        mean_widths,
        sum(selected_anchor_hits) / len(selected_anchor_hits),
        sum(negated_anchor_hits) / len(negated_anchor_hits)
        if negated_anchor_hits else 0.0,
    )


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
    ap.add_argument("--negated-weight", type=float, default=1.0)
    ap.add_argument(
        "--output-dir",
        default="results/adaptive_semantic_window_v1420",
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
    print(" LLM_GPU v1.4.20 Adaptive Semantic Window")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Adaptive mass    : 70%, 80%, 90% role-score mass")
    print("Length ratios    : 25%, 40% of phrase tokens")
    print("Reference        : fixed radius 7 + global")
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

    setting_names = (
        [f"mass_{int(m*100)}" for m in MASS_TARGETS]
        + [f"ratio_{int(r*100)}" for r in LENGTH_RATIOS]
        + [f"radius_{FIXED_RADIUS}", "global"]
    )

    per_setting = {name: [] for name in setting_names}
    per_width = {name: [] for name in setting_names}
    direct_scores, oracle_scores = [], []
    sel_anchor_scores, neg_anchor_scores = [], []
    rows = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(Xneg, Xsel, yneg, ysel, args, seed)
        accs, widths, sel_anchor, neg_anchor = eval_seed(
            states_all, labels_all, args, seed, device
        )

        direct_scores.append(direct)
        oracle_scores.append(oracle)
        sel_anchor_scores.append(sel_anchor)
        neg_anchor_scores.append(neg_anchor)

        for name in setting_names:
            per_setting[name].append(accs[name])
            per_width[name].append(widths[name])

        best_name = max(setting_names, key=lambda n: accs[n])

        row = {
            "seed": seed,
            "direct_accuracy": direct,
            "oracle_accuracy": oracle,
            "selected_anchor_accuracy": sel_anchor,
            "negated_anchor_accuracy": neg_anchor,
            "best_setting": best_name,
            "best_accuracy": accs[best_name],
        }
        for name in setting_names:
            row[f"{name}_accuracy"] = accs[name]
            row[f"{name}_mean_width"] = widths[name]
        rows.append(row)

        acc_text = " ".join(
            f"{name}={accs[name]:.1%}" for name in setting_names
        )
        print(
            f"seed={seed:>3d} direct={direct:>6.1%} "
            f"oracle={oracle:>6.1%} {acc_text} best={best_name}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed","direct_accuracy","oracle_accuracy",
        "selected_anchor_accuracy","negated_anchor_accuracy",
        *[f"{name}_accuracy" for name in setting_names],
        *[f"{name}_mean_width" for name in setting_names],
        "best_setting","best_accuracy",
    ]
    with (out / "per_seed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary_rows = []
    for name in setting_names:
        am, ast = mean_std(per_setting[name])
        wm, wst = mean_std(per_width[name])
        summary_rows.append(
            (
                name,
                am, ast,
                min(per_setting[name]), max(per_setting[name]),
                wm, wst,
            )
        )

    with (out / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "setting","accuracy_mean","accuracy_std",
            "accuracy_min","accuracy_max",
            "mean_width","width_std",
        ])
        w.writerows(summary_rows)

    ranked = sorted(
        [(name, am, ast, wm) for name, am, ast, _, _, wm, _ in summary_rows],
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best_mean, best_std, best_width = ranked[0]

    sm, ss = mean_std(sel_anchor_scores)
    nm, ns = mean_std(neg_anchor_scores)
    dm, ds = mean_std(direct_scores)
    om, os = mean_std(oracle_scores)

    print()
    print("Adaptive window summary")
    print("-" * 88)
    for name, am, ast, amin, amax, wm, wst in summary_rows:
        print(
            f"{name:12s}: {am:.1%} ± {ast:.1%} "
            f"[{amin:.1%}, {amax:.1%}]  "
            f"width={wm:.2f} ± {wst:.2f}"
        )

    fixed_vals = per_setting[f"radius_{FIXED_RADIUS}"]
    fm, fs = mean_std(fixed_vals)

    print()
    print("Best setting                     :", best_name)
    print(f"Best mean accuracy               : {best_mean:.1%} ± {best_std:.1%}")
    print(f"Best mean window width           : {best_width:.2f} tokens")
    print(f"Fixed radius 7                   : {fm:.1%} ± {fs:.1%}")
    print(f"SELECTED anchor accuracy         : {sm:.1%} ± {ss:.1%}")
    print(f"NEGATED anchor accuracy          : {nm:.1%} ± {ns:.1%}")
    print(f"Direct baseline                  : {dm:.1%} ± {ds:.1%}")
    print(f"Oracle                           : {om:.1%} ± {os:.1%}")
    print(f"Best gain vs fixed radius 7      : {best_mean-fm:+.1%}")
    print(f"Best gap vs oracle               : {best_mean-om:+.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.19 best fixed radius 7         : 83.9% ± 1.5%")
    print("v1.4.19 global pooling              : 81.4% ± 1.5%")
    print("v1.4.13 rule-based automatic        : 90.4% ± 3.6%")
    print("v1.4.12 oracle                      : 99.9% ± 0.6%")
    print()
    print("Interpretation")
    print("--------------")
    print(
        "Adaptive windows test whether semantic context size should depend "
        "on role-score mass or phrase length rather than a fixed radius."
    )
    print(
        "An adaptive setting that beats radius 7 with similar or lower "
        "variance would indicate that the semantic field scale varies by phrase."
    )
    print("Output dir:", out)


if __name__ == "__main__":
    main()
