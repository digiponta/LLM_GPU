# semantic_error_attribution_v1421.py
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

MASS = 0.80

VARIANTS = [
    "learned",
    "oracle_explicitness",
    "oracle_anchor",
    "oracle_selected_vector",
    "oracle_negated_vector",
    "oracle_role_vectors",
    "oracle_role_vectors_explicitness",
]


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def gold_role_indices(labels, role):
    idx = torch.nonzero(labels == role, as_tuple=False).flatten()
    return [int(x) for x in idx.tolist()]


def mass_window_from_anchor(states, score, target_mass, anchor):
    n = len(states)
    total = float(score.sum())
    if total <= 1e-8:
        X = states.to(score.device)
        return X[anchor]

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

    X = states.to(score.device)
    local_x = X[lo:hi + 1]
    local_score = score[lo:hi + 1]
    weights = local_score / local_score.sum().clamp_min(1e-6)
    return (weights[:, None] * local_x).sum(0)


def oracle_anchor_vector(states, labels, role, score):
    gold = gold_role_indices(labels, role)
    if not gold:
        return None
    center = gold[len(gold) // 2]
    return mass_window_from_anchor(states, score, MASS, center)


def run_variant(
    name,
    i,
    states,
    labels,
    sel_score,
    neg_score,
    learned_explicit,
    pos_head,
    pmean,
    pstd,
    device,
):
    sample = CONTRAST_SAMPLES[i]
    gold_explicit = bool((labels == NEGATED).any())

    # SELECTED representation
    if name in {
        "oracle_selected_vector",
        "oracle_role_vectors",
        "oracle_role_vectors_explicitness",
    }:
        sel_vec = pooled_gold_role_vector(states, labels, SELECTED).to(device)
    elif name == "oracle_anchor":
        sel_vec = oracle_anchor_vector(states, labels, SELECTED, sel_score)
        if sel_vec is None:
            sel_vec, _, _ = mass_window(states, sel_score, MASS)
    else:
        sel_vec, _, _ = mass_window(states, sel_score, MASS)

    # Explicitness gate
    if name in {"oracle_explicitness", "oracle_role_vectors_explicitness"}:
        use_explicit = gold_explicit
    else:
        use_explicit = bool(learned_explicit)

    fused = position_probs(sel_vec, pos_head, pmean, pstd).clone()

    if use_explicit:
        if name in {
            "oracle_negated_vector",
            "oracle_role_vectors",
            "oracle_role_vectors_explicitness",
        } and gold_explicit:
            neg_vec = pooled_gold_role_vector(states, labels, NEGATED).to(device)
        elif name == "oracle_anchor" and gold_explicit:
            neg_vec = oracle_anchor_vector(states, labels, NEGATED, neg_score)
            if neg_vec is None:
                neg_vec, _, _ = mass_window(states, neg_score, MASS)
        else:
            neg_vec, _, _ = mass_window(states, neg_score, MASS)

        np = position_probs(neg_vec, pos_head, pmean, pstd)
        fused[0] += np[1]
        fused[1] += np[0]

    return int(fused.argmax())


def eval_seed(states_all, labels_all, args, seed, device):
    preds = {
        name: torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)
        for name in VARIANTS
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

        # Train the same learned position decoder for all ablations.
        # Gold role-pooled vectors are used only on training folds, as in prior versions.
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
            position_vectors,
            position_labels,
            args,
            seed + 30000 + 1000 * fold,
            device,
        )

        for i in test:
            states = states_all[i]
            labels = labels_all[i]
            sel_score = role_scores(states, sel_tagger, device)
            neg_score = role_scores(states, neg_tagger, device)
            exp_hat, _ = predict_explicit(
                states, explicit_head, emean, estd, device
            )

            for name in VARIANTS:
                preds[name][i] = run_variant(
                    name,
                    i,
                    states,
                    labels,
                    sel_score,
                    neg_score,
                    exp_hat,
                    pos_head,
                    pmean,
                    pstd,
                    device,
                )

    gold = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
    )

    return {
        name: float((pred == gold).float().mean())
        for name, pred in preds.items()
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
    ap.add_argument(
        "--output-dir",
        default="results/semantic_error_attribution_v1421",
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
    print(" LLM_GPU v1.4.21 Semantic Error Attribution / Oracle Ablation")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Learned window   : adaptive 80% role-score mass")
    print("Ablations        : explicitness / anchor / selected / negated / role vectors")
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

    all_scores = {name: [] for name in VARIANTS}
    direct_scores, full_oracle_scores = [], []
    rows = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        full_oracle = eval_oracle(Xneg, Xsel, yneg, ysel, args, seed)
        scores = eval_seed(states_all, labels_all, args, seed, device)

        direct_scores.append(direct)
        full_oracle_scores.append(full_oracle)
        for name in VARIANTS:
            all_scores[name].append(scores[name])

        row = {
            "seed": seed,
            "direct": direct,
            "full_oracle": full_oracle,
            **scores,
        }
        rows.append(row)

        print(
            f"seed={seed:>3d} direct={direct:>6.1%} "
            + " ".join(f"{name}={scores[name]:.1%}" for name in VARIANTS)
            + f" full_oracle={full_oracle:.1%}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = ["seed", "direct", *VARIANTS, "full_oracle"]
    with (out / "per_seed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    base_mean, base_std = mean_std(all_scores["learned"])
    summary_rows = []

    dm, ds = mean_std(direct_scores)
    om, os = mean_std(full_oracle_scores)
    summary_rows.append(("direct", dm, ds, min(direct_scores), max(direct_scores), dm - base_mean))

    for name in VARIANTS:
        vals = all_scores[name]
        m, s = mean_std(vals)
        summary_rows.append((name, m, s, min(vals), max(vals), m - base_mean))

    summary_rows.append(("full_oracle", om, os, min(full_oracle_scores), max(full_oracle_scores), om - base_mean))

    with (out / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "variant","mean","std","min","max","delta_vs_learned"
        ])
        w.writerows(summary_rows)

    print()
    print("Oracle ablation summary")
    print("-" * 96)
    for name, m, s, mn, mx, delta in summary_rows:
        print(
            f"{name:34s}: {m:.1%} ± {s:.1%} "
            f"[{mn:.1%}, {mx:.1%}]  "
            f"delta={delta:+.1%}"
        )

    print()
    print("Attribution relative to learned baseline")
    print("----------------------------------------")
    exp_gain = mean_std(all_scores["oracle_explicitness"])[0] - base_mean
    anchor_gain = mean_std(all_scores["oracle_anchor"])[0] - base_mean
    sel_gain = mean_std(all_scores["oracle_selected_vector"])[0] - base_mean
    neg_gain = mean_std(all_scores["oracle_negated_vector"])[0] - base_mean
    both_gain = mean_std(all_scores["oracle_role_vectors"])[0] - base_mean
    both_exp_gain = mean_std(all_scores["oracle_role_vectors_explicitness"])[0] - base_mean
    remaining = om - mean_std(all_scores["oracle_role_vectors_explicitness"])[0]

    print(f"Oracle explicitness gain          : {exp_gain:+.1%}")
    print(f"Oracle anchor gain                : {anchor_gain:+.1%}")
    print(f"Oracle SELECTED vector gain       : {sel_gain:+.1%}")
    print(f"Oracle NEGATED vector gain        : {neg_gain:+.1%}")
    print(f"Oracle both role vectors gain     : {both_gain:+.1%}")
    print(f"Role vectors + explicitness gain  : {both_exp_gain:+.1%}")
    print(f"Remaining gap to full oracle      : {remaining:+.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.20 adaptive mass 80             : 83.9% ± 1.7%")
    print("v1.4.19 fixed radius 7              : 83.9% ± 1.5%")
    print("v1.4.13 rule-based automatic        : 90.4% ± 3.6%")
    print("v1.4.12 full oracle                 : 99.9% ± 0.6%")
    print()
    print("Interpretation")
    print("--------------")
    print(
        "Each ablation replaces only one learned component with gold information. "
        "The largest positive delta identifies the dominant source of the residual error."
    )
    print(
        "The full-oracle result remains the upper bound using gold semantic clauses "
        "re-encoded independently, so any remaining gap after oracle role vectors "
        "indicates representation/decoder mismatch rather than role localization alone."
    )
    print("Output dir:", out)


if __name__ == "__main__":
    main()
