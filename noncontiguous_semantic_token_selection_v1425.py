# noncontiguous_semantic_token_selection_v1425.py
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

TOP_KS = [1, 2, 3, 4, 5, 6]
MASS = 0.80


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def topk_pool(states, score, k):
    k = min(k, len(states))
    vals, idx = torch.topk(score, k=k)
    X = states.to(score.device)[idx]
    w = vals / vals.sum().clamp_min(1e-6)
    return (w[:, None] * X).sum(0), idx


def gold_mask_pool(states, labels, role, device):
    idx = torch.nonzero(labels == role, as_tuple=False).flatten()
    if len(idx) == 0:
        return None
    return states.to(device)[idx].mean(0)


def eval_seed(states_all, labels_all, args, seed, device):
    names = [f"topk_{k}" for k in TOP_KS] + ["mass80", "gold_mask"]
    preds = {
        name: torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)
        for name in names
    }

    overlap_stats = {k: [] for k in TOP_KS}

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

            gold_sel_idx = set(
                int(x) for x in torch.nonzero(
                    labels == SELECTED, as_tuple=False
                ).flatten().tolist()
            )

            for k in TOP_KS:
                name = f"topk_{k}"
                sel_vec, sel_idx = topk_pool(states, sel_score, k)
                fused = position_probs(
                    sel_vec, pos_head, pmean, pstd
                ).clone()

                if gold_sel_idx:
                    overlap = len(
                        gold_sel_idx.intersection(
                            int(x) for x in sel_idx.tolist()
                        )
                    ) / len(gold_sel_idx)
                    overlap_stats[k].append(overlap)

                if exp_hat == 1:
                    neg_vec, _ = topk_pool(states, neg_score, k)
                    np = position_probs(
                        neg_vec, pos_head, pmean, pstd
                    )
                    fused[0] += np[1]
                    fused[1] += np[0]

                preds[name][i] = int(fused.argmax())

            # Contiguous reference
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

            # Gold-mask upper bound under same contextual block5 representation
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
    overlaps = {
        k: statistics.mean(overlap_stats[k]) if overlap_stats[k] else 0.0
        for k in TOP_KS
    }

    return accs, overlaps


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
        default="results/noncontiguous_semantic_token_selection_v1425",
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
    print(" LLM_GPU v1.4.25 Non-Contiguous Semantic Token Selection / Top-K Sweep")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Selection        : non-contiguous Top-K by learned role score")
    print("Top-K values     :", ", ".join(map(str, TOP_KS)))
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

    setting_names = [f"topk_{k}" for k in TOP_KS] + ["mass80", "gold_mask"]
    per_setting = {name: [] for name in setting_names}
    per_overlap = {k: [] for k in TOP_KS}
    direct_scores, oracle_scores = [], []
    rows = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(Xneg, Xsel, yneg, ysel, args, seed)
        accs, overlaps = eval_seed(
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

        for k in TOP_KS:
            per_overlap[k].append(overlaps[k])
            row[f"topk_{k}_gold_selected_recall"] = overlaps[k]

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
    for k in TOP_KS:
        fields.append(f"topk_{k}_gold_selected_recall")
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

    overlap_rows = []
    for k in TOP_KS:
        m, s = mean_std(per_overlap[k])
        overlap_rows.append((k, m, s, min(per_overlap[k]), max(per_overlap[k])))

    with (out / "overlap_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "top_k","gold_selected_recall_mean","std","min","max"
        ])
        w.writerows(overlap_rows)

    ranked = sorted(
        [(name, *mean_std(per_setting[name])) for name in setting_names],
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best_mean, best_std = ranked[0]

    dm, ds = mean_std(direct_scores)
    om, os = mean_std(oracle_scores)

    print()
    print("Top-K sweep summary")
    print("-" * 96)
    for name, m, s, mn, mx in summary_rows:
        print(
            f"{name:12s}: {m:.1%} ± {s:.1%} "
            f"[{mn:.1%}, {mx:.1%}]"
        )

    print()
    print("Gold SELECTED token recall")
    print("-" * 56)
    for k, m, s, mn, mx in overlap_rows:
        print(
            f"Top-{k}: {m:.1%} ± {s:.1%} "
            f"[{mn:.1%}, {mx:.1%}]"
        )

    print()
    print("Best setting                  :", best_name)
    print(f"Best mean accuracy            : {best_mean:.1%} ± {best_std:.1%}")
    print(f"Mass-80 reference             : {mean_std(per_setting['mass80'])[0]:.1%}")
    print(f"Gold-mask upper bound         : {mean_std(per_setting['gold_mask'])[0]:.1%}")
    print(f"Direct baseline               : {dm:.1%} ± {ds:.1%}")
    print(f"Full oracle                   : {om:.1%} ± {os:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.24 best decision subspace    : 82.9% ± 1.5%")
    print("v1.4.23 simple mass-80 pooling    : 83.9% ± 1.7%")
    print("v1.4.21 oracle SELECTED vector    : 99.9% ± 0.6%")
    print()
    print("Interpretation")
    print("--------------")
    print(
        "Top-K selection removes the contiguous-window assumption and keeps "
        "only the highest-scoring semantic-role tokens."
    )
    print(
        "If Top-K materially exceeds mass-80, semantic role evidence is "
        "distributed non-contiguously. If gold-mask remains far higher, "
        "the remaining bottleneck is learned token selection quality."
    )
    print("Output dir:", out)


if __name__ == "__main__":
    main()
