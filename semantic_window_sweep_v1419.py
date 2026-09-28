# semantic_window_sweep_v1419.py
from __future__ import annotations

import argparse
import csv
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

RADII = [0, 1, 2, 3, 4, 5, 7]


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


@torch.no_grad()
def role_scores(states, tagger, device):
    return torch.sigmoid(tagger(states.to(device)))


@torch.no_grad()
def pooled_from_anchor(states, score, radius):
    X = states.to(score.device)
    anchor = int(score.argmax())
    lo = max(0, anchor - radius)
    hi = min(len(X), anchor + radius + 1)
    local_score = score[lo:hi]
    weights = local_score / local_score.sum().clamp_min(1e-6)
    return (weights[:, None] * X[lo:hi]).sum(0), anchor


@torch.no_grad()
def pooled_global(states, score):
    X = states.to(score.device)
    weights = score / score.sum().clamp_min(1e-6)
    return (weights[:, None] * X).sum(0)


@torch.no_grad()
def position_probs(vec, head, mean, std):
    return torch.softmax(head((vec[None, :] - mean) / std), dim=-1)[0]


def eval_seed(states_all, labels_all, args, seed, device):
    preds = {r: torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long) for r in RADII}
    preds["global"] = torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)

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

            gold_sel_idx = set(torch.nonzero(labels_all[i] == SELECTED, as_tuple=False).flatten().tolist())
            gold_neg_idx = set(torch.nonzero(labels_all[i] == NEGATED, as_tuple=False).flatten().tolist())
            selected_anchor_hits.append(sel_anchor in gold_sel_idx)
            if gold_neg_idx:
                negated_anchor_hits.append(neg_anchor in gold_neg_idx)

            exp_hat, _ = predict_explicit(states, explicit_head, emean, estd, device)

            # Global route
            sel_global = pooled_global(states, sel_score)
            fused_global = position_probs(sel_global, pos_head, pmean, pstd)
            if exp_hat == 1:
                neg_global = pooled_global(states, neg_score)
                np = position_probs(neg_global, pos_head, pmean, pstd)
                fused_global = fused_global.clone()
                fused_global[0] += args.negated_weight * np[1]
                fused_global[1] += args.negated_weight * np[0]
            preds["global"][i] = int(fused_global.argmax())

            # Radius sweep
            for radius in RADII:
                sel_vec, _ = pooled_from_anchor(states, sel_score, radius)
                fused = position_probs(sel_vec, pos_head, pmean, pstd)
                if exp_hat == 1:
                    neg_vec, _ = pooled_from_anchor(states, neg_score, radius)
                    np = position_probs(neg_vec, pos_head, pmean, pstd)
                    fused = fused.clone()
                    fused[0] += args.negated_weight * np[1]
                    fused[1] += args.negated_weight * np[0]
                preds[radius][i] = int(fused.argmax())

    gold = torch.tensor([s["selected_position"] for s in CONTRAST_SAMPLES], dtype=torch.long)
    accs = {k: float((v == gold).float().mean()) for k, v in preds.items()}
    return (
        accs,
        sum(selected_anchor_hits) / len(selected_anchor_hits),
        sum(negated_anchor_hits) / len(negated_anchor_hits) if negated_anchor_hits else 0.0,
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
    ap.add_argument("--output-dir", default="results/semantic_window_sweep_v1419")
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
    print(" LLM_GPU v1.4.19 Semantic Window Sweep")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Window radii     :", ", ".join(map(str, RADII)), "+ global")
    print("Pooling          : anchor-centered role-weighted local pooling")
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
    Xfull = torch.stack([encode_cached(model, tok, cache, s["text"]) for s in CONTRAST_SAMPLES]).to(device)
    Xneg = torch.stack([encode_cached(model, tok, cache, s["negated_text"]) for s in CONTRAST_SAMPLES]).to(device)
    Xsel = torch.stack([encode_cached(model, tok, cache, s["selected_text"]) for s in CONTRAST_SAMPLES]).to(device)
    ysel = torch.tensor([s["selected_position"] for s in CONTRAST_SAMPLES], dtype=torch.long, device=device)
    yneg = torch.tensor([s["negated_position"] for s in CONTRAST_SAMPLES], dtype=torch.long, device=device)

    per_setting = {r: [] for r in RADII}
    per_setting["global"] = []
    direct_scores, oracle_scores = [], []
    sel_anchor_scores, neg_anchor_scores = [], []
    rows = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(Xneg, Xsel, yneg, ysel, args, seed)
        accs, sel_anchor, neg_anchor = eval_seed(states_all, labels_all, args, seed, device)

        direct_scores.append(direct)
        oracle_scores.append(oracle)
        sel_anchor_scores.append(sel_anchor)
        neg_anchor_scores.append(neg_anchor)
        for key, val in accs.items():
            per_setting[key].append(val)

        best_radius = max(RADII, key=lambda r: accs[r])

        row = {
            "seed": seed,
            "direct_accuracy": direct,
            "oracle_accuracy": oracle,
            "selected_anchor_accuracy": sel_anchor,
            "negated_anchor_accuracy": neg_anchor,
            "global_accuracy": accs["global"],
            "best_radius": best_radius,
            "best_radius_accuracy": accs[best_radius],
        }
        for r in RADII:
            row[f"radius_{r}"] = accs[r]
        rows.append(row)

        radius_text = " ".join(f"r{r}={accs[r]:.1%}" for r in RADII)
        print(
            f"seed={seed:>3d} direct={direct:>6.1%} oracle={oracle:>6.1%} "
            f"{radius_text} global={accs['global']:.1%} best=r{best_radius}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed","direct_accuracy","oracle_accuracy",
        *[f"radius_{r}" for r in RADII],
        "global_accuracy",
        "selected_anchor_accuracy","negated_anchor_accuracy",
        "best_radius","best_radius_accuracy",
    ]
    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary_rows = []
    for r in RADII:
        m, s = mean_std(per_setting[r])
        summary_rows.append((f"radius_{r}", m, s, min(per_setting[r]), max(per_setting[r])))
    gm, gs = mean_std(per_setting["global"])
    summary_rows.append(("global", gm, gs, min(per_setting["global"]), max(per_setting["global"])))

    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["setting","mean","std","min","max"])
        w.writerows(summary_rows)

    ranked = sorted(
        [(name, m, s) for name, m, s, _, _ in summary_rows],
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best_mean, best_std = ranked[0]

    print()
    print("Window sweep summary")
    print("-" * 88)
    for name, m, s, mn, mx in summary_rows:
        print(f"{name:12s}: {m:.1%} ± {s:.1%} [{mn:.1%}, {mx:.1%}]")

    sm, ss = mean_std(sel_anchor_scores)
    nm, ns = mean_std(neg_anchor_scores)
    dm, ds = mean_std(direct_scores)
    om, os = mean_std(oracle_scores)

    print()
    print(f"Best setting                     : {best_name}")
    print(f"Best mean accuracy               : {best_mean:.1%} ± {best_std:.1%}")
    print(f"SELECTED anchor accuracy         : {sm:.1%} ± {ss:.1%}")
    print(f"NEGATED anchor accuracy          : {nm:.1%} ± {ns:.1%}")
    print(f"Direct baseline                  : {dm:.1%} ± {ds:.1%}")
    print(f"Oracle                           : {om:.1%} ± {os:.1%}")
    print(f"Best gain vs global              : {best_mean-gm:+.1%}")
    print(f"Best gap vs oracle               : {best_mean-om:+.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.18 radius 3 anchor pooling     : 83.0% ± 2.4%")
    print("v1.4.18 global pooling              : 81.4% ± 1.5%")
    print("v1.4.13 rule-based automatic        : 90.4% ± 3.6%")
    print("v1.4.12 oracle                      : 99.9% ± 0.6%")
    print()
    print("Interpretation")
    print("--------------")
    print("The sweep estimates the local semantic scale around a learned role anchor.")
    print("A peak at a finite radius would support a local distributed semantic field;")
    print("a monotonic rise toward global would indicate broadly distributed context.")
    print("Output dir:", out)


if __name__ == "__main__":
    main()
