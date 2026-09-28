# semantic_anchor_pooling_v1418.py
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

def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


@torch.no_grad()
def anchor_pool(states, tagger, radius, device):
    X = states.to(device)
    score = torch.sigmoid(tagger(X))

    anchor = int(score.argmax())
    lo = max(0, anchor - radius)
    hi = min(len(X), anchor + radius + 1)

    local_x = X[lo:hi]
    local_score = score[lo:hi]
    weights = local_score / local_score.sum().clamp_min(1e-6)
    vec = (weights[:, None] * local_x).sum(0)

    return vec, anchor, float(score[anchor]), lo, hi


@torch.no_grad()
def global_pool(states, tagger, device):
    X = states.to(device)
    score = torch.sigmoid(tagger(X))
    weights = score / score.sum().clamp_min(1e-6)
    vec = (weights[:, None] * X).sum(0)
    return vec


@torch.no_grad()
def position_probs(vec, head, mean, std):
    x = (vec[None, :] - mean) / std
    return torch.softmax(head(x), dim=-1)[0]


def gold_anchor_indices(labels, role):
    idx = torch.nonzero(labels == role, as_tuple=False).flatten()
    if len(idx) == 0:
        return set()
    return set(int(x) for x in idx.tolist())


def eval_anchor(
    states_all, labels_all,
    args, seed, device
):
    pred_anchor = torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)
    pred_global = torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)
    explicit_pred = torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)

    selected_anchor_hits = []
    negated_anchor_hits = []
    selected_anchor_conf = []
    negated_anchor_conf = []

    for fold, test in enumerate(build_folds()):
        test_set = set(test)
        train = [
            i for i in range(len(CONTRAST_SAMPLES))
            if i not in test_set
        ]

        sel_tagger = train_binary_span_tagger(
            [states_all[i] for i in train],
            [labels_all[i] for i in train],
            SELECTED,
            args,
            seed + 1000 * fold,
            device,
        )

        neg_tagger = train_binary_span_tagger(
            [states_all[i] for i in train],
            [labels_all[i] for i in train],
            NEGATED,
            args,
            seed + 10000 + 1000 * fold,
            device,
        )

        explicit_head, emean, estd = train_explicitness_head(
            [states_all[i] for i in train],
            [labels_all[i] for i in train],
            args,
            seed + 20000 + 1000 * fold,
            device,
        )

        position_vectors = []
        position_labels = []

        for i in train:
            sel = pooled_gold_role_vector(
                states_all[i], labels_all[i], SELECTED
            )
            neg = pooled_gold_role_vector(
                states_all[i], labels_all[i], NEGATED
            )

            if sel is not None:
                position_vectors.append(sel)
                position_labels.append(
                    CONTRAST_SAMPLES[i]["selected_position"]
                )

            if neg is not None:
                position_vectors.append(neg)
                position_labels.append(
                    CONTRAST_SAMPLES[i]["negated_position"]
                )

        pos_head, pmean, pstd = train_position_head(
            position_vectors,
            position_labels,
            args,
            seed + 30000 + 1000 * fold,
            device,
        )

        for i in test:
            states = states_all[i]

            sel_vec, sel_anchor, sel_conf, _, _ = anchor_pool(
                states, sel_tagger, args.window_radius, device
            )
            sel_global = global_pool(
                states, sel_tagger, device
            )

            sel_pos_anchor = position_probs(
                sel_vec, pos_head, pmean, pstd
            )
            sel_pos_global = position_probs(
                sel_global, pos_head, pmean, pstd
            )

            fused_anchor = sel_pos_anchor.clone()
            fused_global = sel_pos_global.clone()

            gold_sel = gold_anchor_indices(
                labels_all[i], SELECTED
            )
            selected_anchor_hits.append(
                sel_anchor in gold_sel
            )
            selected_anchor_conf.append(sel_conf)

            exp_hat, _ = predict_explicit(
                states, explicit_head, emean, estd, device
            )
            explicit_pred[i] = exp_hat

            if exp_hat == 1:
                neg_vec, neg_anchor, neg_conf, _, _ = anchor_pool(
                    states, neg_tagger,
                    args.window_radius, device
                )
                neg_global = global_pool(
                    states, neg_tagger, device
                )

                neg_pos_anchor = position_probs(
                    neg_vec, pos_head, pmean, pstd
                )
                neg_pos_global = position_probs(
                    neg_global, pos_head, pmean, pstd
                )

                fused_anchor[0] += (
                    args.negated_weight * neg_pos_anchor[1]
                )
                fused_anchor[1] += (
                    args.negated_weight * neg_pos_anchor[0]
                )

                fused_global[0] += (
                    args.negated_weight * neg_pos_global[1]
                )
                fused_global[1] += (
                    args.negated_weight * neg_pos_global[0]
                )

                gold_neg = gold_anchor_indices(
                    labels_all[i], NEGATED
                )
                if gold_neg:
                    negated_anchor_hits.append(
                        neg_anchor in gold_neg
                    )
                    negated_anchor_conf.append(neg_conf)

            pred_anchor[i] = int(fused_anchor.argmax())
            pred_global[i] = int(fused_global.argmax())

    gold = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
    )

    explicit_gold = torch.tensor([
        int(bool((lab == NEGATED).any()))
        for lab in labels_all
    ], dtype=torch.long)

    return {
        "anchor_accuracy":
            float((pred_anchor == gold).float().mean()),
        "global_accuracy":
            float((pred_global == gold).float().mean()),
        "explicitness_accuracy":
            float((explicit_pred == explicit_gold).float().mean()),
        "selected_anchor_accuracy":
            sum(selected_anchor_hits) / len(selected_anchor_hits),
        "negated_anchor_accuracy":
            (
                sum(negated_anchor_hits) / len(negated_anchor_hits)
                if negated_anchor_hits else 0.0
            ),
        "selected_anchor_confidence":
            statistics.mean(selected_anchor_conf),
        "negated_anchor_confidence":
            (
                statistics.mean(negated_anchor_conf)
                if negated_anchor_conf else 0.0
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
    ap.add_argument("--window-radius", type=int, default=3)
    ap.add_argument("--negated-weight", type=float, default=1.0)
    ap.add_argument(
        "--output-dir",
        default="results/semantic_anchor_pooling_v1418",
    )
    args = ap.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [
            int(x) for x in args.seeds.split(",")
            if x.strip()
        ]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(
        args.model, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("=" * 118)
    print(" LLM_GPU v1.4.18 Semantic Anchor Pooling")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Anchor detector  : learned SELECTED / NEGATED token score")
    print("Pooling          :", f"anchor ± {args.window_radius} tokens, role-weighted")
    print("Explicit route   : SELECTED + NEGATED -> binder")
    print("Implicit route   : SELECTED only; NEGATED inferred as opposite")
    print("Evaluation       : 5-fold unseen-paraphrase holdout")
    print("Relation seeds   :", f"{seeds[0]}..{seeds[-1]}")
    print("Base model       : completely frozen")
    print()

    print("Encoding block5 token states...")
    states_all = []
    labels_all = []

    for sample in CONTRAST_SAMPLES:
        states, offsets = encode_token_states(
            model, tok, sample["text"]
        )
        labels = token_labels(sample, offsets)
        states_all.append(states)
        labels_all.append(labels)

    explicit_count = sum(
        bool((lab == NEGATED).any())
        for lab in labels_all
    )
    print(
        f"Gold decomposition: explicit={explicit_count}, "
        f"implicit={len(labels_all)-explicit_count}"
    )

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
        dtype=torch.long,
        device=device,
    )

    yneg = torch.tensor(
        [s["negated_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
        device=device,
    )

    rows = []
    direct_scores = []
    oracle_scores = []
    global_scores = []
    anchor_scores = []
    explicit_scores = []
    selected_anchor_scores = []
    negated_anchor_scores = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(
            Xneg, Xsel, yneg, ysel, args, seed
        )

        result = eval_anchor(
            states_all, labels_all,
            args, seed, device
        )

        direct_scores.append(direct)
        oracle_scores.append(oracle)
        global_scores.append(result["global_accuracy"])
        anchor_scores.append(result["anchor_accuracy"])
        explicit_scores.append(
            result["explicitness_accuracy"]
        )
        selected_anchor_scores.append(
            result["selected_anchor_accuracy"]
        )
        negated_anchor_scores.append(
            result["negated_anchor_accuracy"]
        )

        rows.append({
            "seed": seed,
            "direct_accuracy": direct,
            "oracle_accuracy": oracle,
            "global_pool_accuracy":
                result["global_accuracy"],
            "anchor_pool_accuracy":
                result["anchor_accuracy"],
            "explicitness_accuracy":
                result["explicitness_accuracy"],
            "selected_anchor_accuracy":
                result["selected_anchor_accuracy"],
            "negated_anchor_accuracy":
                result["negated_anchor_accuracy"],
            "selected_anchor_confidence":
                result["selected_anchor_confidence"],
            "negated_anchor_confidence":
                result["negated_anchor_confidence"],
            "anchor_gain_vs_global":
                result["anchor_accuracy"]
                - result["global_accuracy"],
            "anchor_gain_vs_direct":
                result["anchor_accuracy"] - direct,
            "anchor_gap_vs_oracle":
                result["anchor_accuracy"] - oracle,
        })

        print(
            f"seed={seed:>3d} "
            f"direct={direct:>6.1%} "
            f"oracle={oracle:>6.1%} "
            f"global={result['global_accuracy']:>6.1%} "
            f"anchor={result['anchor_accuracy']:>6.1%} "
            f"selAnc={result['selected_anchor_accuracy']:>6.1%} "
            f"negAnc={result['negated_anchor_accuracy']:>6.1%} "
            f"delta={result['anchor_accuracy']-result['global_accuracy']:+.1%}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed","direct_accuracy","oracle_accuracy",
        "global_pool_accuracy","anchor_pool_accuracy",
        "explicitness_accuracy",
        "selected_anchor_accuracy",
        "negated_anchor_accuracy",
        "selected_anchor_confidence",
        "negated_anchor_confidence",
        "anchor_gain_vs_global",
        "anchor_gain_vs_direct",
        "anchor_gap_vs_oracle",
    ]

    with (out / "per_seed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    metrics = [
        ("direct_accuracy", direct_scores),
        ("oracle_accuracy", oracle_scores),
        ("global_pool_accuracy", global_scores),
        ("anchor_pool_accuracy", anchor_scores),
        ("explicitness_accuracy", explicit_scores),
        ("selected_anchor_accuracy", selected_anchor_scores),
        ("negated_anchor_accuracy", negated_anchor_scores),
        (
            "anchor_gain_vs_global",
            [
                a-g for a,g in zip(
                    anchor_scores, global_scores
                )
            ],
        ),
        (
            "anchor_gain_vs_direct",
            [
                a-d for a,d in zip(
                    anchor_scores, direct_scores
                )
            ],
        ),
        (
            "anchor_gap_vs_oracle",
            [
                a-o for a,o in zip(
                    anchor_scores, oracle_scores
                )
            ],
        ),
    ]

    with (out / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow(["metric","mean","std","min","max"])
        for name, vals in metrics:
            m, s = mean_std(vals)
            w.writerow([name,m,s,min(vals),max(vals)])

    print()
    print("Multi-seed summary")
    print("-" * 88)

    for label, vals in [
        ("Direct full-phrase baseline", direct_scores),
        ("Oracle gold-span decomposition", oracle_scores),
        ("Global semantic pooling", global_scores),
        ("Semantic anchor pooling", anchor_scores),
        ("Explicitness accuracy", explicit_scores),
        ("SELECTED anchor accuracy", selected_anchor_scores),
        ("NEGATED anchor accuracy", negated_anchor_scores),
    ]:
        m, s = mean_std(vals)
        print(
            f"{label:34s}: "
            f"{m:.1%} ± {s:.1%} "
            f"[{min(vals):.1%}, {max(vals):.1%}]"
        )

    gain_global = [
        a-g for a,g in zip(anchor_scores,global_scores)
    ]
    gain_direct = [
        a-d for a,d in zip(anchor_scores,direct_scores)
    ]
    gap_oracle = [
        a-o for a,o in zip(anchor_scores,oracle_scores)
    ]

    gm,gs = mean_std(gain_global)
    dm,ds = mean_std(gain_direct)
    om,os = mean_std(gap_oracle)

    print(
        f"{'Anchor gain vs global':34s}: "
        f"{gm:+.1%} ± {gs:.1%}"
    )
    print(
        f"{'Anchor gain vs direct':34s}: "
        f"{dm:+.1%} ± {ds:.1%}"
    )
    print(
        f"{'Anchor gap vs oracle':34s}: "
        f"{om:+.1%} ± {os:.1%}"
    )

    print()
    print("Reference")
    print("---------")
    print("v1.4.15 global token pooling         : 77.7% ± 1.1%")
    print("v1.4.16 learned span re-encoding     : 75.5% ± 2.4%")
    print("v1.4.17 boundary re-encoding         : 71.6% ± 2.7%")
    print("v1.4.13 rule-based automatic         : 90.4% ± 3.6%")
    print("v1.4.12 oracle                       : 99.9% ± 0.6%")
    print()
    print("Interpretation")
    print("--------------")
    print(
        "This experiment avoids literal span boundaries. "
        "The highest-scoring semantic-role token is treated as an anchor, "
        "and a small local window is softly pooled using role probabilities."
    )
    print(
        "Improvement over global pooling would support the hypothesis "
        "that block5 preserves semantic role locality but not exact text boundaries."
    )
    print("Output dir:", out)


if __name__ == "__main__":
    main()
