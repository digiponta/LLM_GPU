# learned_span_reencoding_v1416.py
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
from automatic_semantic_decomposition_v1413 import eval_direct, eval_oracle, encode_cached
from learned_semantic_span_detector_v1414 import (
    encode_token_states,
    token_labels,
    build_folds,
    NEGATED,
    SELECTED,
)
from explicit_implicit_semantic_decomposition_v1415 import (
    train_binary_span_tagger,
    train_explicitness_head,
    predict_explicit,
)

STAGE = "block5"


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def train_standalone_position_head(
    model, tok, train_indices, args, seed, device, cache
):
    vectors = []
    labels = []

    for i in train_indices:
        s = CONTRAST_SAMPLES[i]

        sel_vec = encode_cached(
            model, tok, cache, s["selected_text"]
        )
        neg_vec = encode_cached(
            model, tok, cache, s["negated_text"]
        )

        vectors.append(sel_vec)
        labels.append(s["selected_position"])
        vectors.append(neg_vec)
        labels.append(s["negated_position"])

    X = torch.stack(vectors).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    Xs = (X - mean) / std

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], args.position_hidden),
        torch.nn.GELU(),
        torch.nn.Linear(args.position_hidden, 2),
    ).to(device)

    opt = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for _ in range(args.epochs):
        opt.zero_grad()
        loss = F.cross_entropy(head(Xs), y)
        loss.backward()
        opt.step()

    head.eval()
    return head, mean, std


def best_contiguous_span(prob, offsets, threshold):
    """
    Convert token probabilities into one contiguous character span.
    If nothing crosses threshold, fall back to the highest-probability token.
    Adjacent/overlapping selected tokens are merged, and the segment with the
    highest summed probability is returned.
    """
    probs = [float(x) for x in prob.detach().cpu()]
    active = [i for i, p in enumerate(probs) if p >= threshold]

    if not active:
        active = [max(range(len(probs)), key=lambda i: probs[i])]

    groups = []
    cur = [active[0]]
    for idx in active[1:]:
        prev = cur[-1]
        if idx == prev + 1:
            cur.append(idx)
        else:
            groups.append(cur)
            cur = [idx]
    groups.append(cur)

    def score(group):
        return sum(probs[i] for i in group)

    best = max(groups, key=score)
    start = min(offsets[i][0] for i in best)
    end = max(offsets[i][1] for i in best)
    confidence = max(probs[i] for i in best)
    return start, end, confidence


@torch.no_grad()
def detect_span_text(
    raw_text, states, offsets, tagger, threshold, device
):
    X = states.to(device)
    prob = torch.sigmoid(tagger(X))
    start, end, confidence = best_contiguous_span(
        prob, offsets, threshold
    )
    text = raw_text[start:end].strip(" 、，,。；;")
    if not text:
        text = raw_text
    return text, confidence


@torch.no_grad()
def position_probs_from_text(
    model, tok, cache, text, head, mean, std, device
):
    vec = encode_cached(model, tok, cache, text).to(device)
    x = (vec[None, :] - mean) / std
    return torch.softmax(head(x), dim=-1)[0]


def eval_reencoding(
    model, tok,
    states_all, offsets_all, labels_all,
    args, seed, device
):
    pred = torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)
    explicit_pred = torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)

    selected_text_exact = []
    negated_text_exact = []
    selected_detect_conf = []
    negated_detect_conf = []
    reencode_cache = {}

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

        pos_head, pmean, pstd = train_standalone_position_head(
            model, tok, train, args,
            seed + 30000 + 1000 * fold,
            device,
            reencode_cache,
        )

        for i in test:
            raw = CONTRAST_SAMPLES[i]["text"]
            states = states_all[i]
            offsets = offsets_all[i]

            sel_text, sel_conf = detect_span_text(
                raw, states, offsets,
                sel_tagger,
                args.span_threshold,
                device,
            )

            exp_hat, _ = predict_explicit(
                states, explicit_head, emean, estd, device
            )
            explicit_pred[i] = exp_hat

            sel_pos = position_probs_from_text(
                model, tok, reencode_cache,
                sel_text,
                pos_head, pmean, pstd, device
            )
            fused = sel_pos.clone()

            if exp_hat == 1:
                neg_text, neg_conf = detect_span_text(
                    raw, states, offsets,
                    neg_tagger,
                    args.span_threshold,
                    device,
                )
                neg_pos = position_probs_from_text(
                    model, tok, reencode_cache,
                    neg_text,
                    pos_head, pmean, pstd, device
                )
                fused[0] += args.negated_weight * neg_pos[1]
                fused[1] += args.negated_weight * neg_pos[0]

                gold_neg = CONTRAST_SAMPLES[i]["negated_text"]
                negated_text_exact.append(
                    neg_text in gold_neg or gold_neg in neg_text
                )
                negated_detect_conf.append(neg_conf)

            pred[i] = int(fused.argmax())

            gold_sel = CONTRAST_SAMPLES[i]["selected_text"]
            selected_text_exact.append(
                sel_text in gold_sel or gold_sel in sel_text
            )
            selected_detect_conf.append(sel_conf)

    gold = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
    )
    explicit_gold = torch.tensor([
        int((lab == NEGATED).any())
        for lab in labels_all
    ], dtype=torch.long)

    return {
        "accuracy": float((pred == gold).float().mean()),
        "explicitness_accuracy": float(
            (explicit_pred == explicit_gold).float().mean()
        ),
        "selected_text_match": (
            sum(selected_text_exact) / len(selected_text_exact)
        ),
        "negated_text_match": (
            sum(negated_text_exact) / len(negated_text_exact)
            if negated_text_exact else 0.0
        ),
        "selected_confidence": statistics.mean(selected_detect_conf),
        "negated_confidence": (
            statistics.mean(negated_detect_conf)
            if negated_detect_conf else 0.0
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
    ap.add_argument("--span-threshold", type=float, default=0.50)
    ap.add_argument("--negated-weight", type=float, default=1.0)
    ap.add_argument(
        "--output-dir",
        default="results/learned_span_reencoding_v1416",
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
    print(" LLM_GPU v1.4.16 Learned Span Re-encoding")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5")
    print("Span detector    : learned token-level SELECTED / NEGATED")
    print("Span use         : detected tokens -> text -> frozen LLM re-encoding")
    print("Explicit route   : SELECTED + NEGATED -> binder")
    print("Implicit route   : SELECTED only; NEGATED inferred as opposite")
    print("Evaluation       : 5-fold unseen-paraphrase holdout")
    print("Relation seeds   :", f"{seeds[0]}..{seeds[-1]}")
    print("Base model       : completely frozen")
    print()

    print("Encoding raw-phrase token states...")
    states_all = []
    offsets_all = []
    labels_all = []

    for sample in CONTRAST_SAMPLES:
        states, offsets = encode_token_states(
            model, tok, sample["text"]
        )
        labels = token_labels(sample, offsets)
        states_all.append(states)
        offsets_all.append(offsets)
        labels_all.append(labels)

    explicit_count = sum(
        bool((lab == NEGATED).any())
        for lab in labels_all
    )
    print(
        f"Gold decomposition: explicit={explicit_count}, "
        f"implicit={len(labels_all)-explicit_count}"
    )

    reference_cache = {}
    Xfull = torch.stack([
        encode_cached(
            model, tok, reference_cache, s["text"]
        )
        for s in CONTRAST_SAMPLES
    ]).to(device)
    Xneg = torch.stack([
        encode_cached(
            model, tok, reference_cache, s["negated_text"]
        )
        for s in CONTRAST_SAMPLES
    ]).to(device)
    Xsel = torch.stack([
        encode_cached(
            model, tok, reference_cache, s["selected_text"]
        )
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
    reencode_scores = []
    explicit_scores = []
    selected_match_scores = []
    negated_match_scores = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(
            Xneg, Xsel, yneg, ysel, args, seed
        )
        result = eval_reencoding(
            model, tok,
            states_all, offsets_all, labels_all,
            args, seed, device
        )

        direct_scores.append(direct)
        oracle_scores.append(oracle)
        reencode_scores.append(result["accuracy"])
        explicit_scores.append(
            result["explicitness_accuracy"]
        )
        selected_match_scores.append(
            result["selected_text_match"]
        )
        negated_match_scores.append(
            result["negated_text_match"]
        )

        rows.append({
            "seed": seed,
            "direct_accuracy": direct,
            "oracle_accuracy": oracle,
            "reencoding_accuracy": result["accuracy"],
            "explicitness_accuracy":
                result["explicitness_accuracy"],
            "selected_text_match":
                result["selected_text_match"],
            "negated_text_match":
                result["negated_text_match"],
            "selected_confidence":
                result["selected_confidence"],
            "negated_confidence":
                result["negated_confidence"],
            "gain_vs_direct":
                result["accuracy"] - direct,
            "gap_vs_oracle":
                result["accuracy"] - oracle,
        })

        print(
            f"seed={seed:>3d} "
            f"direct={direct:>6.1%} "
            f"oracle={oracle:>6.1%} "
            f"reenc={result['accuracy']:>6.1%} "
            f"explicit={result['explicitness_accuracy']:>6.1%} "
            f"selTxt={result['selected_text_match']:>6.1%} "
            f"negTxt={result['negated_text_match']:>6.1%} "
            f"gain={result['accuracy']-direct:+.1%}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed","direct_accuracy","oracle_accuracy",
        "reencoding_accuracy","explicitness_accuracy",
        "selected_text_match","negated_text_match",
        "selected_confidence","negated_confidence",
        "gain_vs_direct","gap_vs_oracle",
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
        ("reencoding_accuracy", reencode_scores),
        ("explicitness_accuracy", explicit_scores),
        ("selected_text_match", selected_match_scores),
        ("negated_text_match", negated_match_scores),
        (
            "gain_vs_direct",
            [
                r - d
                for r, d in zip(
                    reencode_scores, direct_scores
                )
            ],
        ),
        (
            "gap_vs_oracle",
            [
                r - o
                for r, o in zip(
                    reencode_scores, oracle_scores
                )
            ],
        ),
    ]

    with (out / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            ["metric","mean","std","min","max"]
        )
        for name, vals in metrics:
            m, s = mean_std(vals)
            w.writerow(
                [name,m,s,min(vals),max(vals)]
            )

    print()
    print("Multi-seed summary")
    print("-" * 88)

    for label, vals in [
        ("Direct full-phrase baseline", direct_scores),
        ("Oracle gold-span decomposition", oracle_scores),
        ("Learned span + re-encoding", reencode_scores),
        ("Explicitness accuracy", explicit_scores),
        ("SELECTED text match", selected_match_scores),
        ("NEGATED text match", negated_match_scores),
    ]:
        m, s = mean_std(vals)
        print(
            f"{label:34s}: "
            f"{m:.1%} ± {s:.1%} "
            f"[{min(vals):.1%}, {max(vals):.1%}]"
        )

    gain = [
        r-d for r,d in zip(
            reencode_scores,direct_scores
        )
    ]
    gap = [
        r-o for r,o in zip(
            reencode_scores,oracle_scores
        )
    ]
    gm,gs = mean_std(gain)
    xm,xs = mean_std(gap)

    print(
        f"{'Re-encoding gain vs direct':34s}: "
        f"{gm:+.1%} ± {gs:.1%}"
    )
    print(
        f"{'Re-encoding gap vs oracle':34s}: "
        f"{xm:+.1%} ± {xs:.1%}"
    )

    print()
    print("Reference")
    print("---------")
    print(
        "v1.4.13 rule-based re-encoding      "
        ": 90.4% ± 3.6%"
    )
    print(
        "v1.4.15 learned token pooling       "
        ": 77.7% ± 1.1%"
    )
    print(
        "v1.4.15 explicitness                "
        ": 98.2% ± 2.3%"
    )
    print(
        "v1.4.12 oracle re-encoding          "
        ": 99.9% ± 0.6%"
    )

    print()
    print("Interpretation")
    print("--------------")
    print(
        "The learned detector is unchanged in principle, "
        "but detected token spans are reconstructed as text "
        "and passed through the frozen LLM again."
    )
    print(
        "If accuracy returns toward the v1.4.13 range, "
        "the dominant v1.4.15 bottleneck was token pooling "
        "rather than semantic span detection."
    )
    print("Output dir:", out)


if __name__ == "__main__":
    main()
