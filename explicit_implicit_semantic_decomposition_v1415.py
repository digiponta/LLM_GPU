# explicit_implicit_semantic_decomposition_v1415.py
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch
import torch.nn as nn
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

class BinarySpanTagger(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ExplicitnessHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_binary_span_tagger(train_states, train_labels, positive_role, args, seed, device):
    X = torch.cat(train_states, dim=0).to(device)
    y = torch.cat([
        (lab == positive_role).float() for lab in train_labels
    ], dim=0).to(device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = BinarySpanTagger(X.shape[1], args.span_hidden).to(device)
    pos = y.sum().clamp_min(1.0)
    neg = (1.0 - y).sum().clamp_min(1.0)
    pos_weight = (neg / pos).detach()

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    for _ in range(args.epochs):
        opt.zero_grad()
        logits = model(X)
        loss = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight
        )
        loss.backward()
        opt.step()

    model.eval()
    return model


def train_explicitness_head(train_states, train_labels, args, seed, device):
    X = torch.stack([x.mean(0) for x in train_states]).to(device)
    y = torch.tensor([
        int((lab == NEGATED).any()) for lab in train_labels
    ], dtype=torch.long, device=device)

    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    Xs = (X - mean) / std

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head = ExplicitnessHead(X.shape[1], args.explicit_hidden).to(device)
    counts = torch.bincount(y, minlength=2).float().clamp_min(1)
    weights = counts.sum() / counts
    weights = weights / weights.mean()

    opt = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = F.cross_entropy(head(Xs), y, weight=weights)
        loss.backward()
        opt.step()

    head.eval()
    return head, mean, std


@torch.no_grad()
def weighted_pool(states, tagger, device):
    X = states.to(device)
    weights = torch.sigmoid(tagger(X))
    vec = (weights[:, None] * X).sum(0) / weights.sum().clamp_min(1e-6)
    detected = bool((weights >= 0.5).any())
    confidence = float(weights.max())
    return vec, detected, confidence


@torch.no_grad()
def position_probs(vec, head, mean, std):
    x = ((vec[None, :] - mean) / std)
    return torch.softmax(head(x), dim=-1)[0]


@torch.no_grad()
def predict_explicit(states, head, mean, std, device):
    vec = states.to(device).mean(0, keepdim=True)
    p = torch.softmax(head((vec - mean) / std), dim=-1)[0]
    return int(p.argmax()), float(p[1])


def eval_hybrid(states_all, labels_all, args, seed, device):
    pred = torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)
    explicit_pred = torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)

    selected_detect = []
    negated_detect_explicit = []
    selected_conf = []
    negated_conf = []

    for fold, test in enumerate(build_folds()):
        test_set = set(test)
        train = [
            i for i in range(len(CONTRAST_SAMPLES))
            if i not in test_set
        ]

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
            position_vectors, position_labels, args,
            seed + 30000 + 1000 * fold, device,
        )

        for i in test:
            states = states_all[i]
            sel_vec, sel_det, sel_conf = weighted_pool(
                states, sel_tagger, device
            )
            neg_vec, neg_det, neg_conf = weighted_pool(
                states, neg_tagger, device
            )
            exp_hat, exp_conf = predict_explicit(
                states, explicit_head, emean, estd, device
            )
            explicit_pred[i] = exp_hat

            sel_pos = position_probs(sel_vec, pos_head, pmean, pstd)
            fused = sel_pos.clone()

            # Explicit contrast: use the detected NEGATED span as supporting evidence.
            # Implicit contrast: do not force a literal NEGATED span. Semantically,
            # NEGATED is inferred as opposite(SELECTED), so the selected-position
            # evidence remains the primary route.
            if exp_hat == 1 and neg_conf >= args.negated_conf_threshold:
                neg_pos = position_probs(neg_vec, pos_head, pmean, pstd)
                fused[0] += args.negated_weight * neg_pos[1]
                fused[1] += args.negated_weight * neg_pos[0]
                negated_detect_explicit.append(neg_det)

            pred[i] = int(fused.argmax())
            selected_detect.append(sel_det)
            selected_conf.append(sel_conf)
            negated_conf.append(neg_conf)

    gold = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
    )
    explicit_gold = torch.tensor([
        int((lab == NEGATED).any()) for lab in labels_all
    ], dtype=torch.long)

    return {
        "accuracy": float((pred == gold).float().mean()),
        "explicitness_accuracy": float(
            (explicit_pred == explicit_gold).float().mean()
        ),
        "selected_detection": sum(selected_detect) / len(selected_detect),
        "negated_detection_when_explicit": (
            sum(negated_detect_explicit) / len(negated_detect_explicit)
            if negated_detect_explicit else 0.0
        ),
        "selected_confidence": statistics.mean(selected_conf),
        "negated_confidence": statistics.mean(negated_conf),
    }


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


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
    ap.add_argument("--negated-conf-threshold", type=float, default=0.45)
    ap.add_argument("--negated-weight", type=float, default=1.0)
    ap.add_argument(
        "--output-dir",
        default="results/explicit_implicit_semantic_decomposition_v1415",
    )
    args = ap.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

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
    print(" LLM_GPU v1.4.15 Explicit / Implicit Semantic Decomposition")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Primary route    : learned SELECTED span -> position")
    print("Explicit route   : + learned NEGATED span -> binder")
    print("Implicit route   : NEGATED := opposite(SELECTED)")
    print("Explicitness     : learned phrase-level classifier")
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
        bool((x == NEGATED).any()) for x in labels_all
    )
    implicit_count = len(labels_all) - explicit_count
    print(
        f"Gold decomposition: explicit={explicit_count}, "
        f"implicit={implicit_count}"
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
        dtype=torch.long, device=device
    )
    yneg = torch.tensor(
        [s["negated_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long, device=device
    )

    rows = []
    direct_scores = []
    oracle_scores = []
    hybrid_scores = []
    explicit_scores = []
    selected_detection_scores = []
    negated_detection_scores = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(
            Xneg, Xsel, yneg, ysel, args, seed
        )
        hybrid = eval_hybrid(
            states_all, labels_all, args, seed, device
        )

        direct_scores.append(direct)
        oracle_scores.append(oracle)
        hybrid_scores.append(hybrid["accuracy"])
        explicit_scores.append(hybrid["explicitness_accuracy"])
        selected_detection_scores.append(hybrid["selected_detection"])
        negated_detection_scores.append(
            hybrid["negated_detection_when_explicit"]
        )

        rows.append({
            "seed": seed,
            "direct_accuracy": direct,
            "oracle_accuracy": oracle,
            "hybrid_accuracy": hybrid["accuracy"],
            "explicitness_accuracy": hybrid["explicitness_accuracy"],
            "selected_detection": hybrid["selected_detection"],
            "negated_detection_when_explicit":
                hybrid["negated_detection_when_explicit"],
            "selected_confidence": hybrid["selected_confidence"],
            "negated_confidence": hybrid["negated_confidence"],
            "gain_vs_direct": hybrid["accuracy"] - direct,
            "gap_vs_oracle": hybrid["accuracy"] - oracle,
        })

        print(
            f"seed={seed:>3d} direct={direct:>6.1%} "
            f"oracle={oracle:>6.1%} hybrid={hybrid['accuracy']:>6.1%} "
            f"explicit={hybrid['explicitness_accuracy']:>6.1%} "
            f"selDet={hybrid['selected_detection']:>6.1%} "
            f"negDet={hybrid['negated_detection_when_explicit']:>6.1%} "
            f"gain={hybrid['accuracy']-direct:+.1%}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed","direct_accuracy","oracle_accuracy","hybrid_accuracy",
        "explicitness_accuracy","selected_detection",
        "negated_detection_when_explicit","selected_confidence",
        "negated_confidence","gain_vs_direct","gap_vs_oracle",
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
        ("hybrid_accuracy", hybrid_scores),
        ("explicitness_accuracy", explicit_scores),
        ("selected_detection", selected_detection_scores),
        ("negated_detection_when_explicit", negated_detection_scores),
        (
            "gain_vs_direct",
            [h - d for h, d in zip(hybrid_scores, direct_scores)]
        ),
        (
            "gap_vs_oracle",
            [h - o for h, o in zip(hybrid_scores, oracle_scores)]
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
        ("Explicit/Implicit hybrid", hybrid_scores),
        ("Explicitness accuracy", explicit_scores),
        ("SELECTED span detection", selected_detection_scores),
        ("NEGATED detection (explicit)", negated_detection_scores),
    ]:
        m, s = mean_std(vals)
        print(
            f"{label:34s}: {m:.1%} ± {s:.1%} "
            f"[{min(vals):.1%}, {max(vals):.1%}]"
        )

    gain = [h-d for h,d in zip(hybrid_scores,direct_scores)]
    gap = [h-o for h,o in zip(hybrid_scores,oracle_scores)]
    gm,gs = mean_std(gain)
    xm,xs = mean_std(gap)
    print(f"{'Hybrid gain vs direct':34s}: {gm:+.1%} ± {gs:.1%}")
    print(f"{'Hybrid gap vs oracle':34s}: {xm:+.1%} ± {xs:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.13 rule-based automatic         : 90.4% ± 3.6%")
    print("v1.4.14 learned token-span           : 76.7% ± 2.7%")
    print("v1.4.14 SELECTED detection           : 95.0%")
    print("v1.4.14 NEGATED detection            : 74.9% ± 3.8%")
    print()
    print("Interpretation")
    print("--------------")
    print("SELECTED is the primary semantic route. NEGATED is used only when")
    print("the phrase-level explicitness classifier predicts an explicit negation.")
    print("For implicit contrast, the missing NEGATED role is inferred as the")
    print("opposite of SELECTED rather than forcing a non-existent literal span.")
    print("Output dir:", out)


if __name__ == "__main__":
    main()
