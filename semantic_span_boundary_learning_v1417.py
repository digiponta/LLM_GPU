# semantic_span_boundary_learning_v1417.py
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
from automatic_semantic_decomposition_v1413 import eval_direct, eval_oracle, encode_cached
from learned_semantic_span_detector_v1414 import (
    encode_token_states,
    token_labels,
    build_folds,
    NEGATED,
    SELECTED,
)
from explicit_implicit_semantic_decomposition_v1415 import (
    train_explicitness_head,
    predict_explicit,
)
from learned_span_reencoding_v1416 import (
    train_standalone_position_head,
    position_probs_from_text,
)

class BoundaryPointer(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.start = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.end = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.start(x).squeeze(-1), self.end(x).squeeze(-1)


def gold_boundary(labels, role):
    idx = torch.nonzero(labels == role, as_tuple=False).flatten()
    if len(idx) == 0:
        return None
    return int(idx.min()), int(idx.max())


def train_boundary_model(states_all, labels_all, train, role, args, seed, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    d_in = states_all[train[0]].shape[1]
    model = BoundaryPointer(d_in, args.boundary_hidden).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    usable = [
        i for i in train
        if gold_boundary(labels_all[i], role) is not None
    ]
    if not usable:
        raise RuntimeError("No usable spans for boundary training")

    for _ in range(args.epochs):
        opt.zero_grad()
        losses = []
        for i in usable:
            X = states_all[i].to(device)
            gs, ge = gold_boundary(labels_all[i], role)
            s_logit, e_logit = model(X)
            losses.append(
                F.cross_entropy(
                    s_logit[None, :],
                    torch.tensor([gs], device=device),
                )
                + F.cross_entropy(
                    e_logit[None, :],
                    torch.tensor([ge], device=device),
                )
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

    model.eval()
    return model


@torch.no_grad()
def predict_boundary(model, states, max_span_tokens=12):
    X = states.to(next(model.parameters()).device)
    s_logit, e_logit = model(X)
    s_logp = F.log_softmax(s_logit, dim=0)
    e_logp = F.log_softmax(e_logit, dim=0)

    n = len(states)
    best = None
    best_score = float("-inf")
    for s in range(n):
        max_e = min(n - 1, s + max_span_tokens - 1)
        for e in range(s, max_e + 1):
            score = float(s_logp[s] + e_logp[e])
            if score > best_score:
                best_score = score
                best = (s, e)
    return best


def boundary_to_text(raw, offsets, boundary):
    s, e = boundary
    start = offsets[s][0]
    end = offsets[e][1]
    text = raw[start:end].strip(" 、，,。；;")
    return text if text else raw


def fuzzy_text_match(pred, gold):
    pred = pred.strip(" 、，,。；;")
    gold = gold.strip(" 、，,。；;")
    return pred == gold or pred in gold or gold in pred


def eval_boundary_reencoding(
    model, tok, states_all, offsets_all, labels_all,
    args, seed, device,
):
    pred = torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)
    explicit_pred = torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)

    sel_match = []
    neg_match = []
    sel_boundary_exact = []
    neg_boundary_exact = []
    cache = {}

    for fold, test in enumerate(build_folds()):
        test_set = set(test)
        train = [
            i for i in range(len(CONTRAST_SAMPLES))
            if i not in test_set
        ]

        sel_boundary_model = train_boundary_model(
            states_all, labels_all, train, SELECTED,
            args, seed + 1000 * fold, device,
        )
        neg_boundary_model = train_boundary_model(
            states_all, labels_all, train, NEGATED,
            args, seed + 10000 + 1000 * fold, device,
        )

        explicit_head, emean, estd = train_explicitness_head(
            [states_all[i] for i in train],
            [labels_all[i] for i in train],
            args, seed + 20000 + 1000 * fold, device,
        )

        pos_head, pmean, pstd = train_standalone_position_head(
            model, tok, train, args,
            seed + 30000 + 1000 * fold,
            device, cache,
        )

        for i in test:
            raw = CONTRAST_SAMPLES[i]["text"]
            states = states_all[i]
            offsets = offsets_all[i]

            sel_b = predict_boundary(
                sel_boundary_model, states, args.max_span_tokens
            )
            sel_text = boundary_to_text(raw, offsets, sel_b)
            sel_pos = position_probs_from_text(
                model, tok, cache, sel_text,
                pos_head, pmean, pstd, device,
            )
            fused = sel_pos.clone()

            gold_sel_b = gold_boundary(labels_all[i], SELECTED)
            sel_boundary_exact.append(sel_b == gold_sel_b)
            sel_match.append(
                fuzzy_text_match(sel_text, CONTRAST_SAMPLES[i]["selected_text"])
            )

            exp_hat, _ = predict_explicit(
                states, explicit_head, emean, estd, device
            )
            explicit_pred[i] = exp_hat

            if exp_hat == 1:
                neg_b = predict_boundary(
                    neg_boundary_model, states, args.max_span_tokens
                )
                neg_text = boundary_to_text(raw, offsets, neg_b)
                neg_pos = position_probs_from_text(
                    model, tok, cache, neg_text,
                    pos_head, pmean, pstd, device,
                )
                fused[0] += args.negated_weight * neg_pos[1]
                fused[1] += args.negated_weight * neg_pos[0]

                gold_neg_b = gold_boundary(labels_all[i], NEGATED)
                if gold_neg_b is not None:
                    neg_boundary_exact.append(neg_b == gold_neg_b)
                    neg_match.append(
                        fuzzy_text_match(
                            neg_text,
                            CONTRAST_SAMPLES[i]["negated_text"],
                        )
                    )

            pred[i] = int(fused.argmax())

    gold = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
    )
    explicit_gold = torch.tensor([
        int(gold_boundary(lab, NEGATED) is not None)
        for lab in labels_all
    ], dtype=torch.long)

    return {
        "accuracy": float((pred == gold).float().mean()),
        "explicitness_accuracy": float(
            (explicit_pred == explicit_gold).float().mean()
        ),
        "selected_text_match": sum(sel_match) / len(sel_match),
        "negated_text_match": (
            sum(neg_match) / len(neg_match) if neg_match else 0.0
        ),
        "selected_boundary_exact": (
            sum(sel_boundary_exact) / len(sel_boundary_exact)
        ),
        "negated_boundary_exact": (
            sum(neg_boundary_exact) / len(neg_boundary_exact)
            if neg_boundary_exact else 0.0
        ),
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
    ap.add_argument("--boundary-hidden", type=int, default=64)
    ap.add_argument("--position-hidden", type=int, default=32)
    ap.add_argument("--explicit-hidden", type=int, default=32)
    ap.add_argument("--max-span-tokens", type=int, default=12)
    ap.add_argument("--negated-weight", type=float, default=1.0)
    ap.add_argument(
        "--output-dir",
        default="results/semantic_span_boundary_learning_v1417",
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
    print(" LLM_GPU v1.4.17 Semantic Span Boundary Learning")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Boundary model   : SELECTED start/end + NEGATED start/end")
    print("Span use         : exact boundary -> text -> frozen LLM re-encoding")
    print("Explicit route   : SELECTED + NEGATED -> binder")
    print("Implicit route   : SELECTED only; NEGATED inferred as opposite")
    print("Evaluation       : 5-fold unseen-paraphrase holdout")
    print("Relation seeds   :", f"{seeds[0]}..{seeds[-1]}")
    print("Base model       : completely frozen")
    print()

    print("Encoding block5 token states...")
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
        gold_boundary(lab, NEGATED) is not None
        for lab in labels_all
    )
    print(
        f"Gold decomposition: explicit={explicit_count}, "
        f"implicit={len(labels_all)-explicit_count}"
    )

    ref_cache = {}
    Xfull = torch.stack([
        encode_cached(model, tok, ref_cache, s["text"])
        for s in CONTRAST_SAMPLES
    ]).to(device)
    Xneg = torch.stack([
        encode_cached(model, tok, ref_cache, s["negated_text"])
        for s in CONTRAST_SAMPLES
    ]).to(device)
    Xsel = torch.stack([
        encode_cached(model, tok, ref_cache, s["selected_text"])
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

    rows = []
    direct_scores = []
    oracle_scores = []
    boundary_scores = []
    explicit_scores = []
    sel_text_scores = []
    neg_text_scores = []
    sel_boundary_scores = []
    neg_boundary_scores = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(
            Xneg, Xsel, yneg, ysel, args, seed
        )
        result = eval_boundary_reencoding(
            model, tok,
            states_all, offsets_all, labels_all,
            args, seed, device,
        )

        direct_scores.append(direct)
        oracle_scores.append(oracle)
        boundary_scores.append(result["accuracy"])
        explicit_scores.append(result["explicitness_accuracy"])
        sel_text_scores.append(result["selected_text_match"])
        neg_text_scores.append(result["negated_text_match"])
        sel_boundary_scores.append(result["selected_boundary_exact"])
        neg_boundary_scores.append(result["negated_boundary_exact"])

        rows.append({
            "seed": seed,
            "direct_accuracy": direct,
            "oracle_accuracy": oracle,
            "boundary_accuracy": result["accuracy"],
            "explicitness_accuracy": result["explicitness_accuracy"],
            "selected_text_match": result["selected_text_match"],
            "negated_text_match": result["negated_text_match"],
            "selected_boundary_exact": result["selected_boundary_exact"],
            "negated_boundary_exact": result["negated_boundary_exact"],
            "gain_vs_direct": result["accuracy"] - direct,
            "gap_vs_oracle": result["accuracy"] - oracle,
        })

        print(
            f"seed={seed:>3d} "
            f"direct={direct:>6.1%} "
            f"oracle={oracle:>6.1%} "
            f"boundary={result['accuracy']:>6.1%} "
            f"selTxt={result['selected_text_match']:>6.1%} "
            f"negTxt={result['negated_text_match']:>6.1%} "
            f"selB={result['selected_boundary_exact']:>6.1%} "
            f"negB={result['negated_boundary_exact']:>6.1%}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed","direct_accuracy","oracle_accuracy",
        "boundary_accuracy","explicitness_accuracy",
        "selected_text_match","negated_text_match",
        "selected_boundary_exact","negated_boundary_exact",
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
        ("boundary_accuracy", boundary_scores),
        ("explicitness_accuracy", explicit_scores),
        ("selected_text_match", sel_text_scores),
        ("negated_text_match", neg_text_scores),
        ("selected_boundary_exact", sel_boundary_scores),
        ("negated_boundary_exact", neg_boundary_scores),
        (
            "gain_vs_direct",
            [b-d for b,d in zip(boundary_scores,direct_scores)]
        ),
        (
            "gap_vs_oracle",
            [b-o for b,o in zip(boundary_scores,oracle_scores)]
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
        ("Boundary-learned re-encoding", boundary_scores),
        ("Explicitness accuracy", explicit_scores),
        ("SELECTED text match", sel_text_scores),
        ("NEGATED text match", neg_text_scores),
        ("SELECTED exact boundary", sel_boundary_scores),
        ("NEGATED exact boundary", neg_boundary_scores),
    ]:
        m, s = mean_std(vals)
        print(
            f"{label:34s}: {m:.1%} ± {s:.1%} "
            f"[{min(vals):.1%}, {max(vals):.1%}]"
        )

    gain = [b-d for b,d in zip(boundary_scores,direct_scores)]
    gap = [b-o for b,o in zip(boundary_scores,oracle_scores)]
    gm,gs = mean_std(gain)
    xm,xs = mean_std(gap)

    print(f"{'Boundary gain vs direct':34s}: {gm:+.1%} ± {gs:.1%}")
    print(f"{'Boundary gap vs oracle':34s}: {xm:+.1%} ± {xs:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.16 learned span re-encoding     : 75.5% ± 2.4%")
    print("v1.4.16 SELECTED text match          : 70.2% ± 2.3%")
    print("v1.4.16 NEGATED text match           : 73.9% ± 3.4%")
    print("v1.4.13 rule-based automatic         : 90.4% ± 3.6%")
    print("v1.4.12 oracle                       : 99.9% ± 0.6%")
    print()
    print("Interpretation")
    print("--------------")
    print("This experiment replaces thresholded token grouping with explicit")
    print("START/END boundary prediction. If text-match and final accuracy rise")
    print("substantially, span-boundary reconstruction was the dominant bottleneck.")
    print("Output dir:", out)


if __name__ == "__main__":
    main()
