# semantic_decision_subspace_v1424.py
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
    mass_window,
)

MASS = 0.80
DIMS = [1, 2, 4, 8, 16, 32]


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


class DecisionSubspace(nn.Module):
    def __init__(self, d_in: int, d_sub: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_sub)
        self.norm = nn.LayerNorm(d_sub)
        self.head = nn.Linear(d_sub, 2)

    def forward(self, x):
        z = self.norm(self.proj(x))
        return self.head(z), z


def build_role_examples(states_all, labels_all, tagger, train, role, device):
    X, y = [], []
    for i in train:
        states = states_all[i]
        score = role_scores(states, tagger, device)
        vec, _, _ = mass_window(states, score, MASS)
        X.append(vec.detach())
        if role == SELECTED:
            y.append(CONTRAST_SAMPLES[i]["selected_position"])
        else:
            y.append(CONTRAST_SAMPLES[i]["negated_position"])
    return torch.stack(X).to(device), torch.tensor(y, dtype=torch.long, device=device)


def train_subspace(X, y, d_sub, args, seed, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    Xs = (X - mean) / std

    model = DecisionSubspace(X.shape[1], d_sub).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.subspace_lr,
        weight_decay=args.subspace_weight_decay,
    )

    for _ in range(args.subspace_epochs):
        opt.zero_grad()
        logits, z = model(Xs)
        ce = F.cross_entropy(logits, y)

        # Mild compactness regularizer; avoid arbitrary vector reconstruction.
        center_penalty = z.pow(2).mean()
        loss = ce + args.latent_l2 * center_penalty
        loss.backward()
        opt.step()

    model.eval()
    return model, mean, std


@torch.no_grad()
def subspace_probs(vec, model, mean, std):
    logits, z = model((vec[None, :] - mean) / std)
    return torch.softmax(logits, dim=-1)[0], z[0]


def eval_seed(states_all, labels_all, args, seed, device):
    preds = {
        d: torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)
        for d in DIMS
    }
    selected_margins = {d: [] for d in DIMS}
    negated_margins = {d: [] for d in DIMS}

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

        Xsel, ysel = build_role_examples(
            states_all, labels_all, sel_tagger, train, SELECTED, device
        )

        explicit_train = [
            i for i in train
            if bool((labels_all[i] == NEGATED).any())
        ]
        Xneg, yneg = build_role_examples(
            states_all, labels_all, neg_tagger, explicit_train, NEGATED, device
        )

        models = {}
        for d in DIMS:
            sel_model, sel_mean, sel_std = train_subspace(
                Xsel, ysel, d, args,
                seed + 30000 + 1000 * fold + d,
                device,
            )
            neg_model, neg_mean, neg_std = train_subspace(
                Xneg, yneg, d, args,
                seed + 40000 + 1000 * fold + d,
                device,
            )
            models[d] = (
                sel_model, sel_mean, sel_std,
                neg_model, neg_mean, neg_std,
            )

        for i in test:
            states = states_all[i]
            sel_score = role_scores(states, sel_tagger, device)
            neg_score = role_scores(states, neg_tagger, device)
            sel_vec, _, _ = mass_window(states, sel_score, MASS)

            exp_hat, _ = predict_explicit(
                states, explicit_head, emean, estd, device
            )

            for d in DIMS:
                (
                    sel_model, sel_mean, sel_std,
                    neg_model, neg_mean, neg_std,
                ) = models[d]

                sel_prob, _ = subspace_probs(
                    sel_vec, sel_model, sel_mean, sel_std
                )
                fused = sel_prob.clone()
                selected_margins[d].append(
                    float((sel_prob[1] - sel_prob[0]).detach())
                )

                if exp_hat == 1:
                    neg_vec, _, _ = mass_window(states, neg_score, MASS)
                    neg_prob, _ = subspace_probs(
                        neg_vec, neg_model, neg_mean, neg_std
                    )
                    fused[0] += neg_prob[1]
                    fused[1] += neg_prob[0]
                    negated_margins[d].append(
                        float((neg_prob[1] - neg_prob[0]).detach())
                    )

                preds[d][i] = int(fused.argmax())

    gold = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
    )

    result = {}
    for d in DIMS:
        result[d] = {
            "accuracy": float((preds[d] == gold).float().mean()),
            "selected_margin_abs": statistics.mean(
                abs(x) for x in selected_margins[d]
            ),
            "negated_margin_abs": (
                statistics.mean(abs(x) for x in negated_margins[d])
                if negated_margins[d] else 0.0
            ),
        }
    return result


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
    ap.add_argument("--explicit-hidden", type=int, default=32)

    ap.add_argument("--subspace-epochs", type=int, default=300)
    ap.add_argument("--subspace-lr", type=float, default=1e-3)
    ap.add_argument("--subspace-weight-decay", type=float, default=1e-3)
    ap.add_argument("--latent-l2", type=float, default=1e-4)

    ap.add_argument(
        "--output-dir",
        default="results/semantic_decision_subspace_v1424",
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
    print(" LLM_GPU v1.4.24 Semantic Decision Subspace / Dimensionality Sweep")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Semantic field   : adaptive 80% role-score mass")
    print("Subspace dims    :", ", ".join(map(str, DIMS)))
    print("Objective        : direct FIRST/SECOND classification")
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

    per_dim = {d: [] for d in DIMS}
    sel_margin = {d: [] for d in DIMS}
    neg_margin = {d: [] for d in DIMS}
    direct_scores, oracle_scores = [], []
    rows = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(Xneg, Xsel, yneg, ysel, args, seed)
        result = eval_seed(states_all, labels_all, args, seed, device)

        direct_scores.append(direct)
        oracle_scores.append(oracle)

        row = {
            "seed": seed,
            "direct_accuracy": direct,
            "oracle_accuracy": oracle,
        }

        acc_text = []
        for d in DIMS:
            acc = result[d]["accuracy"]
            per_dim[d].append(acc)
            sel_margin[d].append(result[d]["selected_margin_abs"])
            neg_margin[d].append(result[d]["negated_margin_abs"])

            row[f"dim_{d}_accuracy"] = acc
            row[f"dim_{d}_selected_margin_abs"] = result[d]["selected_margin_abs"]
            row[f"dim_{d}_negated_margin_abs"] = result[d]["negated_margin_abs"]
            acc_text.append(f"d{d}={acc:.1%}")

        rows.append(row)

        print(
            f"seed={seed:>3d} direct={direct:>6.1%} "
            + " ".join(acc_text)
            + f" oracle={oracle:>6.1%}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = ["seed", "direct_accuracy"]
    for d in DIMS:
        fields.extend([
            f"dim_{d}_accuracy",
            f"dim_{d}_selected_margin_abs",
            f"dim_{d}_negated_margin_abs",
        ])
    fields.append("oracle_accuracy")

    with (out / "per_seed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary_rows = []
    for d in DIMS:
        am, ast = mean_std(per_dim[d])
        sm, sst = mean_std(sel_margin[d])
        nm, nst = mean_std(neg_margin[d])
        summary_rows.append(
            (d, am, ast, min(per_dim[d]), max(per_dim[d]),
             sm, sst, nm, nst)
        )

    with (out / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "dimension",
            "accuracy_mean","accuracy_std","accuracy_min","accuracy_max",
            "selected_margin_abs_mean","selected_margin_abs_std",
            "negated_margin_abs_mean","negated_margin_abs_std",
        ])
        w.writerows(summary_rows)

    ranked = sorted(
        [(d, mean_std(per_dim[d])[0], mean_std(per_dim[d])[1]) for d in DIMS],
        key=lambda x: x[1],
        reverse=True,
    )
    best_d, best_mean, best_std = ranked[0]

    dm, ds = mean_std(direct_scores)
    om, os = mean_std(oracle_scores)

    print()
    print("Dimensionality sweep summary")
    print("-" * 96)
    for d, am, ast, amin, amax, sm, sst, nm, nst in summary_rows:
        print(
            f"dim={d:>2d}: {am:.1%} ± {ast:.1%} "
            f"[{amin:.1%}, {amax:.1%}]  "
            f"selMargin={sm:.3f}  negMargin={nm:.3f}"
        )

    print()
    print("Best dimension                  :", best_d)
    print(f"Best mean accuracy              : {best_mean:.1%} ± {best_std:.1%}")
    print(f"Direct baseline                 : {dm:.1%} ± {ds:.1%}")
    print(f"Oracle                          : {om:.1%} ± {os:.1%}")
    print(f"Best gain vs v1.4.23 simple     : {best_mean-0.839:+.1%}")
    print(f"Best gap vs oracle              : {best_mean-om:+.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.23 simple mass-80 pooling       : 83.9% ± 1.7%")
    print("v1.4.23 decision-aware distillation  : 83.8% ± 1.7%")
    print("v1.4.21 oracle SELECTED vector       : 99.9% ± 0.6%")
    print()
    print("Interpretation")
    print("--------------")
    print(
        "This experiment removes vector reconstruction and directly learns "
        "a low-dimensional FIRST/SECOND decision representation from the "
        "adaptive semantic field."
    )
    print(
        "If a small dimension matches or exceeds larger dimensions, the "
        "task-critical relation semantics are concentrated in a compact "
        "decision subspace rather than requiring the full 256-D geometry."
    )
    print("Output dir:", out)


if __name__ == "__main__":
    main()
