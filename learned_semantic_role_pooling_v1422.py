# learned_semantic_role_pooling_v1422.py
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


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


class RoleConditionedPooler(nn.Module):
    """
    Learned semantic pooling over contextual token states.

    Inputs:
      token hidden state h_t (256)
      learned role probability p_t (1)

    The pooler learns attention logits from [h_t, p_t], then projects the
    weighted contextual summary back into the original semantic space.
    """
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model + 1, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, states, role_score, mask=None):
        # states: [T,D], role_score: [T]
        x = torch.cat([states, role_score[:, None]], dim=-1)
        logits = self.attn(x).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        # Role prior stabilizes the attention toward the semantic field.
        logits = logits + torch.log(role_score.clamp_min(1e-6))
        weights = torch.softmax(logits, dim=0)
        pooled = (weights[:, None] * states).sum(0)

        gate = self.gate(
            torch.cat([pooled, role_score.mean()[None]], dim=0)
        ).squeeze()
        refined = self.proj(pooled)
        out = gate * refined + (1.0 - gate) * pooled
        return out, weights


def mass_mask(score, target_mass):
    n = len(score)
    anchor = int(score.argmax())
    total = float(score.sum())
    mask = torch.zeros(n, dtype=torch.bool, device=score.device)

    if total <= 1e-8:
        mask[anchor] = True
        return mask

    lo = hi = anchor
    current = float(score[anchor])
    mask[anchor] = True

    while current / total < target_mass and (lo > 0 or hi < n - 1):
        left_score = float(score[lo - 1]) if lo > 0 else -1.0
        right_score = float(score[hi + 1]) if hi < n - 1 else -1.0

        if right_score > left_score:
            hi += 1
            current += float(score[hi])
            mask[hi] = True
        else:
            lo -= 1
            current += float(score[lo])
            mask[lo] = True

    return mask


def semantic_losses(pred, teacher):
    pred_n = F.normalize(pred, dim=0)
    teach_n = F.normalize(teacher, dim=0)
    cosine = 1.0 - torch.sum(pred_n * teach_n)
    mse = F.mse_loss(pred, teacher)
    return cosine, mse


def train_pooler(
    states_all,
    labels_all,
    tagger,
    role,
    train,
    pos_head,
    pmean,
    pstd,
    args,
    seed,
    device,
):
    usable = [
        i for i in train
        if pooled_gold_role_vector(
            states_all[i], labels_all[i], role
        ) is not None
    ]
    if not usable:
        raise RuntimeError("No usable role vectors for pooler training")

    d_model = states_all[usable[0]].shape[1]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pooler = RoleConditionedPooler(
        d_model=d_model,
        hidden=args.pooler_hidden,
    ).to(device)

    opt = torch.optim.AdamW(
        pooler.parameters(),
        lr=args.pooler_lr,
        weight_decay=args.pooler_weight_decay,
    )

    # Position decoder is a fixed teacher head during pooler distillation.
    for p in pos_head.parameters():
        p.requires_grad_(False)

    for _ in range(args.pooler_epochs):
        opt.zero_grad()
        losses = []

        for i in usable:
            states = states_all[i].to(device)

            with torch.no_grad():
                score = role_scores(
                    states_all[i], tagger, device
                )
                mask = mass_mask(score, MASS)
                teacher = pooled_gold_role_vector(
                    states_all[i], labels_all[i], role
                ).to(device)

            pred, _ = pooler(states, score, mask)

            cos_loss, mse_loss = semantic_losses(pred, teacher)

            if role == SELECTED:
                y = CONTRAST_SAMPLES[i]["selected_position"]
            else:
                y = CONTRAST_SAMPLES[i]["negated_position"]

            logits = pos_head(
                (pred[None, :] - pmean) / pstd
            )
            pos_loss = F.cross_entropy(
                logits,
                torch.tensor([y], device=device),
            )

            loss = (
                args.cosine_weight * cos_loss
                + args.mse_weight * mse_loss
                + args.position_weight * pos_loss
            )
            losses.append(loss)

        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

    pooler.eval()

    # Restore only semantic immutability expectation; head is not optimized here.
    for p in pos_head.parameters():
        p.requires_grad_(True)

    return pooler


@torch.no_grad()
def learned_pool_vector(states, score, pooler, device):
    states = states.to(device)
    mask = mass_mask(score, MASS)
    vec, weights = pooler(states, score, mask)
    return vec, weights


def eval_seed(states_all, labels_all, args, seed, device):
    pred_simple = torch.zeros(
        len(CONTRAST_SAMPLES), dtype=torch.long
    )
    pred_distilled = torch.zeros(
        len(CONTRAST_SAMPLES), dtype=torch.long
    )
    pred_oracle_selected = torch.zeros(
        len(CONTRAST_SAMPLES), dtype=torch.long
    )

    selected_cosines = []
    negated_cosines = []

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

        position_vectors, position_labels = [], []
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

        sel_pooler = train_pooler(
            states_all,
            labels_all,
            sel_tagger,
            SELECTED,
            train,
            pos_head,
            pmean,
            pstd,
            args,
            seed + 40000 + 1000 * fold,
            device,
        )
        neg_pooler = train_pooler(
            states_all,
            labels_all,
            neg_tagger,
            NEGATED,
            train,
            pos_head,
            pmean,
            pstd,
            args,
            seed + 50000 + 1000 * fold,
            device,
        )

        for i in test:
            states = states_all[i]
            labels = labels_all[i]

            sel_score = role_scores(
                states, sel_tagger, device
            )
            neg_score = role_scores(
                states, neg_tagger, device
            )
            exp_hat, _ = predict_explicit(
                states,
                explicit_head,
                emean,
                estd,
                device,
            )

            # v1.4.20 baseline: simple role-probability pooling
            sel_simple, _, _ = mass_window(
                states, sel_score, MASS
            )
            fused_simple = position_probs(
                sel_simple, pos_head, pmean, pstd
            ).clone()

            # v1.4.22 learned semantic pooling
            sel_vec, _ = learned_pool_vector(
                states, sel_score, sel_pooler, device
            )
            fused_distilled = position_probs(
                sel_vec, pos_head, pmean, pstd
            ).clone()

            # v1.4.21 oracle-selected reference in same fold
            sel_oracle = pooled_gold_role_vector(
                states, labels, SELECTED
            ).to(device)
            fused_oracle_sel = position_probs(
                sel_oracle, pos_head, pmean, pstd
            ).clone()

            selected_cosines.append(
                float(
                    F.cosine_similarity(
                        sel_vec[None, :],
                        sel_oracle[None, :],
                    )
                )
            )

            if exp_hat == 1:
                neg_simple, _, _ = mass_window(
                    states, neg_score, MASS
                )
                np_simple = position_probs(
                    neg_simple, pos_head, pmean, pstd
                )

                neg_vec, _ = learned_pool_vector(
                    states, neg_score, neg_pooler, device
                )
                np_distilled = position_probs(
                    neg_vec, pos_head, pmean, pstd
                )

                gold_neg = pooled_gold_role_vector(
                    states, labels, NEGATED
                )

                if gold_neg is not None:
                    gold_neg = gold_neg.to(device)
                    np_oracle = position_probs(
                        gold_neg, pos_head, pmean, pstd
                    )
                    negated_cosines.append(
                        float(
                            F.cosine_similarity(
                                neg_vec[None, :],
                                gold_neg[None, :],
                            )
                        )
                    )
                else:
                    np_oracle = np_distilled

                fused_simple[0] += np_simple[1]
                fused_simple[1] += np_simple[0]

                fused_distilled[0] += np_distilled[1]
                fused_distilled[1] += np_distilled[0]

                fused_oracle_sel[0] += np_oracle[1]
                fused_oracle_sel[1] += np_oracle[0]

            pred_simple[i] = int(fused_simple.argmax())
            pred_distilled[i] = int(
                fused_distilled.argmax()
            )
            pred_oracle_selected[i] = int(
                fused_oracle_sel.argmax()
            )

    gold = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
    )

    return {
        "simple_accuracy": float(
            (pred_simple == gold).float().mean()
        ),
        "distilled_accuracy": float(
            (pred_distilled == gold).float().mean()
        ),
        "oracle_selected_accuracy": float(
            (pred_oracle_selected == gold).float().mean()
        ),
        "selected_cosine": statistics.mean(
            selected_cosines
        ),
        "negated_cosine": (
            statistics.mean(negated_cosines)
            if negated_cosines else 0.0
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

    ap.add_argument("--pooler-hidden", type=int, default=64)
    ap.add_argument("--pooler-epochs", type=int, default=300)
    ap.add_argument("--pooler-lr", type=float, default=1e-3)
    ap.add_argument(
        "--pooler-weight-decay", type=float, default=1e-3
    )
    ap.add_argument(
        "--cosine-weight", type=float, default=1.0
    )
    ap.add_argument(
        "--mse-weight", type=float, default=0.25
    )
    ap.add_argument(
        "--position-weight", type=float, default=0.50
    )
    ap.add_argument(
        "--output-dir",
        default="results/learned_semantic_role_pooling_v1422",
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
    print(" LLM_GPU v1.4.22 Learned Semantic Role Pooling / Oracle Vector Distillation")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Semantic field   : adaptive 80% role-score mass")
    print("Pooler           : role-conditioned attention + residual projection")
    print("Teacher          : gold contextual role vector")
    print(
        "Loss             : "
        f"{args.cosine_weight}*cos + "
        f"{args.mse_weight}*mse + "
        f"{args.position_weight}*position"
    )
    print("Evaluation       : 5-fold unseen-paraphrase holdout")
    print("Relation seeds   :", f"{seeds[0]}..{seeds[-1]}")
    print("Base model       : completely frozen")
    print()

    print("Encoding block5 token states...")
    states_all, labels_all = [], []
    for sample in CONTRAST_SAMPLES:
        states, offsets = encode_token_states(
            model, tok, sample["text"]
        )
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
        dtype=torch.long,
        device=device,
    )
    yneg = torch.tensor(
        [s["negated_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
        device=device,
    )

    direct_scores = []
    oracle_scores = []
    simple_scores = []
    distilled_scores = []
    oracle_selected_scores = []
    selected_cosines = []
    negated_cosines = []
    rows = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(
            Xfull, ysel, args, seed
        )
        oracle = eval_oracle(
            Xneg, Xsel, yneg, ysel, args, seed
        )
        result = eval_seed(
            states_all,
            labels_all,
            args,
            seed,
            device,
        )

        direct_scores.append(direct)
        oracle_scores.append(oracle)
        simple_scores.append(
            result["simple_accuracy"]
        )
        distilled_scores.append(
            result["distilled_accuracy"]
        )
        oracle_selected_scores.append(
            result["oracle_selected_accuracy"]
        )
        selected_cosines.append(
            result["selected_cosine"]
        )
        negated_cosines.append(
            result["negated_cosine"]
        )

        rows.append({
            "seed": seed,
            "direct_accuracy": direct,
            "simple_accuracy":
                result["simple_accuracy"],
            "distilled_accuracy":
                result["distilled_accuracy"],
            "oracle_selected_accuracy":
                result["oracle_selected_accuracy"],
            "full_oracle_accuracy": oracle,
            "selected_cosine":
                result["selected_cosine"],
            "negated_cosine":
                result["negated_cosine"],
            "gain_vs_simple":
                result["distilled_accuracy"]
                - result["simple_accuracy"],
            "gap_vs_oracle_selected":
                result["distilled_accuracy"]
                - result["oracle_selected_accuracy"],
        })

        print(
            f"seed={seed:>3d} "
            f"direct={direct:>6.1%} "
            f"simple={result['simple_accuracy']:>6.1%} "
            f"distilled={result['distilled_accuracy']:>6.1%} "
            f"oracleSel={result['oracle_selected_accuracy']:>6.1%} "
            f"oracle={oracle:>6.1%} "
            f"selCos={result['selected_cosine']:.4f} "
            f"negCos={result['negated_cosine']:.4f}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed",
        "direct_accuracy",
        "simple_accuracy",
        "distilled_accuracy",
        "oracle_selected_accuracy",
        "full_oracle_accuracy",
        "selected_cosine",
        "negated_cosine",
        "gain_vs_simple",
        "gap_vs_oracle_selected",
    ]
    with (out / "per_seed.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        w = csv.DictWriter(
            f, fieldnames=fields
        )
        w.writeheader()
        w.writerows(rows)

    metrics = [
        ("direct_accuracy", direct_scores),
        ("simple_accuracy", simple_scores),
        ("distilled_accuracy", distilled_scores),
        (
            "oracle_selected_accuracy",
            oracle_selected_scores,
        ),
        ("full_oracle_accuracy", oracle_scores),
        ("selected_cosine", selected_cosines),
        ("negated_cosine", negated_cosines),
        (
            "gain_vs_simple",
            [
                d - s
                for d, s in zip(
                    distilled_scores,
                    simple_scores,
                )
            ],
        ),
        (
            "gap_vs_oracle_selected",
            [
                d - o
                for d, o in zip(
                    distilled_scores,
                    oracle_selected_scores,
                )
            ],
        ),
    ]

    with (out / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        w = csv.writer(f)
        w.writerow(
            ["metric", "mean", "std", "min", "max"]
        )
        for name, vals in metrics:
            m, s = mean_std(vals)
            w.writerow(
                [name, m, s, min(vals), max(vals)]
            )

    print()
    print("Multi-seed summary")
    print("-" * 96)

    for label, vals, pct in [
        ("Direct full-phrase baseline", direct_scores, True),
        ("Simple mass-80 pooling", simple_scores, True),
        ("Learned distilled pooling", distilled_scores, True),
        (
            "Oracle SELECTED vector",
            oracle_selected_scores,
            True,
        ),
        ("Full oracle", oracle_scores, True),
        (
            "SELECTED vector cosine",
            selected_cosines,
            False,
        ),
        (
            "NEGATED vector cosine",
            negated_cosines,
            False,
        ),
    ]:
        m, s = mean_std(vals)
        if pct:
            print(
                f"{label:34s}: "
                f"{m:.1%} ± {s:.1%} "
                f"[{min(vals):.1%}, {max(vals):.1%}]"
            )
        else:
            print(
                f"{label:34s}: "
                f"{m:.4f} ± {s:.4f} "
                f"[{min(vals):.4f}, {max(vals):.4f}]"
            )

    gain = [
        d - s
        for d, s in zip(
            distilled_scores, simple_scores
        )
    ]
    gap = [
        d - o
        for d, o in zip(
            distilled_scores,
            oracle_selected_scores,
        )
    ]
    gm, gs = mean_std(gain)
    xm, xs = mean_std(gap)

    print(
        f"{'Distilled gain vs simple':34s}: "
        f"{gm:+.1%} ± {gs:.1%}"
    )
    print(
        f"{'Gap vs oracle SELECTED':34s}: "
        f"{xm:+.1%} ± {xs:.1%}"
    )

    print()
    print("Reference")
    print("---------")
    print(
        "v1.4.21 learned mass-80 baseline     "
        ": 83.9% ± 1.7%"
    )
    print(
        "v1.4.21 oracle SELECTED vector       "
        ": 99.9% ± 0.6%"
    )
    print(
        "v1.4.21 oracle NEGATED vector        "
        ": 96.4% ± 1.7%"
    )
    print(
        "v1.4.13 rule-based automatic         "
        ": 90.4% ± 3.6%"
    )

    print()
    print("Interpretation")
    print("--------------")
    print(
        "The learned pooler is trained only on training folds "
        "to reproduce gold contextual role vectors while preserving "
        "FIRST/SECOND position information."
    )
    print(
        "A substantial accuracy gain together with higher cosine "
        "similarity would confirm that semantic-vector synthesis, "
        "rather than localization, was the dominant residual bottleneck."
    )
    print("Output dir:", out)


if __name__ == "__main__":
    main()
