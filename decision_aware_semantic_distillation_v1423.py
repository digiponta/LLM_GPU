# decision_aware_semantic_distillation_v1423.py
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
from learned_semantic_role_pooling_v1422 import (
    RoleConditionedPooler,
    mass_mask,
    semantic_losses,
)

MASS = 0.80


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def position_logits(vec, head, mean, std):
    return head((vec[None, :] - mean) / std)[0]


def train_decision_pooler(
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
        if pooled_gold_role_vector(states_all[i], labels_all[i], role) is not None
    ]
    if not usable:
        raise RuntimeError("No usable samples for decision-aware pooler")

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

    for p in pos_head.parameters():
        p.requires_grad_(False)

    T = args.distill_temperature

    for _ in range(args.pooler_epochs):
        opt.zero_grad()
        losses = []

        for i in usable:
            states = states_all[i].to(device)

            with torch.no_grad():
                score = role_scores(states_all[i], tagger, device)
                mask = mass_mask(score, MASS)
                teacher = pooled_gold_role_vector(
                    states_all[i], labels_all[i], role
                ).to(device)
                teacher_logits = position_logits(
                    teacher, pos_head, pmean, pstd
                )
                teacher_margin = teacher_logits[1] - teacher_logits[0]

            student, _ = pooler(states, score, mask)
            student_logits = position_logits(
                student, pos_head, pmean, pstd
            )
            student_margin = student_logits[1] - student_logits[0]

            cos_loss, mse_loss = semantic_losses(student, teacher)

            teacher_prob = F.softmax(teacher_logits / T, dim=-1)
            student_logprob = F.log_softmax(student_logits / T, dim=-1)
            kl_loss = F.kl_div(
                student_logprob,
                teacher_prob,
                reduction="batchmean",
            ) * (T * T)

            logit_loss = F.mse_loss(student_logits, teacher_logits)
            margin_loss = F.mse_loss(student_margin, teacher_margin)

            y = (
                CONTRAST_SAMPLES[i]["selected_position"]
                if role == SELECTED
                else CONTRAST_SAMPLES[i]["negated_position"]
            )
            ce_loss = F.cross_entropy(
                student_logits[None, :],
                torch.tensor([y], device=device),
            )

            loss = (
                args.cosine_weight * cos_loss
                + args.mse_weight * mse_loss
                + args.kl_weight * kl_loss
                + args.logit_weight * logit_loss
                + args.margin_weight * margin_loss
                + args.position_weight * ce_loss
            )
            losses.append(loss)

        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

    pooler.eval()
    for p in pos_head.parameters():
        p.requires_grad_(True)

    return pooler


@torch.no_grad()
def pooled_student(states, score, pooler, device):
    states = states.to(device)
    mask = mass_mask(score, MASS)
    vec, weights = pooler(states, score, mask)
    return vec, weights


def eval_seed(states_all, labels_all, args, seed, device):
    n = len(CONTRAST_SAMPLES)
    pred_simple = torch.zeros(n, dtype=torch.long)
    pred_decision = torch.zeros(n, dtype=torch.long)
    pred_oracle_selected = torch.zeros(n, dtype=torch.long)

    sel_cos = []
    neg_cos = []
    sel_logit_mse = []
    neg_logit_mse = []
    sel_margin_abs = []
    neg_margin_abs = []
    sel_margin_sign = []
    neg_margin_sign = []

    for fold, test in enumerate(build_folds()):
        test_set = set(test)
        train = [i for i in range(n) if i not in test_set]

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
            position_vectors,
            position_labels,
            args,
            seed + 30000 + 1000 * fold,
            device,
        )

        sel_pooler = train_decision_pooler(
            states_all, labels_all, sel_tagger, SELECTED, train,
            pos_head, pmean, pstd, args,
            seed + 40000 + 1000 * fold, device,
        )
        neg_pooler = train_decision_pooler(
            states_all, labels_all, neg_tagger, NEGATED, train,
            pos_head, pmean, pstd, args,
            seed + 50000 + 1000 * fold, device,
        )

        for i in test:
            states = states_all[i]
            labels = labels_all[i]

            sel_score = role_scores(states, sel_tagger, device)
            neg_score = role_scores(states, neg_tagger, device)
            exp_hat, _ = predict_explicit(
                states, explicit_head, emean, estd, device
            )

            # Simple v1.4.20 baseline
            sel_simple, _, _ = mass_window(states, sel_score, MASS)
            fused_simple = position_probs(
                sel_simple, pos_head, pmean, pstd
            ).clone()

            # Decision-aware student
            sel_student, _ = pooled_student(
                states, sel_score, sel_pooler, device
            )
            fused_decision = position_probs(
                sel_student, pos_head, pmean, pstd
            ).clone()

            # Oracle SELECTED reference
            sel_teacher = pooled_gold_role_vector(
                states, labels, SELECTED
            ).to(device)
            fused_oracle_sel = position_probs(
                sel_teacher, pos_head, pmean, pstd
            ).clone()

            s_logits = position_logits(sel_student, pos_head, pmean, pstd)
            t_logits = position_logits(sel_teacher, pos_head, pmean, pstd)
            s_margin = float(s_logits[1] - s_logits[0])
            t_margin = float(t_logits[1] - t_logits[0])

            sel_cos.append(float(F.cosine_similarity(
                sel_student[None, :], sel_teacher[None, :]
            )))
            sel_logit_mse.append(float(F.mse_loss(s_logits, t_logits)))
            sel_margin_abs.append(abs(s_margin - t_margin))
            sel_margin_sign.append(
                int((s_margin >= 0) == (t_margin >= 0))
            )

            if exp_hat == 1:
                neg_simple, _, _ = mass_window(states, neg_score, MASS)
                np_simple = position_probs(
                    neg_simple, pos_head, pmean, pstd
                )

                neg_student, _ = pooled_student(
                    states, neg_score, neg_pooler, device
                )
                np_student = position_probs(
                    neg_student, pos_head, pmean, pstd
                )

                gold_neg = pooled_gold_role_vector(
                    states, labels, NEGATED
                )
                if gold_neg is not None:
                    neg_teacher = gold_neg.to(device)
                    np_oracle = position_probs(
                        neg_teacher, pos_head, pmean, pstd
                    )

                    ns_logits = position_logits(
                        neg_student, pos_head, pmean, pstd
                    )
                    nt_logits = position_logits(
                        neg_teacher, pos_head, pmean, pstd
                    )
                    ns_margin = float(ns_logits[1] - ns_logits[0])
                    nt_margin = float(nt_logits[1] - nt_logits[0])

                    neg_cos.append(float(F.cosine_similarity(
                        neg_student[None, :], neg_teacher[None, :]
                    )))
                    neg_logit_mse.append(float(F.mse_loss(
                        ns_logits, nt_logits
                    )))
                    neg_margin_abs.append(abs(ns_margin - nt_margin))
                    neg_margin_sign.append(
                        int((ns_margin >= 0) == (nt_margin >= 0))
                    )
                else:
                    np_oracle = np_student

                fused_simple[0] += np_simple[1]
                fused_simple[1] += np_simple[0]

                fused_decision[0] += np_student[1]
                fused_decision[1] += np_student[0]

                fused_oracle_sel[0] += np_oracle[1]
                fused_oracle_sel[1] += np_oracle[0]

            pred_simple[i] = int(fused_simple.argmax())
            pred_decision[i] = int(fused_decision.argmax())
            pred_oracle_selected[i] = int(fused_oracle_sel.argmax())

    gold = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
    )

    return {
        "simple_accuracy": float((pred_simple == gold).float().mean()),
        "decision_accuracy": float((pred_decision == gold).float().mean()),
        "oracle_selected_accuracy": float(
            (pred_oracle_selected == gold).float().mean()
        ),
        "selected_cosine": statistics.mean(sel_cos),
        "negated_cosine": statistics.mean(neg_cos) if neg_cos else 0.0,
        "selected_logit_mse": statistics.mean(sel_logit_mse),
        "negated_logit_mse": statistics.mean(neg_logit_mse) if neg_logit_mse else 0.0,
        "selected_margin_abs_error": statistics.mean(sel_margin_abs),
        "negated_margin_abs_error": statistics.mean(neg_margin_abs) if neg_margin_abs else 0.0,
        "selected_margin_sign_agreement": statistics.mean(sel_margin_sign),
        "negated_margin_sign_agreement": statistics.mean(neg_margin_sign) if neg_margin_sign else 0.0,
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
    ap.add_argument("--pooler-weight-decay", type=float, default=1e-3)

    ap.add_argument("--cosine-weight", type=float, default=0.25)
    ap.add_argument("--mse-weight", type=float, default=0.10)
    ap.add_argument("--kl-weight", type=float, default=1.00)
    ap.add_argument("--logit-weight", type=float, default=0.50)
    ap.add_argument("--margin-weight", type=float, default=1.00)
    ap.add_argument("--position-weight", type=float, default=1.00)
    ap.add_argument("--distill-temperature", type=float, default=2.0)

    ap.add_argument(
        "--output-dir",
        default="results/decision_aware_semantic_distillation_v1423",
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
    print(" LLM_GPU v1.4.23 Decision-Aware Semantic Distillation")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Semantic field   : adaptive 80% role-score mass")
    print("Pooler           : role-conditioned attention + residual projection")
    print("Teacher          : gold role vector -> frozen position decoder logits")
    print(
        "Loss             : "
        f"{args.cosine_weight}*cos + "
        f"{args.mse_weight}*vec_mse + "
        f"{args.kl_weight}*KL + "
        f"{args.logit_weight}*logit_mse + "
        f"{args.margin_weight}*margin + "
        f"{args.position_weight}*CE"
    )
    print("Temperature      :", args.distill_temperature)
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

    direct_scores, simple_scores, decision_scores = [], [], []
    oracle_selected_scores, oracle_scores = [], []
    sel_cos, neg_cos = [], []
    sel_msign, neg_msign = [], []
    sel_lmse, neg_lmse = [], []
    sel_merr, neg_merr = [], []
    rows = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(Xneg, Xsel, yneg, ysel, args, seed)
        r = eval_seed(states_all, labels_all, args, seed, device)

        direct_scores.append(direct)
        simple_scores.append(r["simple_accuracy"])
        decision_scores.append(r["decision_accuracy"])
        oracle_selected_scores.append(r["oracle_selected_accuracy"])
        oracle_scores.append(oracle)
        sel_cos.append(r["selected_cosine"])
        neg_cos.append(r["negated_cosine"])
        sel_lmse.append(r["selected_logit_mse"])
        neg_lmse.append(r["negated_logit_mse"])
        sel_merr.append(r["selected_margin_abs_error"])
        neg_merr.append(r["negated_margin_abs_error"])
        sel_msign.append(r["selected_margin_sign_agreement"])
        neg_msign.append(r["negated_margin_sign_agreement"])

        rows.append({
            "seed": seed,
            "direct_accuracy": direct,
            "simple_accuracy": r["simple_accuracy"],
            "decision_accuracy": r["decision_accuracy"],
            "oracle_selected_accuracy": r["oracle_selected_accuracy"],
            "full_oracle_accuracy": oracle,
            "selected_cosine": r["selected_cosine"],
            "negated_cosine": r["negated_cosine"],
            "selected_logit_mse": r["selected_logit_mse"],
            "negated_logit_mse": r["negated_logit_mse"],
            "selected_margin_abs_error": r["selected_margin_abs_error"],
            "negated_margin_abs_error": r["negated_margin_abs_error"],
            "selected_margin_sign_agreement": r["selected_margin_sign_agreement"],
            "negated_margin_sign_agreement": r["negated_margin_sign_agreement"],
            "gain_vs_simple": r["decision_accuracy"] - r["simple_accuracy"],
            "gap_vs_oracle_selected": r["decision_accuracy"] - r["oracle_selected_accuracy"],
        })

        print(
            f"seed={seed:>3d} "
            f"direct={direct:>6.1%} "
            f"simple={r['simple_accuracy']:>6.1%} "
            f"decision={r['decision_accuracy']:>6.1%} "
            f"oracleSel={r['oracle_selected_accuracy']:>6.1%} "
            f"selSign={r['selected_margin_sign_agreement']:>6.1%} "
            f"negSign={r['negated_margin_sign_agreement']:>6.1%} "
            f"selCos={r['selected_cosine']:.4f}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = list(rows[0].keys())
    with (out / "per_seed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    metrics = [
        ("direct_accuracy", direct_scores),
        ("simple_accuracy", simple_scores),
        ("decision_accuracy", decision_scores),
        ("oracle_selected_accuracy", oracle_selected_scores),
        ("full_oracle_accuracy", oracle_scores),
        ("selected_cosine", sel_cos),
        ("negated_cosine", neg_cos),
        ("selected_logit_mse", sel_lmse),
        ("negated_logit_mse", neg_lmse),
        ("selected_margin_abs_error", sel_merr),
        ("negated_margin_abs_error", neg_merr),
        ("selected_margin_sign_agreement", sel_msign),
        ("negated_margin_sign_agreement", neg_msign),
        (
            "gain_vs_simple",
            [d - s for d, s in zip(decision_scores, simple_scores)],
        ),
        (
            "gap_vs_oracle_selected",
            [d - o for d, o in zip(decision_scores, oracle_selected_scores)],
        ),
    ]

    with (out / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "min", "max"])
        for name, vals in metrics:
            m, s = mean_std(vals)
            w.writerow([name, m, s, min(vals), max(vals)])

    print()
    print("Multi-seed summary")
    print("-" * 96)

    for label, vals, pct in [
        ("Direct full-phrase baseline", direct_scores, True),
        ("Simple mass-80 pooling", simple_scores, True),
        ("Decision-aware distillation", decision_scores, True),
        ("Oracle SELECTED vector", oracle_selected_scores, True),
        ("Full oracle", oracle_scores, True),
        ("SELECTED margin sign agreement", sel_msign, True),
        ("NEGATED margin sign agreement", neg_msign, True),
        ("SELECTED vector cosine", sel_cos, False),
        ("NEGATED vector cosine", neg_cos, False),
        ("SELECTED logit MSE", sel_lmse, False),
        ("NEGATED logit MSE", neg_lmse, False),
        ("SELECTED margin abs error", sel_merr, False),
        ("NEGATED margin abs error", neg_merr, False),
    ]:
        m, s = mean_std(vals)
        if pct:
            print(
                f"{label:36s}: {m:.1%} ± {s:.1%} "
                f"[{min(vals):.1%}, {max(vals):.1%}]"
            )
        else:
            print(
                f"{label:36s}: {m:.4f} ± {s:.4f} "
                f"[{min(vals):.4f}, {max(vals):.4f}]"
            )

    gain = [d - s for d, s in zip(decision_scores, simple_scores)]
    gap = [d - o for d, o in zip(decision_scores, oracle_selected_scores)]
    gm, gs = mean_std(gain)
    xm, xs = mean_std(gap)

    print(f"{'Decision gain vs simple':36s}: {gm:+.1%} ± {gs:.1%}")
    print(f"{'Gap vs oracle SELECTED':36s}: {xm:+.1%} ± {xs:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.22 simple mass-80 pooling       : 83.9% ± 1.7%")
    print("v1.4.22 vector-distilled pooling     : 82.0% ± 2.5%")
    print("v1.4.22 SELECTED vector cosine       : 0.9358 ± 0.0033")
    print("v1.4.21 oracle SELECTED vector       : 99.9% ± 0.6%")
    print()
    print("Interpretation")
    print("--------------")
    print(
        "Decision-aware distillation explicitly matches teacher logits "
        "and FIRST/SECOND margin in addition to vector similarity."
    )
    print(
        "If margin-sign agreement and routing accuracy rise while vector cosine "
        "changes little, the task-critical information is concentrated in the "
        "position-decoder decision subspace rather than in global vector geometry."
    )
    print("Output dir:", out)


if __name__ == "__main__":
    main()
