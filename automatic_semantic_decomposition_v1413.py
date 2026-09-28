# automatic_semantic_decomposition_v1413.py
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_contrast_decomposition_v1412 import CONTRAST_SAMPLES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer
from relation_paraphrase_generalization_v145 import encode_text_stages

STAGE = "block5"
HIDDEN = 16
PROJ_DIM = 16
SUPCON_WEIGHT = 0.25
N_FOLDS = 5

ROLE_NEGATED = 0
ROLE_SELECTED = 1


class Projection(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, HIDDEN),
            nn.GELU(),
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, PROJ_DIM),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(PROJ_DIM, 2)

    def forward(self, x):
        return self.fc(x)


def supcon(z, y, temp):
    n = z.shape[0]
    sim = (z @ z.T) / temp
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same = y[:, None].eq(y[None, :]) & ~eye
    logits = sim.masked_fill(eye, float("-inf"))
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos = same.sum(1).clamp_min(1)
    return -(logp.masked_fill(~same, 0.0).sum(1) / pos).mean()


def fit_standardizer(X):
    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return mean, std


def standardize(X, mean, std):
    return (X - mean) / std


def train_model(X, y, args, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    proj = Projection(X.shape[1]).to(X.device)
    head = Head().to(X.device)
    opt = torch.optim.AdamW(
        list(proj.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for _ in range(args.epochs):
        opt.zero_grad()
        z = proj(X)
        loss = (
            F.cross_entropy(head(z), y)
            + SUPCON_WEIGHT * supcon(z, y, args.temperature)
        )
        loss.backward()
        opt.step()

    proj.eval()
    head.eval()
    return proj, head


@torch.no_grad()
def predict_probs(proj, head, X):
    return torch.softmax(head(proj(X)), dim=-1)


def build_folds():
    folds = [[] for _ in range(N_FOLDS)]
    for label in (0, 1):
        idx = [
            i for i, s in enumerate(CONTRAST_SAMPLES)
            if s["selected_position"] == label
        ]
        assert len(idx) == 20
        for k in range(N_FOLDS):
            folds[k].extend(idx[k * 4:(k + 1) * 4])
    return folds


def auto_split_clauses(text: str):
    """Syntactic candidate extraction only; no gold spans or labels are used."""
    text = text.strip()

    # First use explicit punctuation boundaries.
    parts = [
        p.strip(" 、，,。；;")
        for p in re.split(r"[、，,。；;]+", text)
        if p.strip(" 、，,。；;")
    ]

    # If punctuation gives no decomposition, expose a common contrast boundary.
    # This is a surface syntax rule, not a semantic FIRST/SECOND rule.
    if len(parts) == 1 and "ではなく" in text:
        left, right = text.split("ではなく", 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            parts = [left + "ではない", right]

    # A few phrases use connective morphology without punctuation.
    if len(parts) == 1:
        for marker in ("を除くと", "を外した結果", "を退けて"):
            if marker in text:
                left, right = text.split(marker, 1)
                left = left.strip()
                right = right.strip()
                if left and right:
                    parts = [left + marker.rstrip("とて"), right]
                    break

    # Keep at most four non-empty candidates and remove exact duplicates.
    out = []
    for p in parts:
        if p and p not in out:
            out.append(p)
    return out[:4] if out else [text]


def encode_cached(model, tok, cache, text):
    if text not in cache:
        cache[text] = encode_text_stages(model, tok, text)[STAGE]
    return cache[text]


def train_span_models(
    Xneg_train, Xsel_train, yneg_train, ysel_train, args, seed
):
    # Role model: gold training spans only.
    Xrole = torch.cat([Xneg_train, Xsel_train], dim=0)
    yrole = torch.cat([
        torch.full(
            (len(Xneg_train),), ROLE_NEGATED,
            dtype=torch.long, device=Xrole.device
        ),
        torch.full(
            (len(Xsel_train),), ROLE_SELECTED,
            dtype=torch.long, device=Xrole.device
        ),
    ])

    mr, sr = fit_standardizer(Xrole)
    Xrole_std = standardize(Xrole, mr, sr)
    role_proj, role_head = train_model(
        Xrole_std, yrole, args, seed + 100
    )

    # Shared position model: predicts FIRST/SECOND from either semantic span.
    Xpos = torch.cat([Xneg_train, Xsel_train], dim=0)
    ypos = torch.cat([yneg_train, ysel_train], dim=0)
    mp, sp = fit_standardizer(Xpos)
    Xpos_std = standardize(Xpos, mp, sp)
    pos_proj, pos_head = train_model(
        Xpos_std, ypos, args, seed + 200
    )

    return {
        "role": (role_proj, role_head, mr, sr),
        "position": (pos_proj, pos_head, mp, sp),
    }


def infer_auto_phrase(model_pack, clause_vectors):
    role_proj, role_head, mr, sr = model_pack["role"]
    pos_proj, pos_head, mp, sp = model_pack["position"]

    X = torch.stack(clause_vectors).to(mr.device)
    role_p = predict_probs(
        role_proj, role_head, standardize(X, mr, sr)
    ).cpu()
    pos_p = predict_probs(
        pos_proj, pos_head, standardize(X, mp, sp)
    ).cpu()

    n = len(clause_vectors)

    if n == 1:
        selected_idx = 0
        negated_idx = None
    else:
        # Globally assign distinct NEGATED and SELECTED clauses.
        best_score = -1.0
        negated_idx = 0
        selected_idx = 1
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                score = float(role_p[i, ROLE_NEGATED] * role_p[j, ROLE_SELECTED])
                if score > best_score:
                    best_score = score
                    negated_idx = i
                    selected_idx = j

    selected_prob = pos_p[selected_idx].clone()

    if negated_idx is not None:
        neg_prob = pos_p[negated_idx]
        # Deterministic binder: selected should be opposite negated.
        fused = selected_prob.clone()
        fused[0] += neg_prob[1]
        fused[1] += neg_prob[0]
        pred = int(fused.argmax())
        role_consistent = (
            int(role_p[selected_idx].argmax()) == ROLE_SELECTED
            and int(role_p[negated_idx].argmax()) == ROLE_NEGATED
        )
        positional_consistent = (
            int(selected_prob.argmax()) == 1 - int(neg_prob.argmax())
        )
    else:
        pred = int(selected_prob.argmax())
        role_consistent = (
            int(role_p[selected_idx].argmax()) == ROLE_SELECTED
        )
        positional_consistent = True

    return pred, role_consistent, positional_consistent


def eval_direct(Xfull, ysel, args, seed):
    pred = torch.zeros(len(ysel), dtype=torch.long)
    for k, test in enumerate(build_folds()):
        ts = set(test)
        train = [i for i in range(len(ysel)) if i not in ts]
        m, s = fit_standardizer(Xfull[train])
        Xtr = standardize(Xfull[train], m, s)
        Xte = standardize(Xfull[test], m, s)
        proj, head = train_model(
            Xtr, ysel[train], args, seed + 1000 * k
        )
        pred[test] = predict_probs(proj, head, Xte).argmax(-1).cpu()
    return float((pred == ysel.cpu()).float().mean())


def eval_oracle(
    Xneg, Xsel, yneg, ysel, args, seed
):
    pred = torch.zeros(len(ysel), dtype=torch.long)

    for k, test in enumerate(build_folds()):
        ts = set(test)
        train = [i for i in range(len(ysel)) if i not in ts]

        pack = train_span_models(
            Xneg[train], Xsel[train],
            yneg[train], ysel[train],
            args, seed + 10000 + 1000 * k
        )

        _, _, mp, sp = pack["position"]
        pos_proj, pos_head, _, _ = pack["position"]

        pneg = predict_probs(
            pos_proj, pos_head, standardize(Xneg[test], mp, sp)
        ).cpu()
        psel = predict_probs(
            pos_proj, pos_head, standardize(Xsel[test], mp, sp)
        ).cpu()

        fused = psel.clone()
        fused[:, 0] += pneg[:, 1]
        fused[:, 1] += pneg[:, 0]
        pred[test] = fused.argmax(-1)

    return float((pred == ysel.cpu()).float().mean())


def eval_auto(
    model, tok, gold_cache, clause_cache,
    Xneg, Xsel, yneg, ysel, args, seed
):
    pred = torch.zeros(len(ysel), dtype=torch.long)
    role_ok = []
    pos_ok = []
    clause_counts = []

    for k, test in enumerate(build_folds()):
        ts = set(test)
        train = [i for i in range(len(ysel)) if i not in ts]

        pack = train_span_models(
            Xneg[train], Xsel[train],
            yneg[train], ysel[train],
            args, seed + 20000 + 1000 * k
        )

        for idx in test:
            text = CONTRAST_SAMPLES[idx]["text"]
            clauses = auto_split_clauses(text)
            clause_counts.append(len(clauses))

            vectors = [
                encode_cached(model, tok, clause_cache, c)
                for c in clauses
            ]
            p, rok, pok = infer_auto_phrase(pack, vectors)
            pred[idx] = p
            role_ok.append(rok)
            pos_ok.append(pok)

    acc = float((pred == ysel.cpu()).float().mean())
    role_consistency = sum(role_ok) / len(role_ok)
    position_consistency = sum(pos_ok) / len(pos_ok)
    multi_clause_rate = sum(c >= 2 for c in clause_counts) / len(clause_counts)
    mean_clause_count = statistics.mean(clause_counts)

    return (
        acc, role_consistency, position_consistency,
        multi_clause_rate, mean_clause_count
    )


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--seeds", default="1-20")
    ap.add_argument(
        "--output-dir",
        default="results/automatic_semantic_decomposition_v1413",
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
    print(" LLM_GPU v1.4.13 Automatic Semantic Decomposition")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            :", STAGE)
    print("Contrast samples : 40")
    print("Training spans   : gold negated/selected spans from train folds only")
    print("Test spans       : automatic clause extraction from raw phrase")
    print("Automatic route  : clause split -> role/position decode -> binder")
    print("Evaluation       : 5-fold unseen-paraphrase holdout")
    print("Relation seeds   :", f"{seeds[0]}..{seeds[-1]}")
    print("Base model       : completely frozen")
    print()

    gold_cache = {}
    clause_cache = {}

    print("Encoding full phrases and gold training spans...")
    Xfull = torch.stack([
        encode_cached(model, tok, gold_cache, s["text"])
        for s in CONTRAST_SAMPLES
    ]).to(device)
    Xneg = torch.stack([
        encode_cached(model, tok, gold_cache, s["negated_text"])
        for s in CONTRAST_SAMPLES
    ]).to(device)
    Xsel = torch.stack([
        encode_cached(model, tok, gold_cache, s["selected_text"])
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

    # Show deterministic splitter coverage once; independent of seed.
    split_counts = [len(auto_split_clauses(s["text"])) for s in CONTRAST_SAMPLES]
    print(
        "Automatic split  : "
        f"{sum(c >= 2 for c in split_counts)}/{len(split_counts)} "
        "phrases produce >=2 clauses"
    )
    print(
        "Mean clauses     : "
        f"{statistics.mean(split_counts):.2f}"
    )

    rows = []
    direct_scores = []
    oracle_scores = []
    auto_scores = []
    role_consistency_scores = []
    pos_consistency_scores = []

    print()
    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(
            Xneg, Xsel, yneg, ysel, args, seed
        )
        (
            auto, role_c, pos_c,
            multi_clause_rate, mean_clause_count
        ) = eval_auto(
            model, tok, gold_cache, clause_cache,
            Xneg, Xsel, yneg, ysel, args, seed
        )

        direct_scores.append(direct)
        oracle_scores.append(oracle)
        auto_scores.append(auto)
        role_consistency_scores.append(role_c)
        pos_consistency_scores.append(pos_c)

        rows.append({
            "seed": seed,
            "direct_accuracy": direct,
            "oracle_decomposition_accuracy": oracle,
            "automatic_decomposition_accuracy": auto,
            "auto_role_consistency": role_c,
            "auto_position_consistency": pos_c,
            "multi_clause_rate": multi_clause_rate,
            "mean_clause_count": mean_clause_count,
            "auto_gain_vs_direct": auto - direct,
            "auto_gap_vs_oracle": auto - oracle,
        })

        print(
            f"seed={seed:>3d}  direct={direct:>6.1%}  "
            f"oracle={oracle:>6.1%}  auto={auto:>6.1%}  "
            f"roleC={role_c:>6.1%}  posC={pos_c:>6.1%}  "
            f"gain={auto-direct:+.1%}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed", "direct_accuracy",
        "oracle_decomposition_accuracy",
        "automatic_decomposition_accuracy",
        "auto_role_consistency",
        "auto_position_consistency",
        "multi_clause_rate",
        "mean_clause_count",
        "auto_gain_vs_direct",
        "auto_gap_vs_oracle",
    ]
    with (out / "per_seed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    metrics = [
        ("direct_accuracy", direct_scores),
        ("oracle_decomposition_accuracy", oracle_scores),
        ("automatic_decomposition_accuracy", auto_scores),
        ("auto_role_consistency", role_consistency_scores),
        ("auto_position_consistency", pos_consistency_scores),
        (
            "auto_gain_vs_direct",
            [a - d for a, d in zip(auto_scores, direct_scores)]
        ),
        (
            "auto_gap_vs_oracle",
            [a - o for a, o in zip(auto_scores, oracle_scores)]
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
    print("-" * 88)
    for label, vals in [
        ("Direct full-phrase baseline", direct_scores),
        ("Oracle gold-span decomposition", oracle_scores),
        ("Automatic decomposition", auto_scores),
        ("Automatic role consistency", role_consistency_scores),
        ("Automatic position consistency", pos_consistency_scores),
    ]:
        m, s = mean_std(vals)
        print(
            f"{label:34s}: {m:.1%} ± {s:.1%} "
            f"[{min(vals):.1%}, {max(vals):.1%}]"
        )

    gain = [a - d for a, d in zip(auto_scores, direct_scores)]
    gap = [a - o for a, o in zip(auto_scores, oracle_scores)]
    gm, gs = mean_std(gain)
    xm, xs = mean_std(gap)

    print(f"{'Auto gain vs direct':34s}: {gm:+.1%} ± {gs:.1%}")
    print(f"{'Auto gap vs oracle':34s}: {xm:+.1%} ± {xs:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.12 direct full phrase           : 61.0% ± 4.8%")
    print("v1.4.12 oracle decomposed + binder   : 99.9% ± 0.6%")
    print()
    print("Interpretation")
    print("--------------")
    print("Gold spans are used only as supervision inside each training fold.")
    print("At test time, the system receives only the raw contrast phrase.")
    print("It automatically extracts clause candidates, predicts semantic roles")
    print("and FIRST/SECOND position, then applies the deterministic binder.")
    print("The remaining gap to the oracle quantifies decomposition error itself.")
    print("Output dir:", out)


if __name__ == "__main__":
    main()
