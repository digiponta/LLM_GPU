# learned_semantic_span_detector_v1414.py
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

STAGE_INDEX = 5
N_FOLDS = 5
OTHER = 0
NEGATED = 1
SELECTED = 2

FIRST_SURFACES = [
    "前側の概念","前の概念","前の候補","先の候補","前者",
    "前側","先の","前",
]
SECOND_SURFACES = [
    "後ろ側の概念","後ろの概念","後ろの候補","後の候補","後者",
    "後ろ側","後の","後ろ",
]


class SpanTagger(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)


class PositionHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


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


def find_surface_spans(text: str, surfaces):
    spans = []
    occupied = [False] * len(text)
    for surface in sorted(surfaces, key=len, reverse=True):
        start = 0
        while True:
            i = text.find(surface, start)
            if i < 0:
                break
            j = i + len(surface)
            if not any(occupied[i:j]):
                spans.append((i, j))
                for k in range(i, j):
                    occupied[k] = True
            start = i + 1
    return sorted(spans)


def gold_role_spans(sample):
    first = find_surface_spans(sample["text"], FIRST_SURFACES)
    second = find_surface_spans(sample["text"], SECOND_SURFACES)
    if sample["selected_position"] == 0:
        return second, first
    return first, second


@torch.no_grad()
def encode_token_states(model, tok, text):
    device = next(model.parameters()).device
    wrapper = f"人: {text}\nAI: "
    enc = tok.backend.encode(wrapper, add_special_tokens=False)
    ids = [tok.bos_id] + enc.ids
    offsets = [(0, 0)] + list(enc.offsets)

    tid = torch.tensor([ids], dtype=torch.long, device=device)
    x = model.embedding(tid)
    if model.position_embedding is not None:
        pos = torch.arange(tid.shape[1], device=device)
        x = x + model.position_embedding(pos).unsqueeze(0)

    for i, block in enumerate(model.blocks, start=1):
        x = block(x)
        if i == STAGE_INDEX:
            break

    prefix_len = len("人: ")
    phrase_start = prefix_len
    phrase_end = prefix_len + len(text)

    token_indices = []
    local_offsets = []
    for ti, (a, b) in enumerate(offsets):
        if b <= phrase_start or a >= phrase_end or a == b:
            continue
        token_indices.append(ti)
        local_offsets.append((
            max(0, a - phrase_start),
            min(len(text), b - phrase_start),
        ))

    states = x[0, token_indices].detach().cpu()
    return states, local_offsets


def token_labels(sample, offsets):
    neg_spans, sel_spans = gold_role_spans(sample)
    labels = []
    for a, b in offsets:
        lab = OTHER
        if any(a < sj and b > si for si, sj in neg_spans):
            lab = NEGATED
        if any(a < sj and b > si for si, sj in sel_spans):
            lab = SELECTED
        labels.append(lab)
    return torch.tensor(labels, dtype=torch.long)


def pooled_gold_role_vector(states, labels, role):
    mask = labels == role
    if not bool(mask.any()):
        return None
    return states[mask].mean(0)


def train_span_tagger(train_states, train_labels, args, seed, device):
    X = torch.cat(train_states, dim=0).to(device)
    y = torch.cat(train_labels, dim=0).to(device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tagger = SpanTagger(X.shape[1], args.span_hidden).to(device)
    counts = torch.bincount(y, minlength=3).float().clamp_min(1)
    weights = counts.sum() / counts
    weights = weights / weights.mean()

    opt = torch.optim.AdamW(
        tagger.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    for _ in range(args.epochs):
        opt.zero_grad()
        logits = tagger(X)
        loss = F.cross_entropy(logits, y, weight=weights)
        loss.backward()
        opt.step()

    tagger.eval()
    return tagger


def train_position_head(vectors, labels, args, seed, device):
    X = torch.stack(vectors).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    Xs = (X - mean) / std

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head = PositionHead(X.shape[1], args.position_hidden).to(device)
    opt = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = F.cross_entropy(head(Xs), y)
        loss.backward()
        opt.step()

    head.eval()
    return head, mean, std


@torch.no_grad()
def learned_infer(tagger, pos_head, mean, std, states, args, device):
    X = states.to(device)
    tag_prob = torch.softmax(tagger(X), dim=-1)

    selected_w = tag_prob[:, SELECTED]
    negated_w = tag_prob[:, NEGATED]

    selected_vec = (
        (selected_w[:, None] * X).sum(0)
        / selected_w.sum().clamp_min(1e-6)
    )
    negated_vec = (
        (negated_w[:, None] * X).sum(0)
        / negated_w.sum().clamp_min(1e-6)
    )

    selected_pos = torch.softmax(
        pos_head(((selected_vec[None, :] - mean) / std)), dim=-1
    )[0]
    negated_pos = torch.softmax(
        pos_head(((negated_vec[None, :] - mean) / std)), dim=-1
    )[0]

    selected_conf = float(selected_w.max())
    negated_conf = float(negated_w.max())

    fused = selected_pos.clone()
    if negated_conf >= args.negated_conf_threshold:
        fused[0] += negated_pos[1]
        fused[1] += negated_pos[0]

    pred = int(fused.argmax())
    hard = tag_prob.argmax(-1)
    has_selected = bool((hard == SELECTED).any())
    has_negated = bool((hard == NEGATED).any())

    return pred, has_selected, has_negated, selected_conf, negated_conf


def eval_learned(states_all, labels_all, args, seed, device):
    pred = torch.zeros(len(CONTRAST_SAMPLES), dtype=torch.long)
    detected_selected = []
    detected_negated = []
    selected_confidences = []
    negated_confidences = []

    for fold, test in enumerate(build_folds()):
        test_set = set(test)
        train = [
            i for i in range(len(CONTRAST_SAMPLES))
            if i not in test_set
        ]

        tagger = train_span_tagger(
            [states_all[i] for i in train],
            [labels_all[i] for i in train],
            args, seed + 1000 * fold, device,
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

        pos_head, mean, std = train_position_head(
            position_vectors, position_labels, args,
            seed + 10000 + 1000 * fold, device,
        )

        for i in test:
            p, has_sel, has_neg, sc, nc = learned_infer(
                tagger, pos_head, mean, std,
                states_all[i], args, device
            )
            pred[i] = p
            detected_selected.append(has_sel)
            detected_negated.append(has_neg)
            selected_confidences.append(sc)
            negated_confidences.append(nc)

    gold = torch.tensor(
        [s["selected_position"] for s in CONTRAST_SAMPLES],
        dtype=torch.long,
    )
    accuracy = float((pred == gold).float().mean())

    return {
        "accuracy": accuracy,
        "selected_detection": sum(detected_selected) / len(detected_selected),
        "negated_detection": sum(detected_negated) / len(detected_negated),
        "selected_confidence": statistics.mean(selected_confidences),
        "negated_confidence": statistics.mean(negated_confidences),
    }


def mean_std(values):
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.mean(values), statistics.stdev(values)


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
    ap.add_argument("--negated-conf-threshold", type=float, default=0.45)
    ap.add_argument(
        "--output-dir",
        default="results/learned_semantic_span_detector_v1414",
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
    print(" LLM_GPU v1.4.14 Learned Semantic Span Detector")
    print("=" * 118)
    print("Device           :", device)
    print("Base             :", args.model)
    print("Stage            : block5 token states")
    print("Span labels      : OTHER / NEGATED / SELECTED")
    print("Span detector    :", f"256 -> {args.span_hidden} -> 3")
    print("Position decoder :", f"256 -> {args.position_hidden} -> 2")
    print("Test input       : raw phrase only")
    print("Evaluation       : 5-fold unseen-paraphrase holdout")
    print("Relation seeds   :", f"{seeds[0]}..{seeds[-1]}")
    print("Base model       : completely frozen")
    print()

    print("Encoding block5 token states with tokenizer offset mapping...")
    states_all = []
    labels_all = []
    offsets_all = []

    for sample in CONTRAST_SAMPLES:
        states, offsets = encode_token_states(
            model, tok, sample["text"]
        )
        labels = token_labels(sample, offsets)
        states_all.append(states)
        labels_all.append(labels)
        offsets_all.append(offsets)

    token_total = sum(len(x) for x in labels_all)
    neg_total = sum(int((x == NEGATED).sum()) for x in labels_all)
    sel_total = sum(int((x == SELECTED).sum()) for x in labels_all)
    gold_neg_phrases = sum(
        bool((x == NEGATED).any()) for x in labels_all
    )
    gold_sel_phrases = sum(
        bool((x == SELECTED).any()) for x in labels_all
    )

    print("Token labels      :", token_total)
    print("NEGATED tokens    :", neg_total)
    print("SELECTED tokens   :", sel_total)
    print(
        "Gold span phrases : "
        f"NEGATED {gold_neg_phrases}/40, "
        f"SELECTED {gold_sel_phrases}/40"
    )
    print()

    # Full phrase vectors and oracle normalized-span vectors for matched reference.
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
    learned_scores = []
    sel_detect_scores = []
    neg_detect_scores = []

    print("Per-seed results")
    print("-" * 118)

    for seed in seeds:
        direct = eval_direct(Xfull, ysel, args, seed)
        oracle = eval_oracle(
            Xneg, Xsel, yneg, ysel, args, seed
        )
        learned = eval_learned(
            states_all, labels_all, args, seed, device
        )

        direct_scores.append(direct)
        oracle_scores.append(oracle)
        learned_scores.append(learned["accuracy"])
        sel_detect_scores.append(learned["selected_detection"])
        neg_detect_scores.append(learned["negated_detection"])

        rows.append({
            "seed": seed,
            "direct_accuracy": direct,
            "oracle_accuracy": oracle,
            "learned_accuracy": learned["accuracy"],
            "selected_detection": learned["selected_detection"],
            "negated_detection": learned["negated_detection"],
            "selected_confidence": learned["selected_confidence"],
            "negated_confidence": learned["negated_confidence"],
            "gain_vs_direct": learned["accuracy"] - direct,
            "gap_vs_oracle": learned["accuracy"] - oracle,
        })

        print(
            f"seed={seed:>3d}  direct={direct:>6.1%}  "
            f"oracle={oracle:>6.1%}  learned={learned['accuracy']:>6.1%}  "
            f"selDet={learned['selected_detection']:>6.1%}  "
            f"negDet={learned['negated_detection']:>6.1%}  "
            f"gain={learned['accuracy']-direct:+.1%}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed","direct_accuracy","oracle_accuracy",
        "learned_accuracy","selected_detection",
        "negated_detection","selected_confidence",
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
        ("learned_accuracy", learned_scores),
        ("selected_detection", sel_detect_scores),
        ("negated_detection", neg_detect_scores),
        (
            "gain_vs_direct",
            [a - d for a, d in zip(learned_scores, direct_scores)]
        ),
        (
            "gap_vs_oracle",
            [a - o for a, o in zip(learned_scores, oracle_scores)]
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
        ("Learned span decomposition", learned_scores),
        ("SELECTED span detection", sel_detect_scores),
        ("NEGATED span detection", neg_detect_scores),
    ]:
        m, s = mean_std(vals)
        print(
            f"{label:34s}: {m:.1%} ± {s:.1%} "
            f"[{min(vals):.1%}, {max(vals):.1%}]"
        )

    gain = [a - d for a, d in zip(learned_scores, direct_scores)]
    gap = [a - o for a, o in zip(learned_scores, oracle_scores)]
    gm, gs = mean_std(gain)
    xm, xs = mean_std(gap)

    print(f"{'Learned gain vs direct':34s}: {gm:+.1%} ± {gs:.1%}")
    print(f"{'Learned gap vs oracle':34s}: {xm:+.1%} ± {xs:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.13 rule-based automatic         : 90.4% ± 3.6%")
    print("v1.4.13 role consistency             : 69.4% ± 4.9%")
    print("v1.4.13 oracle                       : 99.8% ± 0.8%")
    print()
    print("Interpretation")
    print("--------------")
    print("The rule-based clause splitter is replaced by a learned token-level")
    print("semantic span detector operating on frozen block5 hidden states.")
    print("Gold role spans are used only in training folds. At test time, only the")
    print("raw phrase is available; detected role spans are pooled, position-decoded,")
    print("and combined by the deterministic semantic binder.")
    print("Output dir:", out)


if __name__ == "__main__":
    main()
