# relational_semantic_representation_v139.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from semantic_symmetric_comparison_v137 import DATASET, CLASSES, PAIR_META
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
STAGES = ["embedding","block1","block2","block3","block4","block5","block6","final_norm"]

PAIR_KEYS = []
PAIR_TO_ID = {}
SAMPLES = []

# DATASET order is reciprocal: target=a then target=b for every PAIR_META row.
for pair_id, topic_id, a, b, pa, pb in PAIR_META:
    key = tuple(sorted((a,b), key=lambda x: CLASS_TO_ID[x]))
    if key not in PAIR_TO_ID:
        PAIR_TO_ID[key] = len(PAIR_KEYS)
        PAIR_KEYS.append(key)

    SAMPLES.append({
        "prompt": pa,
        "target": a,
        "contrast": b,
        "pair": key,
        "pair_id": PAIR_TO_ID[key],
        "direction": 0 if key == (a,b) else 1,
    })
    SAMPLES.append({
        "prompt": pb,
        "target": b,
        "contrast": a,
        "pair": key,
        "pair_id": PAIR_TO_ID[key],
        "direction": 0 if key == (b,a) else 1,
    })

assert len(SAMPLES) == 60
assert len(PAIR_KEYS) == 15


class LinearHead(nn.Module):
    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_out)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def extract_stages(model: LanguageModel, tok: Tokenizer, prompt: str):
    device = next(model.parameters()).device
    ids = tok.encode(f"人: {prompt}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)

    x = model.embedding(token_ids)
    if model.position_embedding is not None:
        pos = torch.arange(token_ids.shape[1], device=device)
        x = x + model.position_embedding(pos).unsqueeze(0)

    out = {"embedding": x[:, -1, :][0].detach().cpu()}
    for i, block in enumerate(model.blocks, start=1):
        x = block(x)
        out[f"block{i}"] = x[:, -1, :][0].detach().cpu()

    x = model.final_norm(x)
    out["final_norm"] = x[:, -1, :][0].detach().cpu()
    return out


def standardize(Xtr, Xte):
    mean = Xtr.mean(0, keepdim=True)
    std = Xtr.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std, (Xte-mean)/std


def train_head(X, y, n_out, epochs, lr, wd, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    m = LinearHead(X.shape[1], n_out).to(X.device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        loss = loss_fn(m(X), y)
        loss.backward()
        opt.step()

    m.eval()
    return m


def build_folds():
    # Each class has 10 target examples. Keep 2/class/fold.
    folds = [[] for _ in range(5)]
    for c in CLASSES:
        idx = [i for i,s in enumerate(SAMPLES) if s["target"] == c]
        assert len(idx) == 10
        for k in range(5):
            folds[k].extend(idx[2*k:2*k+2])
    return folds


def oof_predict(X, y, n_out, folds, epochs, lr, wd, seeds, seed_offset):
    oof = torch.zeros((len(y), n_out), dtype=torch.float32)

    for k, test in enumerate(folds):
        train = [i for i in range(len(y)) if i not in set(test)]
        Xtr, Xte = standardize(X[train], X[test])

        probs = []
        for seed in seeds:
            m = train_head(
                Xtr, y[train], n_out, epochs, lr, wd,
                seed + seed_offset + 1000*k,
            )
            with torch.no_grad():
                probs.append(torch.softmax(m(Xte), dim=-1).cpu())

        oof[test] = torch.stack(probs).mean(0)

    return oof.argmax(-1), oof


def per_class_recall(gold, pred):
    out = {}
    for i,c in enumerate(CLASSES):
        mask = gold == i
        out[c] = float((pred[mask] == gold[mask]).float().mean().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--output-dir", default="results/relational_semantic_representation_v139")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("="*112)
    print(" LLM_GPU v1.3.9 Relational Semantic Representation Diagnostic")
    print("="*112)
    print("Device          :", device)
    print("Base            :", args.model)
    print("Dataset         : v1.3.7 symmetric reciprocal comparison, 60 prompts")
    print("Stages          :", ", ".join(STAGES))
    print("Target head     : 6 classes")
    print("Contrast head   : 6 classes")
    print("Pair head       : 15 unordered class pairs")
    print("Direction metric: ordered target/contrast correctness")
    print("Evaluation      : 5-fold target-class-stratified CV")
    print("Seeds/fold      :", seeds)
    print("Base model      : completely frozen")
    print()

    print("Extracting hidden states...")
    feats = {s:[] for s in STAGES}
    for i,sample in enumerate(SAMPLES, start=1):
        vecs = extract_stages(model, tok, sample["prompt"])
        for stage in STAGES:
            feats[stage].append(vecs[stage])
        if i % 10 == 0:
            print(f"  encoded {i}/60")

    feats = {s:torch.stack(v).to(device) for s,v in feats.items()}

    y_target = torch.tensor([CLASS_TO_ID[s["target"]] for s in SAMPLES], dtype=torch.long, device=device)
    y_contrast = torch.tensor([CLASS_TO_ID[s["contrast"]] for s in SAMPLES], dtype=torch.long, device=device)
    y_pair = torch.tensor([s["pair_id"] for s in SAMPLES], dtype=torch.long, device=device)

    folds = build_folds()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for stage in STAGES:
        X = feats[stage]

        pred_t, _ = oof_predict(
            X, y_target, len(CLASSES), folds,
            args.epochs, args.lr, args.weight_decay, seeds, 0,
        )
        pred_c, _ = oof_predict(
            X, y_contrast, len(CLASSES), folds,
            args.epochs, args.lr, args.weight_decay, seeds, 10000,
        )
        pred_p, _ = oof_predict(
            X, y_pair, len(PAIR_KEYS), folds,
            args.epochs, args.lr, args.weight_decay, seeds, 20000,
        )

        yt = y_target.cpu()
        yc = y_contrast.cpu()
        yp = y_pair.cpu()

        target_acc = float((pred_t == yt).float().mean().item())
        contrast_acc = float((pred_c == yc).float().mean().item())
        pair_acc = float((pred_p == yp).float().mean().item())

        # Role-free concept-pair recognition using the independently predicted target/contrast classes.
        predicted_unordered = []
        for a,b in zip(pred_t.tolist(), pred_c.tolist()):
            predicted_unordered.append(tuple(sorted((a,b))))
        gold_unordered = [tuple(sorted((int(a),int(b)))) for a,b in zip(yt.tolist(),yc.tolist())]
        unordered_joint = sum(p == g for p,g in zip(predicted_unordered,gold_unordered)) / len(gold_unordered)

        # Strict ordered-role correctness: both heads must assign the correct roles.
        ordered_joint = float(((pred_t == yt) & (pred_c == yc)).float().mean().item())

        # Direction given a correctly recognized unordered pair.
        correct_pair_mask = torch.tensor(
            [p == g for p,g in zip(predicted_unordered,gold_unordered)],
            dtype=torch.bool,
        )
        if bool(correct_pair_mask.any()):
            direction_given_pair = float(
                (((pred_t == yt) & (pred_c == yc))[correct_pair_mask]).float().mean().item()
            )
        else:
            direction_given_pair = 0.0

        target_per = per_class_recall(yt, pred_t)
        contrast_per = per_class_recall(yc, pred_c)

        print()
        print(stage.upper())
        print("-"*86)
        print(f"target accuracy             : {target_acc:.1%}")
        print(f"contrast accuracy           : {contrast_acc:.1%}")
        print(f"unordered pair head         : {pair_acc:.1%}")
        print(f"unordered pair from 2 heads : {unordered_joint:.1%}")
        print(f"ordered target+contrast     : {ordered_joint:.1%}")
        print(f"direction given pair known  : {direction_given_pair:.1%}")
        print("target recall:")
        for c in CLASSES:
            print(f"  {c:11s} {target_per[c]:.1%}")

        rows.append([
            stage,
            target_acc,
            contrast_acc,
            pair_acc,
            unordered_joint,
            ordered_joint,
            direction_given_pair,
            *[target_per[c] for c in CLASSES],
            *[contrast_per[c] for c in CLASSES],
        ])

    with (outdir/"summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "stage",
            "target_accuracy",
            "contrast_accuracy",
            "unordered_pair_head_accuracy",
            "unordered_pair_from_two_heads_accuracy",
            "ordered_target_contrast_accuracy",
            "direction_given_pair_accuracy",
            *[f"target_recall_{c}" for c in CLASSES],
            *[f"contrast_recall_{c}" for c in CLASSES],
        ])
        w.writerows(rows)

    print()
    print("Relational decomposition")
    print("------------------------")
    for r in rows:
        print(
            f"{r[0]:10s} target={r[1]:.1%} contrast={r[2]:.1%} "
            f"pair={r[3]:.1%} pair2h={r[4]:.1%} ordered={r[5]:.1%} "
            f"dir|pair={r[6]:.1%}"
        )

    best_pair = max(rows, key=lambda r:r[3])
    best_ordered = max(rows, key=lambda r:r[5])
    print()
    print(f"Best unordered-pair stage : {best_pair[0]} {best_pair[3]:.1%}")
    print(f"Best ordered-role stage   : {best_ordered[0]} {best_ordered[5]:.1%}")
    print()
    print("Interpretation")
    print("--------------")
    print("High unordered-pair accuracy with low ordered-role accuracy means both concepts")
    print("are represented but target/contrast role direction is lost.")
    print("Low pair accuracy means even the two concepts are not stably recoverable from")
    print("the final-token hidden state.")
    print("This directly tests whether semantic representation needs explicit concept+role")
    print("structure rather than one undifferentiated vector.")
    print("Output dir:", outdir)


if __name__ == "__main__":
    main()
