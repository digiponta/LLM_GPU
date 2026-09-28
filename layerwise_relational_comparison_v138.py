# layerwise_relational_comparison_v138.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from semantic_symmetric_comparison_v137 import DATASET, CLASSES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
STAGES = ["embedding","block1","block2","block3","block4","block5","block6","final_norm"]


class LinearProbe(nn.Module):
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
        positions = torch.arange(token_ids.shape[1], device=device)
        x = x + model.position_embedding(positions).unsqueeze(0)

    out = {"embedding": x[:, -1, :][0].detach().cpu()}

    for i, block in enumerate(model.blocks, start=1):
        x = block(x)
        if i <= 6:
            out[f"block{i}"] = x[:, -1, :][0].detach().cpu()

    x = model.final_norm(x)
    out["final_norm"] = x[:, -1, :][0].detach().cpu()
    return out


def standardize(Xtr, Xte):
    mean = Xtr.mean(0, keepdim=True)
    std = Xtr.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (Xtr - mean) / std, (Xte - mean) / std


def train_probe(X, y, epochs, lr, wd, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    p = LinearProbe(X.shape[1], len(CLASSES)).to(X.device)
    opt = torch.optim.AdamW(p.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        p.train()
        opt.zero_grad()
        loss = loss_fn(p(X), y)
        loss.backward()
        opt.step()

    p.eval()
    return p


def build_folds():
    # 10 comparison prompts per class -> 5 folds, 2/class/fold.
    folds = [[] for _ in range(5)]
    for c in CLASSES:
        idx = [i for i,(label,_,_) in enumerate(DATASET) if label == c]
        assert len(idx) == 10
        for k in range(5):
            folds[k].extend(idx[2*k:2*k+2])
    return folds


def evaluate_stage(X, y, folds, epochs, lr, wd, seeds):
    oof = torch.zeros((len(y), len(CLASSES)), dtype=torch.float32)

    for k, test in enumerate(folds):
        train = [i for i in range(len(y)) if i not in set(test)]
        Xtr, Xte = standardize(X[train], X[test])
        probs = []

        for seed in seeds:
            p = train_probe(
                Xtr, y[train], epochs, lr, wd,
                seed + 1000*k
            )
            with torch.no_grad():
                probs.append(torch.softmax(p(Xte), dim=-1).cpu())

        oof[test] = torch.stack(probs).mean(0)

    pred = oof.argmax(-1)
    yy = y.cpu()
    acc = float((pred == yy).float().mean().item())

    per = {}
    cm = [[0]*len(CLASSES) for _ in CLASSES]
    for g,p in zip(yy.tolist(), pred.tolist()):
        cm[g][p] += 1

    for i,c in enumerate(CLASSES):
        mask = yy == i
        per[c] = float((pred[mask] == yy[mask]).float().mean().item())

    return acc, per, cm


def write_confusion(path: Path, cm):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["gold"] + CLASSES)
        for i,c in enumerate(CLASSES):
            w.writerow([c] + cm[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--output-dir", default="results/layerwise_relational_comparison_v138")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("="*110)
    print(" LLM_GPU v1.3.8 Layer-wise Relational Comparison Diagnostic")
    print("="*110)
    print("Device         :", device)
    print("Base           :", args.model)
    print("Dataset        : v1.3.7 symmetric comparison, 60 prompts")
    print("Classes        :", ", ".join(CLASSES))
    print("Stages         :", ", ".join(STAGES))
    print("Probe          : Linear 256 -> 6")
    print("Evaluation     : 5-fold stratified by class")
    print("Seeds/fold     :", seeds)
    print("Base model     : completely frozen")
    print()

    print("Extracting layer-wise hidden states...")
    feats = {s:[] for s in STAGES}
    for i,(_,_,prompt) in enumerate(DATASET, start=1):
        vecs = extract_stages(model, tok, prompt)
        for s in STAGES:
            feats[s].append(vecs[s])
        if i % 10 == 0:
            print(f"  encoded {i}/60")

    feats = {s:torch.stack(v).to(device) for s,v in feats.items()}
    y = torch.tensor([CLASS_TO_ID[label] for label,_,_ in DATASET], dtype=torch.long, device=device)
    folds = build_folds()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = []

    for stage in STAGES:
        acc, per, cm = evaluate_stage(
            feats[stage], y, folds,
            args.epochs, args.lr, args.weight_decay, seeds
        )
        print()
        print(stage.upper())
        print("-"*80)
        print(f"accuracy={acc:.1%}")
        for c in CLASSES:
            print(f"  {c:11s} {per[c]:.1%}")

        write_confusion(outdir / f"{stage}_confusion.csv", cm)
        summary.append([
            stage, acc, *[per[c] for c in CLASSES]
        ])

    with (outdir/"summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["stage","accuracy",*[f"recall_{c}" for c in CLASSES]])
        w.writerows(summary)

    print()
    print("Layer-wise comparison")
    print("---------------------")
    for row in summary:
        print(
            f"{row[0]:10s} acc={row[1]:.1%} "
            f"gpu={row[2]:.1%} cpu={row[3]:.1%} llm={row[4]:.1%} "
            f"trans={row[5]:.1%} cuda={row[6]:.1%} python={row[7]:.1%}"
        )

    best = max(summary, key=lambda r:r[1])
    print()
    print(f"Best stage: {best[0]} accuracy={best[1]:.1%}")
    print("Reference v1.3.7 Block1 symmetric comparison within-axis: 5.0%")
    print()
    print("Interpretation")
    print("--------------")
    print("If deeper layers substantially outperform Block1, relational direction emerges")
    print("later than basic concept identity and semantic routing may need layer-dependent taps.")
    print("If all layers remain near chance, the current base model has not learned stable")
    print("reciprocal contrast/negation relations and training data/objectives must address them.")
    print("Output dir:", outdir)


if __name__ == "__main__":
    main()
