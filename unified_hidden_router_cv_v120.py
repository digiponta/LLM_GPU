# unified_hidden_router_cv_v120.py
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from fresh_generalization_cases_v114 import CASES
from evaluate_transformer_continuation_category_repair_v01117 import (
    DEFAULT_TOKENIZER, DEFAULT_MODEL,
)
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASSES = ["gpu","cpu","llm","transformer","cuda","python"]
CLASS_TO_ID = {name:i for i,name in enumerate(CLASSES)}


class LinearRouter(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.fc(x)


class MLPRouter(nn.Module):
    def __init__(self, d_model: int, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def encode_hidden(model, tok, prompt: str):
    device = next(model.parameters()).device
    ids = tok.encode(f"人: {prompt}\nAI: ", add_bos=True)
    x = torch.tensor([ids[-model.context_length:]], dtype=torch.long, device=device)
    return model.forward_hidden(x)[:, -1, :][0].detach()


def build_folds(labels, n_folds: int, seed: int):
    # Stratified folds. Each class has 10 Fresh-v2 technical prompts, so with
    # five folds each test fold receives exactly two prompts per class.
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)

    rng = random.Random(seed)
    folds = [[] for _ in range(n_folds)]
    for cls in range(len(CLASSES)):
        idxs = list(by_class[cls])
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[j % n_folds].append(idx)

    for f in folds:
        f.sort()
    return folds


def standardize_train_test(X_train, X_test):
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (X_train - mean) / std, (X_test - mean) / std


def make_model(arch: str, d_model: int, hidden: int, n_classes: int):
    if arch == "linear":
        return LinearRouter(d_model, n_classes)
    if arch == "mlp":
        return MLPRouter(d_model, hidden, n_classes)
    raise ValueError(arch)


def train_once(X_train, y_train, arch, hidden, epochs, lr, weight_decay, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = make_model(arch, X_train.shape[1], hidden, len(CLASSES)).to(X_train.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

    model.eval()
    return model


def evaluate_arch(X, y, prompts, folds, arch, hidden, epochs, lr, weight_decay, seeds):
    n = X.shape[0]
    ensemble_probs = torch.zeros((n, len(CLASSES)), dtype=torch.float32)
    fold_of = [-1] * n

    for fold_idx, test_idx in enumerate(folds):
        test_set = set(test_idx)
        train_idx = [i for i in range(n) if i not in test_set]

        Xt = X[train_idx]
        Xv = X[test_idx]
        yt = y[train_idx]
        Xt, Xv = standardize_train_test(Xt, Xv)

        seed_probs = []
        for seed in seeds:
            m = train_once(
                Xt, yt, arch, hidden, epochs, lr, weight_decay,
                seed + 1000 * fold_idx
            )
            seed_probs.append(torch.softmax(m(Xv), dim=-1).cpu())

        p = torch.stack(seed_probs, dim=0).mean(dim=0)
        ensemble_probs[test_idx] = p
        for i in test_idx:
            fold_of[i] = fold_idx + 1

    pred = ensemble_probs.argmax(dim=-1)
    correct = int((pred == y.cpu()).sum().item())
    acc = correct / n

    confusion = [[0 for _ in CLASSES] for _ in CLASSES]
    rows = []
    for i in range(n):
        gold_i = int(y[i].item())
        pred_i = int(pred[i].item())
        confusion[gold_i][pred_i] += 1
        ranked = torch.argsort(ensemble_probs[i], descending=True)
        top1 = int(ranked[0].item())
        top2 = int(ranked[1].item())
        rows.append({
            "id": f"V{i+1:03d}",
            "fold": fold_of[i],
            "gold": CLASSES[gold_i],
            "pred": CLASSES[pred_i],
            "correct": int(gold_i == pred_i),
            "top1_score": float(ensemble_probs[i, top1].item()),
            "top2": CLASSES[top2],
            "top2_score": float(ensemble_probs[i, top2].item()),
            "margin": float((ensemble_probs[i, top1] - ensemble_probs[i, top2]).item()),
            "prompt": prompts[i],
            **{f"p_{c}": float(ensemble_probs[i,j].item()) for j,c in enumerate(CLASSES)},
        })

    per_class = {}
    recalls = []
    for cls_i, cls in enumerate(CLASSES):
        total = sum(confusion[cls_i])
        hit = confusion[cls_i][cls_i]
        rec = hit / total if total else 0.0
        recalls.append(rec)
        per_class[cls] = (hit, total, rec)

    macro = sum(recalls) / len(recalls)
    return {
        "accuracy": acc,
        "correct": correct,
        "macro_recall": macro,
        "confusion": confusion,
        "per_class": per_class,
        "rows": rows,
    }


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_confusion(path: Path, confusion):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["gold"] + CLASSES)
        for i, cls in enumerate(CLASSES):
            w.writerow([cls] + confusion[i])


def print_result(arch, result):
    print()
    print(f"{arch.upper()} router")
    print("-" * 72)
    print(f"OOF accuracy    : {result['correct']}/60 ({result['accuracy']:.1%})")
    print(f"Macro recall    : {result['macro_recall']:.1%}")
    print("Per-class:")
    for cls in CLASSES:
        hit,total,rec = result["per_class"][cls]
        print(f"  {cls:11s} {hit}/{total} ({rec:.1%})")
    print("Confusion matrix: rows=gold, columns=predicted")
    print("gold\\pred   " + " ".join(f"{x[:5]:>5s}" for x in CLASSES))
    for i,g in enumerate(CLASSES):
        print(f"{g[:9]:9s} " + " ".join(f"{v:5d}" for v in result["confusion"][i]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--fold-seed", type=int, default=120)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--output-dir", default="results/unified_hidden_router_cv_v120")
    args = p.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    base, _ = LanguageModel.load_checkpoint(args.model, device=device)
    base.eval()
    for q in base.parameters():
        q.requires_grad_(False)

    tech = [(intent,prompt) for intent,prompt,_,_ in CASES if intent in CLASS_TO_ID]
    prompts = [p for _,p in tech]
    y = torch.tensor([CLASS_TO_ID[i] for i,_ in tech], dtype=torch.long, device=device)
    X = torch.stack([encode_hidden(base,tok,prompt) for prompt in prompts], dim=0)

    folds = build_folds(y.tolist(), 5, args.fold_seed)
    print("=" * 92)
    print(" LLM_GPU v1.2.0 Unified Hidden-State Semantic Router - 5-Fold CV")
    print("=" * 92)
    print("Device            :", device)
    print("Frozen base model :", args.model)
    print("Samples           :", len(prompts))
    print("Classes           :", ", ".join(CLASSES))
    print("Folds             : 5 stratified (2 examples/class/fold)")
    print("Seeds/fold        :", ", ".join(map(str,seeds)))
    print("Architectures     : Linear and MLP(256->64->6)")
    print("Important         : Fresh v2 is now a development set; this is CV, not final test.")
    print()

    results = {}
    for arch in ("linear","mlp"):
        results[arch] = evaluate_arch(
            X,y,prompts,folds,arch,args.hidden,args.epochs,args.lr,
            args.weight_decay,seeds
        )
        print_result(arch, results[arch])

    outdir = Path(args.output_dir)
    for arch,res in results.items():
        write_rows(outdir / f"{arch}_oof_predictions.csv", res["rows"])
        write_confusion(outdir / f"{arch}_confusion.csv", res["confusion"])

    summary_path = outdir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["architecture","accuracy","macro_recall","correct","total"])
        for arch in ("linear","mlp"):
            r=results[arch]
            w.writerow([arch,r["accuracy"],r["macro_recall"],r["correct"],60])

    best = max(("linear","mlp"), key=lambda a:(results[a]["accuracy"],results[a]["macro_recall"]))
    print()
    print("Comparison")
    print("----------")
    print(f"Linear OOF : {results['linear']['accuracy']:.1%}")
    print(f"MLP OOF    : {results['mlp']['accuracy']:.1%}")
    print(f"Best CV    : {best} ({results[best]['accuracy']:.1%})")
    print()
    print("Decision guide")
    print("--------------")
    print("If hidden-state CV is clearly above the 60% score-calibration reference,")
    print("continue with a unified hidden-state router. If not, expand semantic")
    print("training data before deploying a learned router.")
    print()
    print("Output dir:", outdir)


if __name__ == "__main__":
    main()
