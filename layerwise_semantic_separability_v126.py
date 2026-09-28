# layerwise_semantic_separability_v126.py
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from semantic_router_dataset_v121 import DATASET, CLASSES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
STAGES = ["embedding","block1","block2","block3","block4","block5","block6","final_norm"]


class LinearProbe(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.fc(x)


def family_folds():
    fams = defaultdict(list)
    for label,family,_ in DATASET:
        fams[label].append(family)

    order = {c:list(dict.fromkeys(fams[c])) for c in CLASSES}
    for c in CLASSES:
        assert len(order[c]) == 5

    folds = []
    for k in range(5):
        test = [i for i,(label,family,_) in enumerate(DATASET) if family == order[label][k]]
        assert len(test) == 60
        folds.append(test)
    return folds


@torch.no_grad()
def extract_stage_vectors(model: LanguageModel, tok: Tokenizer, prompt: str):
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
        out[f"block{i}"] = x[:, -1, :][0].detach().cpu()

    x = model.final_norm(x)
    out["final_norm"] = x[:, -1, :][0].detach().cpu()
    return out


def standardize(X_train, X_test):
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (X_train - mean) / std, (X_test - mean) / std


def train_probe(X, y, epochs, lr, weight_decay, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    probe = LinearProbe(X.shape[1], len(CLASSES)).to(X.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        loss = loss_fn(probe(X), y)
        loss.backward()
        opt.step()

    probe.eval()
    return probe


@torch.no_grad()
def eval_stage(X, y, folds, epochs, lr, weight_decay, seeds):
    n = len(y)
    oof = torch.zeros((n, len(CLASSES)), dtype=torch.float32)

    for fold_idx, test_idx in enumerate(folds):
        test_set = set(test_idx)
        train_idx = [i for i in range(n) if i not in test_set]

        Xtr = X[train_idx]
        Xte = X[test_idx]
        ytr = y[train_idx]
        Xtr, Xte = standardize(Xtr, Xte)

        seed_probs = []
        for seed in seeds:
            # Training requires gradients, so temporarily leave no_grad.
            with torch.enable_grad():
                probe = train_probe(
                    Xtr, ytr, epochs, lr, weight_decay,
                    seed + fold_idx * 1000,
                )
            seed_probs.append(torch.softmax(probe(Xte), dim=-1).cpu())

        oof[test_idx] = torch.stack(seed_probs, dim=0).mean(dim=0)

    pred = oof.argmax(dim=-1)
    y_cpu = y.cpu()

    cm = [[0] * len(CLASSES) for _ in CLASSES]
    for g,p in zip(y_cpu.tolist(), pred.tolist()):
        cm[g][p] += 1

    correct = sum(cm[i][i] for i in range(len(CLASSES)))
    per = {}
    for i,c in enumerate(CLASSES):
        total = sum(cm[i])
        hit = cm[i][i]
        per[c] = (hit, total, hit/total if total else 0.0)

    macro = sum(v[2] for v in per.values()) / len(per)
    return {
        "accuracy": correct / len(y_cpu),
        "correct": correct,
        "macro": macro,
        "confusion": cm,
        "per_class": per,
        "probs": oof,
    }


def write_confusion(path: Path, cm):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["gold"] + CLASSES)
        for i,c in enumerate(CLASSES):
            w.writerow([c] + cm[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--output-dir", default="results/layerwise_semantic_separability_v126")
    args = p.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    if len(model.blocks) != 6:
        raise ValueError(f"Expected 6 transformer blocks, got {len(model.blocks)}")

    prompts = [x[2] for x in DATASET]
    y = torch.tensor([CLASS_TO_ID[x[0]] for x in DATASET], dtype=torch.long, device=device)

    print("=" * 100)
    print(" LLM_GPU v1.2.6 Layer-wise Semantic Separability Diagnostic")
    print("=" * 100)
    print("Device          :", device)
    print("Base            :", args.model)
    print("Dataset         : 300 prompts, 6 classes x 5 families x 10")
    print("Stages          :", ", ".join(STAGES))
    print("Probe           : Linear 256 -> 6")
    print("Evaluation      : 5-fold family-held-out CV")
    print("Seeds/fold      :", seeds)
    print("Base model      : completely frozen")
    print()

    features = {stage: [] for stage in STAGES}
    print("Extracting layer-wise hidden states...")
    for i,prompt in enumerate(prompts, start=1):
        vecs = extract_stage_vectors(model, tok, prompt)
        for stage in STAGES:
            features[stage].append(vecs[stage])
        if i % 50 == 0:
            print(f"  encoded {i}/300")

    features = {
        stage: torch.stack(vectors).to(device)
        for stage,vectors in features.items()
    }

    folds = family_folds()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = []
    all_results = {}

    for stage in STAGES:
        result = eval_stage(
            features[stage], y, folds,
            args.epochs, args.lr, args.weight_decay, seeds,
        )
        all_results[stage] = result

        print()
        print(stage.upper())
        print("-" * 78)
        print(
            f"Family-held-out OOF : {result['correct']}/300 "
            f"({result['accuracy']:.1%}) macro={result['macro']:.1%}"
        )
        for c in CLASSES:
            hit,total,rec = result["per_class"][c]
            print(f"  {c:11s} {hit:2d}/{total:2d} ({rec:.1%})")

        write_confusion(outdir / f"{stage}_confusion.csv", result["confusion"])
        summary.append([
            stage, result["accuracy"], result["macro"],
            *[result["per_class"][c][2] for c in CLASSES]
        ])

    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "stage","accuracy","macro_recall",
            *[f"recall_{c}" for c in CLASSES]
        ])
        w.writerows(summary)

    print()
    print("Layer-wise comparison")
    print("---------------------")
    for row in summary:
        print(
            f"{row[0]:10s} acc={row[1]:.1%} macro={row[2]:.1%} "
            f"gpu={row[3]:.1%} cpu={row[4]:.1%} llm={row[5]:.1%} "
            f"trans={row[6]:.1%} cuda={row[7]:.1%} python={row[8]:.1%}"
        )

    best = max(summary, key=lambda r: (r[1], r[2]))
    print()
    print(f"Best stage: {best[0]}  accuracy={best[1]:.1%}")
    print("Reference v1.2.2 frozen final representation: 55.0%")
    print("Reference v1.2.5 adapted late blocks best:    54.7%")
    print()
    print("Decision guide")
    print("--------------")
    print("If an earlier block is clearly better than Block6/FinalNorm, later processing")
    print("is erasing semantic separability and routing should tap that earlier layer.")
    print("If all blocks remain weak for GPU/CPU, the base LM training/data is the likely")
    print("bottleneck and semantic supervision must reach earlier representation learning.")
    print("Output dir:", outdir)


if __name__ == "__main__":
    main()
