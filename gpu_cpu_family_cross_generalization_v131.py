# gpu_cpu_family_cross_generalization_v131.py
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from semantic_router_dataset_v121 import DATASET
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

GPU = "gpu"
CPU = "cpu"


class BinaryProbe(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.fc = nn.Linear(d_in, 2)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def block1_vector(model: LanguageModel, tok: Tokenizer, prompt: str):
    device = next(model.parameters()).device
    ids = tok.encode(f"人: {prompt}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    h = model.embedding(x)
    if model.position_embedding is not None:
        pos = torch.arange(x.shape[1], device=device)
        h = h + model.position_embedding(pos).unsqueeze(0)

    h = model.blocks[0](h)
    return h[0, -1, :].detach().cpu()


def standardize(Xtr, Xte):
    mean = Xtr.mean(0, keepdim=True)
    std = Xtr.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (Xtr - mean) / std, (Xte - mean) / std


def train_probe(X, y, epochs, lr, wd, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    m = BinaryProbe(X.shape[1]).to(X.device)
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


@torch.no_grad()
def ensemble_predict(Xtr, ytr, Xte, epochs, lr, wd, seeds):
    Xtr, Xte = standardize(Xtr, Xte)
    probs = []
    for seed in seeds:
        with torch.enable_grad():
            m = train_probe(Xtr, ytr, epochs, lr, wd, seed)
        probs.append(torch.softmax(m(Xte), dim=-1).cpu())
    return torch.stack(probs).mean(0)


def accuracy_parts(y, pred):
    correct = int((pred == y).sum().item())
    total = len(y)
    gpu_mask = y == 0
    cpu_mask = y == 1
    gpu_acc = float((pred[gpu_mask] == y[gpu_mask]).float().mean().item()) if gpu_mask.any() else 0.0
    cpu_acc = float((pred[cpu_mask] == y[cpu_mask]).float().mean().item()) if cpu_mask.any() else 0.0
    return correct / total, gpu_acc, cpu_acc


def family_lists():
    gpu_families = list(dict.fromkeys(f for l,f,_ in DATASET if l == GPU))
    cpu_families = list(dict.fromkeys(f for l,f,_ in DATASET if l == CPU))
    assert len(gpu_families) == 5
    assert len(cpu_families) == 5
    return gpu_families, cpu_families


def family_indices(label, family):
    return [i for i,(l,f,_) in enumerate(DATASET) if l == label and f == family]


def pair_indices(gfam, cfam):
    return family_indices(GPU, gfam) + family_indices(CPU, cfam)


def pair_labels(indices, device=None):
    return torch.tensor(
        [0 if DATASET[i][0] == GPU else 1 for i in indices],
        dtype=torch.long,
        device=device,
    )


def within_pair_cv(X, gfam, cfam, epochs, lr, wd, seeds):
    gidx = family_indices(GPU, gfam)
    cidx = family_indices(CPU, cfam)
    assert len(gidx) == 10 and len(cidx) == 10

    # Deterministic 5-fold: 2 GPU + 2 CPU examples/fold.
    all_probs = []
    all_gold = []
    for fold in range(5):
        test = gidx[2*fold:2*fold+2] + cidx[2*fold:2*fold+2]
        train = [i for i in gidx+cidx if i not in set(test)]
        ytr = pair_labels(train, X.device)
        yte = pair_labels(test).cpu()
        p = ensemble_predict(
            X[train], ytr, X[test],
            epochs, lr, wd,
            [s + fold*1000 for s in seeds],
        )
        all_probs.append(p)
        all_gold.append(yte)

    probs = torch.cat(all_probs)
    gold = torch.cat(all_gold)
    pred = probs.argmax(-1)
    return accuracy_parts(gold, pred)


def leave_pair_out(X, gfam, cfam, epochs, lr, wd, seeds):
    test = pair_indices(gfam, cfam)
    train = [
        i for i,(label,family,_) in enumerate(DATASET)
        if label in (GPU,CPU)
        and not (label == GPU and family == gfam)
        and not (label == CPU and family == cfam)
    ]
    assert len(train) == 80 and len(test) == 20

    ytr = pair_labels(train, X.device)
    yte = pair_labels(test).cpu()
    p = ensemble_predict(X[train], ytr, X[test], epochs, lr, wd, seeds)
    return accuracy_parts(yte, p.argmax(-1))


def source_pair_to_others(X, gfam, cfam, epochs, lr, wd, seeds):
    train = pair_indices(gfam, cfam)
    test = [
        i for i,(label,family,_) in enumerate(DATASET)
        if label in (GPU,CPU)
        and not (label == GPU and family == gfam)
        and not (label == CPU and family == cfam)
    ]
    assert len(train) == 20 and len(test) == 80

    ytr = pair_labels(train, X.device)
    yte = pair_labels(test).cpu()
    p = ensemble_predict(X[train], ytr, X[test], epochs, lr, wd, seeds)
    return accuracy_parts(yte, p.argmax(-1))


def write_matrix(path, gpu_families, cpu_families, values):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["gpu_family\\cpu_family"] + cpu_families)
        for g in gpu_families:
            w.writerow([g] + [values[(g,c)] for c in cpu_families])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--output-dir", default="results/gpu_cpu_family_cross_generalization_v131")
    args = p.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)

    gpu_families, cpu_families = family_lists()

    print("="*106)
    print(" LLM_GPU v1.3.1 GPU-CPU Family Cross-Generalization Matrix")
    print("="*106)
    print("Device            :", device)
    print("Base              :", args.model)
    print("Representation    : frozen Block1 final-token hidden state")
    print("Probe             : Linear 256 -> 2")
    print("GPU families      :", ", ".join(gpu_families))
    print("CPU families      :", ", ".join(cpu_families))
    print("Seeds             :", seeds)
    print()
    print("Matrices:")
    print("  A. within-pair 5-fold CV")
    print("  B. leave-target-pair-out generalization")
    print("  C. source-pair-only -> all other families")
    print()

    print("Extracting Block1 vectors for GPU/CPU prompts...")
    vectors = []
    for i,(_,_,prompt) in enumerate(DATASET, start=1):
        vectors.append(block1_vector(model, tok, prompt))
        if i % 50 == 0:
            print(f"  encoded {i}/300")
    X = torch.stack(vectors).to(device)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    modes = {
        "within_pair": within_pair_cv,
        "leave_pair_out": leave_pair_out,
        "source_pair_to_others": source_pair_to_others,
    }

    detailed_rows = []
    matrices = {}

    for mode_name, fn in modes.items():
        print()
        print(mode_name.upper())
        print("-"*90)
        acc_values = {}
        gpu_values = {}
        cpu_values = {}

        for gfam in gpu_families:
            line = []
            for cfam in cpu_families:
                acc,gacc,cacc = fn(
                    X,gfam,cfam,args.epochs,args.lr,args.weight_decay,seeds
                )
                acc_values[(gfam,cfam)] = acc
                gpu_values[(gfam,cfam)] = gacc
                cpu_values[(gfam,cfam)] = cacc
                detailed_rows.append([
                    mode_name,gfam,cfam,acc,gacc,cacc
                ])
                line.append(f"{acc:.0%}")
            print(f"{gfam:14s}: " + " ".join(f"{x:>5s}" for x in line))

        matrices[mode_name] = acc_values
        write_matrix(outdir/f"{mode_name}_accuracy.csv",gpu_families,cpu_families,acc_values)
        write_matrix(outdir/f"{mode_name}_gpu_recall.csv",gpu_families,cpu_families,gpu_values)
        write_matrix(outdir/f"{mode_name}_cpu_recall.csv",gpu_families,cpu_families,cpu_values)

        vals = list(acc_values.values())
        print(f"Mean={sum(vals)/len(vals):.1%}  Min={min(vals):.1%}  Max={max(vals):.1%}")

    with (outdir/"family_pair_details.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["mode","gpu_family","cpu_family","accuracy","gpu_recall","cpu_recall"])
        w.writerows(detailed_rows)

    # Aggregate vulnerability by target family using leave-pair-out matrix.
    loo = matrices["leave_pair_out"]
    gpu_target_rows = []
    for gfam in gpu_families:
        vals = [loo[(gfam,c)] for c in cpu_families]
        gpu_target_rows.append([gfam,sum(vals)/len(vals),min(vals),max(vals)])
    cpu_target_rows = []
    for cfam in cpu_families:
        vals = [loo[(g,cfam)] for g in gpu_families]
        cpu_target_rows.append([cfam,sum(vals)/len(vals),min(vals),max(vals)])

    with (outdir/"target_family_vulnerability.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["side","family","mean_leave_pair_out_accuracy","min","max"])
        for r in gpu_target_rows:
            w.writerow(["gpu",*r])
        for r in cpu_target_rows:
            w.writerow(["cpu",*r])

    print()
    print("Target-family vulnerability (leave-pair-out mean)")
    print("------------------------------------------------")
    for fam,meanv,minv,maxv in sorted(gpu_target_rows,key=lambda r:r[1]):
        print(f"GPU {fam:14s} mean={meanv:.1%} min={minv:.1%} max={maxv:.1%}")
    for fam,meanv,minv,maxv in sorted(cpu_target_rows,key=lambda r:r[1]):
        print(f"CPU {fam:14s} mean={meanv:.1%} min={minv:.1%} max={maxv:.1%}")

    worst = min(loo.items(), key=lambda kv: kv[1])
    best = max(loo.items(), key=lambda kv: kv[1])
    print()
    print("Leave-pair-out extremes")
    print("-----------------------")
    print(f"Worst target pair: GPU={worst[0][0]} CPU={worst[0][1]} accuracy={worst[1]:.1%}")
    print(f"Best target pair : GPU={best[0][0]} CPU={best[0][1]} accuracy={best[1]:.1%}")
    print()
    print("Interpretation:")
    print("A low within-pair score means the two family definitions overlap even in-domain.")
    print("High within-pair but low leave-pair-out means family-specific wording is being learned")
    print("instead of a transferable GPU-vs-CPU concept boundary.")
    print("Low source-pair-to-others means that pair is a poor semantic teaching basis.")
    print("Output dir:", outdir)


if __name__ == "__main__":
    main()
