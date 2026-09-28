# class_specific_layer_gating_v128.py
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
GPU_ID = CLASS_TO_ID["gpu"]


class LinearGeneralRouter(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.fc(x)


class GPUSpecialist(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def family_folds():
    fams = defaultdict(list)
    for label, family, _ in DATASET:
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
def extract_block1_block4(model: LanguageModel, tok: Tokenizer, prompt: str):
    device = next(model.parameters()).device
    ids = tok.encode(f"人: {prompt}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)

    x = model.embedding(token_ids)
    if model.position_embedding is not None:
        positions = torch.arange(token_ids.shape[1], device=device)
        x = x + model.position_embedding(positions).unsqueeze(0)

    block1 = None
    block4 = None
    for i, block in enumerate(model.blocks, start=1):
        x = block(x)
        if i == 1:
            block1 = x[:, -1, :][0].detach().cpu()
        if i == 4:
            block4 = x[:, -1, :][0].detach().cpu()
            break

    if block1 is None or block4 is None:
        raise RuntimeError("Could not extract Block1/Block4 features.")
    return block1, block4


def standardize(Xtr, Xte):
    mean = Xtr.mean(0, keepdim=True)
    std = Xtr.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std, (Xte-mean)/std


def train_general(X, y, epochs, lr, wd, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    m = LinearGeneralRouter(X.shape[1], len(CLASSES)).to(X.device)
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


def train_gpu_specialist(X, y_binary, hidden, epochs, lr, wd, positive_weight, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    m = GPUSpecialist(X.shape[1], hidden).to(X.device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    weights = torch.tensor([1.0, positive_weight], device=X.device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        loss = loss_fn(m(X), y_binary)
        loss.backward()
        opt.step()

    m.eval()
    return m


def confusion_and_per_class(y, pred):
    cm = [[0]*len(CLASSES) for _ in CLASSES]
    for g,p in zip(y.tolist(), pred.tolist()):
        cm[g][p] += 1

    correct = sum(cm[i][i] for i in range(len(CLASSES)))
    per = {}
    for i,c in enumerate(CLASSES):
        total = sum(cm[i])
        hit = cm[i][i]
        per[c] = (hit,total,hit/total if total else 0.0)
    macro = sum(v[2] for v in per.values())/len(per)
    return correct/len(y), correct, macro, cm, per


def apply_gate(general_pred, gpu_prob, threshold):
    pred = general_pred.clone()
    mask = (gpu_prob >= threshold) & (general_pred != GPU_ID)
    pred[mask] = GPU_ID
    return pred, mask


def write_confusion(path, cm):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["gold"]+CLASSES)
        for i,c in enumerate(CLASSES):
            w.writerow([c]+cm[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--general-lr", type=float, default=1e-3)
    p.add_argument("--specialist-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--specialist-hidden", type=int, default=64)
    p.add_argument("--positive-weight", type=float, default=5.0)
    p.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.85,0.90,0.95")
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--output-dir", default="results/class_specific_layer_gating_v128")
    args = p.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    base, _ = LanguageModel.load_checkpoint(args.model, device=device)
    base.eval()
    for q in base.parameters():
        q.requires_grad_(False)

    prompts = [x[2] for x in DATASET]
    y = torch.tensor([CLASS_TO_ID[x[0]] for x in DATASET], dtype=torch.long, device=device)

    print("="*102)
    print(" LLM_GPU v1.2.8 Class-Specific Layer Gating")
    print("="*102)
    print("Device            :", device)
    print("Base              :", args.model)
    print("General router    : Block1 -> Linear 256 -> 6")
    print("GPU specialist    : Block4 -> MLP 256 -> 64 -> 2")
    print("Specialist target : GPU vs Non-GPU")
    print("Positive weight   :", args.positive_weight)
    print("Threshold sweep   :", thresholds)
    print("Evaluation        : 5-fold family-held-out CV")
    print("Seeds/fold        :", seeds)
    print("Base model        : completely frozen")
    print()

    print("Extracting Block1 and Block4 features...")
    b1_list, b4_list = [], []
    for i,prompt in enumerate(prompts, start=1):
        b1,b4 = extract_block1_block4(base, tok, prompt)
        b1_list.append(b1)
        b4_list.append(b4)
        if i % 50 == 0:
            print(f"  encoded {i}/300")

    X1 = torch.stack(b1_list).to(device)
    X4 = torch.stack(b4_list).to(device)
    folds = family_folds()

    general_oof = torch.zeros((len(DATASET),len(CLASSES)), dtype=torch.float32)
    gpu_oof = torch.zeros(len(DATASET), dtype=torch.float32)

    for fold_idx,test_idx in enumerate(folds):
        test_set = set(test_idx)
        train_idx = [i for i in range(len(DATASET)) if i not in test_set]

        X1tr,X1te = standardize(X1[train_idx], X1[test_idx])
        X4tr,X4te = standardize(X4[train_idx], X4[test_idx])
        ytr = y[train_idx]
        ybin = (ytr == GPU_ID).long()

        general_probs = []
        specialist_probs = []

        for seed in seeds:
            g = train_general(
                X1tr,ytr,args.epochs,args.general_lr,args.weight_decay,
                seed + 1000*fold_idx,
            )
            s = train_gpu_specialist(
                X4tr,ybin,args.specialist_hidden,args.epochs,args.specialist_lr,
                args.weight_decay,args.positive_weight,seed + 2000*fold_idx,
            )

            with torch.no_grad():
                general_probs.append(torch.softmax(g(X1te),dim=-1).cpu())
                specialist_probs.append(torch.softmax(s(X4te),dim=-1)[:,1].cpu())

        general_oof[test_idx] = torch.stack(general_probs).mean(0)
        gpu_oof[test_idx] = torch.stack(specialist_probs).mean(0)

    y_cpu = y.cpu()
    general_pred = general_oof.argmax(-1)
    base_acc,base_correct,base_macro,base_cm,base_per = confusion_and_per_class(y_cpu,general_pred)

    print()
    print("BASELINE: Block1 general router")
    print("-"*82)
    print(f"Family-held-out : {base_correct}/300 ({base_acc:.1%}) macro={base_macro:.1%}")
    for c in CLASSES:
        h,n,r=base_per[c]
        print(f"  {c:11s} {h:2d}/{n:2d} ({r:.1%})")

    # Specialist diagnostic independent of gating.
    gpu_gold = (y_cpu == GPU_ID)
    print()
    print("GPU specialist score diagnostic")
    print("-"*82)
    print(f"Mean GPU score on GPU     : {gpu_oof[gpu_gold].mean().item():.4f}")
    print(f"Mean GPU score on Non-GPU : {gpu_oof[~gpu_gold].mean().item():.4f}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_confusion(outdir/"baseline_block1_confusion.csv",base_cm)

    sweep_rows = []
    best = None

    for th in thresholds:
        pred,override = apply_gate(general_pred,gpu_oof,th)
        acc,correct,macro,cm,per = confusion_and_per_class(y_cpu,pred)

        override_count = int(override.sum().item())
        helpful = int(((pred == y_cpu) & (general_pred != y_cpu) & override).sum().item())
        harmful = int(((pred != y_cpu) & (general_pred == y_cpu) & override).sum().item())
        correct_gpu_overrides = int(((y_cpu == GPU_ID) & override).sum().item())
        false_gpu_overrides = int(((y_cpu != GPU_ID) & override).sum().item())
        precision = correct_gpu_overrides / override_count if override_count else 0.0

        print()
        print(f"threshold={th:.2f}")
        print("-"*82)
        print(f"Family-held-out : {correct}/300 ({acc:.1%}) macro={macro:.1%}")
        print(
            f"Overrides={override_count} helpful={helpful} harmful={harmful} "
            f"GPU-correct={correct_gpu_overrides} GPU-false={false_gpu_overrides} "
            f"precision={precision:.1%}"
        )
        for c in CLASSES:
            h,n,r=per[c]
            print(f"  {c:11s} {h:2d}/{n:2d} ({r:.1%})")

        tag=str(th).replace(".","p")
        write_confusion(outdir/f"threshold_{tag}_confusion.csv",cm)
        sweep_rows.append([
            th,acc,macro,override_count,helpful,harmful,
            correct_gpu_overrides,false_gpu_overrides,precision,
            *[per[c][2] for c in CLASSES],
        ])
        key=(acc,macro,precision,-false_gpu_overrides)
        if best is None or key > best[0]:
            best=(key,th)

    with (outdir/"threshold_sweep.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow([
            "threshold","accuracy","macro_recall","overrides","helpful","harmful",
            "correct_gpu_overrides","false_gpu_overrides","override_precision",
            *[f"recall_{c}" for c in CLASSES],
        ])
        w.writerows(sweep_rows)

    rows=[]
    for i,(label,family,prompt) in enumerate(DATASET):
        rows.append({
            "id":f"G{i+1:03d}",
            "gold":label,
            "family":family,
            "prompt":prompt,
            "general_pred":CLASSES[int(general_pred[i])],
            "general_conf":float(general_oof[i].max().item()),
            "gpu_specialist_score":float(gpu_oof[i].item()),
        })
    with (outdir/"oof_scores.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print()
    print("Comparison")
    print("----------")
    print(f"Block1 baseline : {base_acc:.1%}")
    for row in sweep_rows:
        print(
            f"th={row[0]:.2f} acc={row[1]:.1%} macro={row[2]:.1%} "
            f"gpu={row[9]:.1%} overrides={row[3]} falseGPU={row[7]}"
        )
    print()
    print(f"Best threshold by dev family-CV: {best[1]:.2f}")
    print("Reference v1.2.7 Block1 Linear: 60.7%")
    print()
    print("Important:")
    print("Threshold selection uses the same 300-prompt development set. This is not")
    print("a final unbiased generalization result. A future untouched Fresh-v3 set is")
    print("required before deploying the gating policy.")
    print("Output dir:",outdir)


if __name__=="__main__":
    main()
