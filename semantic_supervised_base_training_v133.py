# semantic_supervised_base_training_v133.py
from __future__ import annotations

import argparse
import copy
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_router_dataset_v121 import DATASET, CLASSES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

GPU, CPU = "gpu", "cpu"
CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}


class SemanticHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class LinearProbe(nn.Module):
    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_out)

    def forward(self, x):
        return self.fc(x)


def family_order():
    fams = defaultdict(list)
    for label, family, _ in DATASET:
        fams[label].append(family)
    out = {c:list(dict.fromkeys(fams[c])) for c in CLASSES}
    for c in CLASSES:
        assert len(out[c]) == 5
    return out


def encode(tok, model, prompt):
    ids = tok.encode(f"人: {prompt}\nAI: ", add_bos=True)
    return ids[-model.context_length:]


def configure_trainable(model: LanguageModel):
    for p in model.parameters():
        p.requires_grad_(False)

    for p in model.embedding.parameters():
        p.requires_grad_(True)
    if model.position_embedding is not None:
        for p in model.position_embedding.parameters():
            p.requires_grad_(True)
    for block in model.blocks[:2]:
        for p in block.parameters():
            p.requires_grad_(True)

    # Blocks 3-6, FinalNorm, and LM Head stay frozen.
    return model


def forward_with_block1(model, token_ids):
    x = model.embedding(token_ids)
    if model.position_embedding is not None:
        pos = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = x + model.position_embedding(pos).unsqueeze(0)

    x = model.blocks[0](x)
    block1_last = x[:, -1, :]

    for block in model.blocks[1:]:
        x = block(x)
    x = model.final_norm(x)
    logits = model.lm_head(x)
    return logits, block1_last


def lm_loss(logits, token_ids):
    if token_ids.shape[1] < 2:
        return logits.new_zeros(())
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        token_ids[:, 1:].reshape(-1),
    )


def semantic_ce(head, h, label):
    y = torch.tensor([0 if label == GPU else 1], dtype=torch.long, device=h.device)
    return F.cross_entropy(head(h), y)


def anchor_family_loss(model, head, token_cache, train_indices, order, device, seed):
    rng = random.Random(seed)
    reps = []
    ys = []

    for label, y in ((GPU,0),(CPU,1)):
        seen_families = []
        for fam in order[label]:
            candidates = [i for i in train_indices if DATASET[i][0] == label and DATASET[i][1] == fam]
            if not candidates:
                continue
            idx = rng.choice(candidates)
            ids = torch.tensor([token_cache[idx]], dtype=torch.long, device=device)
            _, h = forward_with_block1(model, ids)
            reps.append(h[0])
            ys.append(y)
            seen_families.append(fam)

    if len(reps) < 4:
        return torch.tensor(0.0, device=device)

    z = torch.stack(reps)
    y = torch.tensor(ys, dtype=torch.long, device=device)

    compact = z.new_zeros(())
    centroids = []
    for cls in (0,1):
        cls_z = z[y == cls]
        centroid = cls_z.mean(0)
        centroids.append(centroid)
        compact = compact + ((cls_z - centroid) ** 2).mean()
    compact = compact / 2.0

    c0 = F.normalize(centroids[0], dim=0)
    c1 = F.normalize(centroids[1], dim=0)
    separation = F.relu(torch.tensor(1.0, device=device) - torch.norm(c0-c1, p=2)).pow(2)

    return compact + separation


def train_base(
    base, token_cache, train_indices, order, *,
    epochs, base_lr, semantic_lr, wd, lm_weight, sem_weight, inv_weight, seed,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = next(base.parameters()).device
    model = copy.deepcopy(base).to(device)
    configure_trainable(model)
    head = SemanticHead(model.d_model).to(device)

    early_params = list(model.embedding.parameters())
    if model.position_embedding is not None:
        early_params += list(model.position_embedding.parameters())
    early_params += list(model.blocks[0].parameters()) + list(model.blocks[1].parameters())

    opt = torch.optim.AdamW([
        {"params": early_params, "lr": base_lr},
        {"params": head.parameters(), "lr": semantic_lr},
    ], weight_decay=wd)

    rng = random.Random(seed)
    for epoch in range(epochs):
        model.train()
        head.train()
        shuffled = list(train_indices)
        rng.shuffle(shuffled)

        for idx in shuffled:
            label = DATASET[idx][0]
            ids = torch.tensor([token_cache[idx]], dtype=torch.long, device=device)

            opt.zero_grad()
            logits, h = forward_with_block1(model, ids)
            loss = lm_weight * lm_loss(logits, ids)
            if label in (GPU, CPU):
                loss = loss + sem_weight * semantic_ce(head, h, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(early_params + list(head.parameters()), 1.0)
            opt.step()

        # One explicit family-invariance update per epoch using one anchor/family.
        opt.zero_grad()
        inv = anchor_family_loss(model, head, token_cache, train_indices, order, device, seed + epoch)
        (inv_weight * inv).backward()
        torch.nn.utils.clip_grad_norm_(early_params + list(head.parameters()), 1.0)
        opt.step()

    model.eval()
    head.eval()
    return model, head


@torch.no_grad()
def block1_features(model, token_cache, indices, device):
    rows = []
    for idx in indices:
        ids = torch.tensor([token_cache[idx]], dtype=torch.long, device=device)
        _, h = forward_with_block1(model, ids)
        rows.append(h[0].detach())
    return torch.stack(rows)


def standardize(Xtr, Xte):
    mean = Xtr.mean(0, keepdim=True)
    std = Xtr.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std, (Xte-mean)/std


def train_probe(X, y, n_out, epochs, lr, wd, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    p = LinearProbe(X.shape[1], n_out).to(X.device)
    opt = torch.optim.AdamW(p.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        p.train(); opt.zero_grad()
        loss = loss_fn(p(X), y)
        loss.backward(); opt.step()
    p.eval()
    return p


def pair_probe_accuracy(model, token_cache, train_pair, test_pair, args, seed, device):
    Xtr = block1_features(model, token_cache, train_pair, device)
    Xte = block1_features(model, token_cache, test_pair, device)
    Xtr, Xte = standardize(Xtr, Xte)

    ytr = torch.tensor(
        [0 if DATASET[i][0] == GPU else 1 for i in train_pair],
        dtype=torch.long, device=device,
    )
    yte = torch.tensor(
        [0 if DATASET[i][0] == GPU else 1 for i in test_pair],
        dtype=torch.long,
    )
    p = train_probe(Xtr, ytr, 2, args.probe_epochs, args.probe_lr, args.weight_decay, seed)
    with torch.no_grad():
        pred = p(Xte).argmax(-1).cpu()
    return float((pred == yte).float().mean().item())


def multiclass_metrics(y, pred):
    cm = [[0]*len(CLASSES) for _ in CLASSES]
    for g,p in zip(y.tolist(), pred.tolist()):
        cm[g][p] += 1
    correct = sum(cm[i][i] for i in range(len(CLASSES)))
    per = {}
    for i,c in enumerate(CLASSES):
        n = sum(cm[i]); h = cm[i][i]
        per[c] = (h,n,h/n if n else 0.0)
    return correct/len(y), correct, cm, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--base-lr", type=float, default=1e-5)
    ap.add_argument("--semantic-lr", type=float, default=3e-4)
    ap.add_argument("--lm-weight", type=float, default=1.0)
    ap.add_argument("--semantic-weight", type=float, default=0.35)
    ap.add_argument("--invariance-weight", type=float, default=0.10)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--probe-epochs", type=int, default=250)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--output-dir", default="results/semantic_supervised_base_v133")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    base, _ = LanguageModel.load_checkpoint(args.model, device=device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    order = family_order()
    token_cache = [encode(tok, base, prompt) for _,_,prompt in DATASET]
    all_idx = list(range(len(DATASET)))
    y_all = torch.tensor([CLASS_TO_ID[l] for l,_,_ in DATASET], dtype=torch.long)

    print("="*108)
    print(" LLM_GPU v1.3.3 Semantic-Supervised Base Representation Training")
    print("="*108)
    print("Device              :", device)
    print("Base                :", args.model)
    print("Trainable           : Embedding + Block1 + Block2")
    print("Frozen              : Blocks3-6 + FinalNorm + LM Head")
    print("Loss                : LM + semantic CE + family-invariance anchor loss")
    print("Weights             :", f"LM={args.lm_weight} semantic={args.semantic_weight} invariance={args.invariance_weight}")
    print("Base LR             :", args.base_lr)
    print("Semantic head LR    :", args.semantic_lr)
    print("Epochs/model        :", args.epochs)
    print("Primary baseline    : v1.3.1 leave-pair-out mean = 39.0%")
    print("Pairwise baseline   : v1.2.9 GPU-vs-CPU = 57.0%")
    print("6-class baseline    : v1.2.7 Block1 = 60.7%")
    print()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Exact 25-pair leave-target-pair-out evaluation.
    matrix = {}
    matrix_rows = []
    print("25-pair leave-target-pair-out evaluation")
    print("----------------------------------------")
    for gi,gfam in enumerate(order[GPU]):
        vals = []
        for ci,cfam in enumerate(order[CPU]):
            test_pair = [
                i for i,(l,f,_) in enumerate(DATASET)
                if (l == GPU and f == gfam) or (l == CPU and f == cfam)
            ]
            train_indices = [i for i in all_idx if i not in set(test_pair)]
            train_pair = [i for i in train_indices if DATASET[i][0] in (GPU,CPU)]

            model, _ = train_base(
                base, token_cache, train_indices, order,
                epochs=args.epochs, base_lr=args.base_lr, semantic_lr=args.semantic_lr,
                wd=args.weight_decay, lm_weight=args.lm_weight,
                sem_weight=args.semantic_weight, inv_weight=args.invariance_weight,
                seed=args.seed + gi*100 + ci,
            )
            acc = pair_probe_accuracy(
                model, token_cache, train_pair, test_pair,
                args, args.seed + 10000 + gi*100 + ci, device,
            )
            matrix[(gfam,cfam)] = acc
            matrix_rows.append([gfam,cfam,acc])
            vals.append(f"{acc:.0%}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"{gfam:14s}: " + " ".join(f"{v:>5s}" for v in vals))

    mvals = list(matrix.values())
    leave_mean = sum(mvals)/len(mvals)
    print(f"Leave-pair-out mean={leave_mean:.1%} min={min(mvals):.1%} max={max(mvals):.1%}")

    with (outdir/"leave_pair_out_accuracy.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["gpu_family\\cpu_family"] + order[CPU])
        for gfam in order[GPU]:
            w.writerow([gfam] + [matrix[(gfam,cfam)] for cfam in order[CPU]])

    # 2) Standard five-fold held-out pairwise + 6-class routing.
    pair_correct = 0
    pair_total = 0
    six_oof = torch.zeros((len(DATASET),len(CLASSES)), dtype=torch.float32)

    print()
    print("5-fold family-held-out semantic/routing evaluation")
    print("--------------------------------------------------")
    for k in range(5):
        test = [i for i,(l,f,_) in enumerate(DATASET) if f == order[l][k]]
        testset = set(test)
        train = [i for i in all_idx if i not in testset]
        train_pair = [i for i in train if DATASET[i][0] in (GPU,CPU)]
        test_pair = [i for i in test if DATASET[i][0] in (GPU,CPU)]

        model, _ = train_base(
            base, token_cache, train, order,
            epochs=args.epochs, base_lr=args.base_lr, semantic_lr=args.semantic_lr,
            wd=args.weight_decay, lm_weight=args.lm_weight,
            sem_weight=args.semantic_weight, inv_weight=args.invariance_weight,
            seed=args.seed + 1000*k,
        )

        # Pairwise probe.
        Xtrp = block1_features(model, token_cache, train_pair, device)
        Xtep = block1_features(model, token_cache, test_pair, device)
        Xtrp, Xtep = standardize(Xtrp, Xtep)
        ytrp = torch.tensor([0 if DATASET[i][0]==GPU else 1 for i in train_pair], dtype=torch.long, device=device)
        ytep = torch.tensor([0 if DATASET[i][0]==GPU else 1 for i in test_pair], dtype=torch.long)
        pp = train_probe(Xtrp,ytrp,2,args.probe_epochs,args.probe_lr,args.weight_decay,args.seed+2000*k)
        with torch.no_grad():
            predp = pp(Xtep).argmax(-1).cpu()
        pair_correct += int((predp == ytep).sum().item())
        pair_total += len(ytep)

        # 6-class probe.
        Xtr = block1_features(model, token_cache, train, device)
        Xte = block1_features(model, token_cache, test, device)
        Xtr,Xte = standardize(Xtr,Xte)
        probe = train_probe(
            Xtr, y_all[train].to(device), len(CLASSES),
            args.probe_epochs,args.probe_lr,args.weight_decay,args.seed+5000*k,
        )
        with torch.no_grad():
            six_oof[test] = torch.softmax(probe(Xte),dim=-1).cpu()

        del model, pp, probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pair_acc = pair_correct/pair_total
    pred = six_oof.argmax(-1)
    six_acc,six_correct,cm,per = multiclass_metrics(y_all,pred)

    print(f"GPU-vs-CPU pairwise : {pair_correct}/{pair_total} ({pair_acc:.1%})")
    print(f"6-class             : {six_correct}/300 ({six_acc:.1%})")
    for c in CLASSES:
        h,n,r = per[c]
        print(f"  {c:11s} {h:2d}/{n:2d} ({r:.1%})")

    with (outdir/"six_class_confusion.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["gold"]+CLASSES)
        for i,c in enumerate(CLASSES):
            w.writerow([c]+cm[i])

    with (outdir/"summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["leave_pair_out_mean",leave_mean])
        w.writerow(["leave_pair_out_min",min(mvals)])
        w.writerow(["leave_pair_out_max",max(mvals)])
        w.writerow(["gpu_cpu_pairwise_5fold",pair_acc])
        w.writerow(["six_class_accuracy",six_acc])
        for c in CLASSES:
            w.writerow([f"six_class_recall_{c}",per[c][2]])

    print()
    print("Decision")
    print("--------")
    print("Targets:")
    print("  leave-pair-out mean >= 55-60%  (baseline 39.0%)")
    print("  GPU-vs-CPU pairwise >= 70%     (baseline 57.0%)")
    print("  6-class >= 58-60%              (baseline 60.7%)")
    print("If the first two improve without collapsing the third, semantic supervision")
    print("is successfully changing base representation formation rather than only routing.")
    print("Fresh-v2 remains development data; untouched Fresh-v3 is still required later.")
    print("Output dir:", outdir)


if __name__ == "__main__":
    main()
