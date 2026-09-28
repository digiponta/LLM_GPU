# role_position_disentanglement_v142.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from semantic_role_position_v142 import DATASET, CLASSES, PAIR_KEYS
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
STAGES = ["block1","block2","block3","block4","block5","block6","final_norm"]
POOLINGS = [
    "final_token",
    "mean_all",
    "mean_first_half",
    "mean_second_half",
    "mean_first_second_concat",
]

class LinearHead(nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.fc = nn.Linear(d_in, n_out)

    def forward(self, x):
        return self.fc(x)

@torch.no_grad()
def extract_sequences(model, tok, prompt):
    device = next(model.parameters()).device
    ids = tok.encode(f"人: {prompt}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)

    x = model.embedding(token_ids)
    if model.position_embedding is not None:
        pos = torch.arange(token_ids.shape[1], device=device)
        x = x + model.position_embedding(pos).unsqueeze(0)

    out = {}
    for i,block in enumerate(model.blocks, start=1):
        x = block(x)
        out[f"block{i}"] = x[0].detach().cpu()

    x = model.final_norm(x)
    out["final_norm"] = x[0].detach().cpu()
    return out

def pool_sequence(seq, mode):
    T = seq.shape[0]
    split = max(1, T // 2)
    first = seq[:split]
    second = seq[split:] if split < T else seq[-1:]

    if mode == "final_token":
        return seq[-1]
    if mode == "mean_all":
        return seq.mean(0)
    if mode == "mean_first_half":
        return first.mean(0)
    if mode == "mean_second_half":
        return second.mean(0)
    if mode == "mean_first_second_concat":
        return torch.cat([first.mean(0), second.mean(0)], dim=0)
    raise ValueError(mode)

def standardize(Xtr, Xte):
    mean = Xtr.mean(0, keepdim=True)
    std = Xtr.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std, (Xte-mean)/std

def train_head(X, y, n_out, args, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    m = LinearHead(X.shape[1], n_out).to(X.device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        m.train()
        opt.zero_grad()
        loss = loss_fn(m(X), y)
        loss.backward()
        opt.step()

    m.eval()
    return m

def predict_split(X, y, n_out, train_idx, test_idx, args, seeds, offset):
    Xtr,Xte = standardize(X[train_idx], X[test_idx])
    probs = []
    for seed in seeds:
        m = train_head(Xtr, y[train_idx], n_out, args, seed+offset)
        with torch.no_grad():
            probs.append(torch.softmax(m(Xte), dim=-1).cpu())
    return torch.stack(probs).mean(0).argmax(-1)

def build_cv_folds():
    # Balance target class and surface order in each fold.
    folds = [[] for _ in range(5)]
    for c in CLASSES:
        for order in (0,1):
            idx = [i for i,s in enumerate(DATASET)
                   if s["target"] == c and s["surface_order"] == order]
            assert len(idx) == 10
            for k in range(5):
                folds[k].extend(idx[2*k:2*k+2])
    return folds

def cv_predict(X, y, n_out, folds, args, seeds, offset):
    oof = torch.zeros((len(y), n_out), dtype=torch.float32)
    for k,test in enumerate(folds):
        test_set = set(test)
        train = [i for i in range(len(y)) if i not in test_set]
        pred = predict_split(X, y, n_out, train, test, args, seeds, offset+1000*k)
        # Need probabilities for exact assignment; use one-hot from averaged argmax here.
        oof[test] = torch.nn.functional.one_hot(pred, num_classes=n_out).float()
    return oof.argmax(-1)

def accuracy(pred, gold):
    return float((pred == gold.cpu()).float().mean().item())

def target_from_pair_role(pair_id, role_direction):
    a,b = PAIR_KEYS[int(pair_id)]
    target = a if int(role_direction) == 0 else b
    return CLASS_TO_ID[target]

def evaluate_cross_order(X, y_role, y_pair, y_target, train_order, test_order, args, seeds, offset):
    train = [i for i,s in enumerate(DATASET) if s["surface_order"] == train_order]
    test = [i for i,s in enumerate(DATASET) if s["surface_order"] == test_order]

    pr = predict_split(X, y_role, 2, train, test, args, seeds, offset)
    pp = predict_split(X, y_pair, 15, train, test, args, seeds, offset+5000)

    gr = y_role[test].cpu()
    gp = y_pair[test].cpu()
    gt = y_target[test].cpu()

    recon = torch.tensor([
        target_from_pair_role(p,r) for p,r in zip(pp.tolist(),pr.tolist())
    ], dtype=torch.long)

    return {
        "role": float((pr == gr).float().mean().item()),
        "pair": float((pp == gp).float().mean().item()),
        "target": float((recon == gt).float().mean().item()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--output-dir", default="results/role_position_disentanglement_v142")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer.load(args.tokenizer)
    model,_ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("="*118)
    print(" LLM_GPU v1.4.2 Role-Position Disentanglement Diagnostic")
    print("="*118)
    print("Device         :", device)
    print("Base           :", args.model)
    print("Dataset        : 120 prompts = 30 topics x 2 target roles x 2 surface orders")
    print("Role label     : canonical-first vs canonical-second concept is target")
    print("Position label : target-first vs contrast-first surface order")
    print("Stages         :", ", ".join(STAGES))
    print("Pooling        :", ", ".join(POOLINGS))
    print("Base model     : completely frozen")
    print()

    seqs = {s:[] for s in STAGES}
    print("Extracting token-wise hidden states...")
    for i,sample in enumerate(DATASET, start=1):
        d = extract_sequences(model, tok, sample["prompt"])
        for stage in STAGES:
            seqs[stage].append(d[stage])
        if i % 20 == 0:
            print(f"  encoded {i}/120")

    y_role = torch.tensor([s["role_direction"] for s in DATASET], dtype=torch.long, device=device)
    y_position = torch.tensor([s["surface_order"] for s in DATASET], dtype=torch.long, device=device)
    y_pair = torch.tensor([s["pair_id"] for s in DATASET], dtype=torch.long, device=device)
    y_target = torch.tensor([CLASS_TO_ID[s["target"]] for s in DATASET], dtype=torch.long)

    folds = build_cv_folds()
    rows = []

    for stage in STAGES:
        for pooling in POOLINGS:
            X = torch.stack([pool_sequence(s,pooling) for s in seqs[stage]]).to(device)
            baseoff = 10000*STAGES.index(stage) + 500*POOLINGS.index(pooling)

            pred_role = cv_predict(X,y_role,2,folds,args,seeds,baseoff+10)
            pred_pos = cv_predict(X,y_position,2,folds,args,seeds,baseoff+20)
            pred_pair = cv_predict(X,y_pair,15,folds,args,seeds,baseoff+30)

            role_cv = accuracy(pred_role,y_role)
            pos_cv = accuracy(pred_pos,y_position)
            pair_cv = accuracy(pred_pair,y_pair)

            recon = torch.tensor([
                target_from_pair_role(p,r)
                for p,r in zip(pred_pair.tolist(),pred_role.tolist())
            ], dtype=torch.long)
            target_cv = float((recon == y_target).float().mean().item())

            t2c = evaluate_cross_order(
                X,y_role,y_pair,y_target,0,1,args,seeds,baseoff+100
            )
            c2t = evaluate_cross_order(
                X,y_role,y_pair,y_target,1,0,args,seeds,baseoff+200
            )
            cross_role = (t2c["role"] + c2t["role"]) / 2
            cross_pair = (t2c["pair"] + c2t["pair"]) / 2
            cross_target = (t2c["target"] + c2t["target"]) / 2

            print(
                f"{stage:10s} {pooling:24s} "
                f"roleCV={role_cv:6.1%} posCV={pos_cv:6.1%} pairCV={pair_cv:6.1%} "
                f"targetCV={target_cv:6.1%} crossRole={cross_role:6.1%} "
                f"crossPair={cross_pair:6.1%} crossTarget={cross_target:6.1%}"
            )

            rows.append([
                stage,pooling,X.shape[1],
                role_cv,pos_cv,pair_cv,target_cv,
                t2c["role"],c2t["role"],cross_role,
                t2c["pair"],c2t["pair"],cross_pair,
                t2c["target"],c2t["target"],cross_target,
            ])

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir/"summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "stage","pooling","feature_dim",
            "role_cv_accuracy","position_cv_accuracy","pair_cv_accuracy","target_cv_accuracy",
            "role_targetfirst_to_contrastfirst","role_contrastfirst_to_targetfirst","role_cross_order_mean",
            "pair_targetfirst_to_contrastfirst","pair_contrastfirst_to_targetfirst","pair_cross_order_mean",
            "target_targetfirst_to_contrastfirst","target_contrastfirst_to_targetfirst","target_cross_order_mean",
        ])
        w.writerows(rows)

    best_role_cv = max(rows, key=lambda r:r[3])
    best_cross_role = max(rows, key=lambda r:r[9])
    best_cross_target = max(rows, key=lambda r:r[15])

    print()
    print("Best mixed-order role CV")
    print("------------------------")
    print(f"{best_role_cv[0]} / {best_role_cv[1]} : {best_role_cv[3]:.1%}")

    print()
    print("Best cross-position role generalization")
    print("---------------------------------------")
    print(f"{best_cross_role[0]} / {best_cross_role[1]} : {best_cross_role[9]:.1%}")

    print()
    print("Best cross-position reconstructed target")
    print("----------------------------------------")
    print(f"{best_cross_target[0]} / {best_cross_target[1]} : {best_cross_target[15]:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.1 first-half direction peak : 63.3%")
    print("Binary chance                    : 50.0%")
    print()
    print("Interpretation")
    print("--------------")
    print("High mixed-order role CV but near-chance cross-position role means v1.4.1 mainly")
    print("used surface position as a shortcut.")
    print("Cross-position role clearly above 50% means target/contrast role survives word-order")
    print("changes and is genuinely represented beyond simple first/second position.")
    print("Position CV is reported separately to quantify how strongly surface order itself is encoded.")
    print("Output dir:", outdir)

if __name__ == "__main__":
    main()
