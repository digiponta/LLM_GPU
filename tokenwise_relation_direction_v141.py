# tokenwise_relation_direction_v141.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from semantic_symmetric_comparison_v137 import CLASSES, PAIR_META
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
STAGES = ["block1","block2","block3","block4","block5","block6","final_norm"]
POOLINGS = [
    "final_token",
    "mean_all",
    "max_all",
    "mean_first_half",
    "mean_second_half",
    "mean_first_second_concat",
]

PAIR_KEYS = []
PAIR_TO_ID = {}
SAMPLES = []

for pair_id, topic_id, a, b, pa, pb in PAIR_META:
    key = tuple(sorted((a,b), key=lambda x: CLASS_TO_ID[x]))
    if key not in PAIR_TO_ID:
        PAIR_TO_ID[key] = len(PAIR_KEYS)
        PAIR_KEYS.append(key)

    SAMPLES.append({
        "prompt": pa,
        "target": a,
        "contrast": b,
        "pair_id": PAIR_TO_ID[key],
        "direction": 0 if a == key[0] else 1,
    })
    SAMPLES.append({
        "prompt": pb,
        "target": b,
        "contrast": a,
        "pair_id": PAIR_TO_ID[key],
        "direction": 0 if b == key[0] else 1,
    })

assert len(SAMPLES) == 60


class LinearHead(nn.Module):
    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_out)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def extract_sequences(model: LanguageModel, tok: Tokenizer, prompt: str):
    device = next(model.parameters()).device
    ids = tok.encode(f"人: {prompt}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)

    x = model.embedding(token_ids)
    if model.position_embedding is not None:
        pos = torch.arange(token_ids.shape[1], device=device)
        x = x + model.position_embedding(pos).unsqueeze(0)

    out = {}
    for i, block in enumerate(model.blocks, start=1):
        x = block(x)
        out[f"block{i}"] = x[0].detach().cpu()

    x = model.final_norm(x)
    out["final_norm"] = x[0].detach().cpu()
    return out


def pool_sequence(seq: torch.Tensor, mode: str):
    # seq: [T,D]
    T = seq.shape[0]
    split = max(1, T // 2)
    first = seq[:split]
    second = seq[split:] if split < T else seq[-1:]

    if mode == "final_token":
        return seq[-1]
    if mode == "mean_all":
        return seq.mean(0)
    if mode == "max_all":
        return seq.max(0).values
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
    return (Xtr - mean) / std, (Xte - mean) / std


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
    folds = [[] for _ in range(5)]
    for c in CLASSES:
        idx = [i for i,s in enumerate(SAMPLES) if s["target"] == c]
        assert len(idx) == 10
        for k in range(5):
            folds[k].extend(idx[2*k:2*k+2])
    return folds


def oof_predict(X, y, n_out, folds, args, seeds, offset):
    oof = torch.zeros((len(y), n_out), dtype=torch.float32)

    for k,test in enumerate(folds):
        train = [i for i in range(len(y)) if i not in set(test)]
        Xtr, Xte = standardize(X[train], X[test])
        probs = []
        for seed in seeds:
            m = train_head(
                Xtr, y[train], n_out,
                args.epochs, args.lr, args.weight_decay,
                seed + offset + 1000*k,
            )
            with torch.no_grad():
                probs.append(torch.softmax(m(Xte), dim=-1).cpu())
        oof[test] = torch.stack(probs).mean(0)

    return oof.argmax(-1), oof


def target_from_pair_direction(pair_id: int, direction: int) -> int:
    a,b = PAIR_KEYS[pair_id]
    return CLASS_TO_ID[a if direction == 0 else b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--output-dir", default="results/tokenwise_relation_direction_v141")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer.load(args.tokenizer)
    model,_ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("="*116)
    print(" LLM_GPU v1.4.1 Token-wise Relation Direction Diagnostic")
    print("="*116)
    print("Device        :", device)
    print("Base          :", args.model)
    print("Dataset       : v1.3.7 symmetric reciprocal comparison, 60 prompts")
    print("Stages        :", ", ".join(STAGES))
    print("Pooling modes :", ", ".join(POOLINGS))
    print("Task          : explicit direction 2-way")
    print("Auxiliary     : pair 15-way + reconstructed target")
    print("Evaluation    : 5-fold target-class-stratified CV")
    print("Seeds/fold    :", seeds)
    print("Base model    : completely frozen")
    print()

    seqs = {stage:[] for stage in STAGES}
    print("Extracting token-wise hidden sequences...")
    for i,sample in enumerate(SAMPLES, start=1):
        sdict = extract_sequences(model, tok, sample["prompt"])
        for stage in STAGES:
            seqs[stage].append(sdict[stage])
        if i % 10 == 0:
            print(f"  encoded {i}/60")

    y_pair = torch.tensor([s["pair_id"] for s in SAMPLES], dtype=torch.long, device=device)
    y_dir = torch.tensor([s["direction"] for s in SAMPLES], dtype=torch.long, device=device)
    y_target = torch.tensor([CLASS_TO_ID[s["target"]] for s in SAMPLES], dtype=torch.long)

    folds = build_folds()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for stage in STAGES:
        for pooling in POOLINGS:
            X = torch.stack([pool_sequence(s, pooling) for s in seqs[stage]]).to(device)

            pred_dir,_ = oof_predict(
                X, y_dir, 2, folds, args, seeds,
                offset=10000 + 100*POOLINGS.index(pooling) + 1000*STAGES.index(stage),
            )
            pred_pair,_ = oof_predict(
                X, y_pair, 15, folds, args, seeds,
                offset=30000 + 100*POOLINGS.index(pooling) + 1000*STAGES.index(stage),
            )

            yd = y_dir.cpu()
            yp = y_pair.cpu()

            dir_acc = float((pred_dir == yd).float().mean().item())
            pair_acc = float((pred_pair == yp).float().mean().item())

            recon_target = torch.tensor([
                target_from_pair_direction(int(p), int(d))
                for p,d in zip(pred_pair.tolist(), pred_dir.tolist())
            ])
            target_acc = float((recon_target == y_target).float().mean().item())

            pair_mask = pred_pair == yp
            dir_given_pair = float(
                (pred_dir[pair_mask] == yd[pair_mask]).float().mean().item()
            ) if bool(pair_mask.any()) else 0.0

            # Gold pair + predicted direction isolates direction information.
            oracle_pair_target = torch.tensor([
                target_from_pair_direction(int(p), int(d))
                for p,d in zip(yp.tolist(), pred_dir.tolist())
            ])
            gold_pair_pred_dir_target = float(
                (oracle_pair_target == y_target).float().mean().item()
            )

            print(
                f"{stage:10s} {pooling:24s} "
                f"dir={dir_acc:6.1%} pair={pair_acc:6.1%} "
                f"target={target_acc:6.1%} dir|pair={dir_given_pair:6.1%}"
            )

            rows.append([
                stage, pooling, X.shape[1],
                dir_acc, pair_acc, target_acc, dir_given_pair,
                gold_pair_pred_dir_target,
            ])

    with (outdir/"summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "stage","pooling","feature_dim",
            "direction_accuracy","pair_accuracy",
            "reconstructed_target_accuracy",
            "direction_given_pair_accuracy",
            "gold_pair_pred_direction_target_accuracy",
        ])
        w.writerows(rows)

    best_dir = max(rows, key=lambda r:r[3])
    best_target = max(rows, key=lambda r:r[5])

    print()
    print("Best direction representation")
    print("-----------------------------")
    print(
        f"stage={best_dir[0]} pooling={best_dir[1]} "
        f"direction={best_dir[3]:.1%} pair={best_dir[4]:.1%} "
        f"target={best_dir[5]:.1%}"
    )

    print()
    print("Best reconstructed target")
    print("-------------------------")
    print(
        f"stage={best_target[0]} pooling={best_target[1]} "
        f"direction={best_target[3]:.1%} pair={best_target[4]:.1%} "
        f"target={best_target[5]:.1%}"
    )

    print()
    print("Reference")
    print("---------")
    print("v1.4.0 best final-token explicit direction : 21.7% (Block3)")
    print("Chance for direction                      : 50.0%")
    print()
    print("Interpretation")
    print("--------------")
    print("If token-wise pooling raises direction above chance, relation direction exists in")
    print("distributed token states but is lost by final-token compression.")
    print("If every pooling mode remains at/below chance, the frozen base representation does")
    print("not contain a stable linearly decodable target/contrast direction signal.")
    print("Output dir:", outdir)


if __name__ == "__main__":
    main()
