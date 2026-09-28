# explicit_relation_direction_v140.py
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
STAGES = ["embedding","block1","block2","block3","block4","block5","block6","final_norm"]

PAIR_KEYS = []
PAIR_TO_ID = {}
SAMPLES = []

for pair_id, topic_id, a, b, pa, pb in PAIR_META:
    key = tuple(sorted((a,b), key=lambda x: CLASS_TO_ID[x]))
    if key not in PAIR_TO_ID:
        PAIR_TO_ID[key] = len(PAIR_KEYS)
        PAIR_KEYS.append(key)

    # direction=0 => target is canonical first element of sorted pair
    # direction=1 => target is canonical second element
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
    folds = [[] for _ in range(5)]
    for c in CLASSES:
        idx = [i for i,s in enumerate(SAMPLES) if s["target"] == c]
        assert len(idx) == 10
        for k in range(5):
            folds[k].extend(idx[2*k:2*k+2])
    return folds


def oof_predict(X, y, n_out, folds, args, seeds, offset):
    oof = torch.zeros((len(y), n_out), dtype=torch.float32)

    for k, test in enumerate(folds):
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
    target = a if direction == 0 else b
    return CLASS_TO_ID[target]


def contrast_from_pair_direction(pair_id: int, direction: int) -> int:
    a,b = PAIR_KEYS[pair_id]
    contrast = b if direction == 0 else a
    return CLASS_TO_ID[contrast]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--output-dir", default="results/explicit_relation_direction_v140")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("="*112)
    print(" LLM_GPU v1.4.0 Explicit Relation-Direction Head Diagnostic")
    print("="*112)
    print("Device            :", device)
    print("Base              :", args.model)
    print("Dataset           : v1.3.7 symmetric reciprocal comparison, 60 prompts")
    print("Pair head         : 15-way unordered concept pair")
    print("Direction head    : 2-way canonical-first vs canonical-second target")
    print("Reconstruction    : pair + direction -> target/contrast")
    print("Stages            :", ", ".join(STAGES))
    print("Evaluation        : 5-fold target-class-stratified CV")
    print("Seeds/fold        :", seeds)
    print("Base model        : completely frozen")
    print()

    feats = {s:[] for s in STAGES}
    print("Extracting hidden states...")
    for i,sample in enumerate(SAMPLES, start=1):
        vecs = extract_stages(model, tok, sample["prompt"])
        for stage in STAGES:
            feats[stage].append(vecs[stage])
        if i % 10 == 0:
            print(f"  encoded {i}/60")

    feats = {s:torch.stack(v).to(device) for s,v in feats.items()}

    y_pair = torch.tensor([s["pair_id"] for s in SAMPLES], dtype=torch.long, device=device)
    y_dir = torch.tensor([s["direction"] for s in SAMPLES], dtype=torch.long, device=device)
    y_target = torch.tensor([CLASS_TO_ID[s["target"]] for s in SAMPLES], dtype=torch.long)
    y_contrast = torch.tensor([CLASS_TO_ID[s["contrast"]] for s in SAMPLES], dtype=torch.long)

    folds = build_folds()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for stage in STAGES:
        X = feats[stage]

        pred_pair, _ = oof_predict(X, y_pair, 15, folds, args, seeds, 0)
        pred_dir, _ = oof_predict(X, y_dir, 2, folds, args, seeds, 10000)

        yp = y_pair.cpu()
        yd = y_dir.cpu()

        pair_acc = float((pred_pair == yp).float().mean().item())
        dir_acc = float((pred_dir == yd).float().mean().item())

        recon_target = torch.tensor([
            target_from_pair_direction(int(p), int(d))
            for p,d in zip(pred_pair.tolist(), pred_dir.tolist())
        ], dtype=torch.long)
        recon_contrast = torch.tensor([
            contrast_from_pair_direction(int(p), int(d))
            for p,d in zip(pred_pair.tolist(), pred_dir.tolist())
        ], dtype=torch.long)

        target_acc = float((recon_target == y_target).float().mean().item())
        contrast_acc = float((recon_contrast == y_contrast).float().mean().item())
        ordered_acc = float(((recon_target == y_target) & (recon_contrast == y_contrast)).float().mean().item())

        # Oracle decomposition.
        oracle_pair_target = torch.tensor([
            target_from_pair_direction(int(p), int(d))
            for p,d in zip(yp.tolist(), pred_dir.tolist())
        ], dtype=torch.long)
        oracle_direction_target = torch.tensor([
            target_from_pair_direction(int(p), int(d))
            for p,d in zip(pred_pair.tolist(), yd.tolist())
        ], dtype=torch.long)

        oracle_pair_acc = float((oracle_pair_target == y_target).float().mean().item())
        oracle_direction_acc = float((oracle_direction_target == y_target).float().mean().item())

        pair_mask = pred_pair == yp
        if bool(pair_mask.any()):
            dir_given_pair = float((pred_dir[pair_mask] == yd[pair_mask]).float().mean().item())
        else:
            dir_given_pair = 0.0

        print()
        print(stage.upper())
        print("-"*88)
        print(f"pair accuracy                         : {pair_acc:.1%}")
        print(f"direction accuracy                    : {dir_acc:.1%}")
        print(f"reconstructed target                  : {target_acc:.1%}")
        print(f"reconstructed contrast                : {contrast_acc:.1%}")
        print(f"ordered target+contrast               : {ordered_acc:.1%}")
        print(f"direction given predicted pair correct: {dir_given_pair:.1%}")
        print(f"oracle gold pair + predicted direction: {oracle_pair_acc:.1%}")
        print(f"predicted pair + oracle gold direction: {oracle_direction_acc:.1%}")

        rows.append([
            stage, pair_acc, dir_acc, target_acc, contrast_acc, ordered_acc,
            dir_given_pair, oracle_pair_acc, oracle_direction_acc,
        ])

    with (outdir/"summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "stage",
            "pair_accuracy",
            "direction_accuracy",
            "reconstructed_target_accuracy",
            "reconstructed_contrast_accuracy",
            "ordered_target_contrast_accuracy",
            "direction_given_pair_accuracy",
            "gold_pair_pred_direction_target_accuracy",
            "pred_pair_gold_direction_target_accuracy",
        ])
        w.writerows(rows)

    print()
    print("Structured relational comparison")
    print("--------------------------------")
    for r in rows:
        print(
            f"{r[0]:10s} pair={r[1]:.1%} dir={r[2]:.1%} "
            f"target={r[3]:.1%} ordered={r[5]:.1%} "
            f"dir|pair={r[6]:.1%} goldPair+dir={r[7]:.1%} pair+goldDir={r[8]:.1%}"
        )

    best_target = max(rows, key=lambda r:r[3])
    best_dir = max(rows, key=lambda r:r[2])

    print()
    print(f"Best reconstructed target stage: {best_target[0]} {best_target[3]:.1%}")
    print(f"Best explicit direction stage   : {best_dir[0]} {best_dir[2]:.1%}")
    print()
    print("Interpretation")
    print("--------------")
    print("If direction is well above 50% while pair remains strong, explicit role encoding")
    print("can recover relational semantics from the frozen representation.")
    print("If direction stays near 50%, role direction itself is not linearly encoded.")
    print("Oracle columns identify whether pair identity or direction is the dominant limit.")
    print("Output dir:", outdir)


if __name__ == "__main__":
    main()
