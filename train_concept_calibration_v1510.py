# train_concept_calibration_v1510.py
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from chat import DEFAULT_MODEL, DEFAULT_TOKENIZER, semantic_vector

CONCEPTS = ["AI", "LLM", "CPU", "GPU", "CUDA"]

TRAIN_SAMPLES = {
    "AI": [
        "AIは人工知能です",
        "人工知能は知的な処理を機械で実現する技術です",
        "AIは推論や認識を行う計算技術です",
        "人工知能には機械学習や推論技術が含まれます",
        "AIは人間の知的能力を模倣する技術です",
        "人工知能システムは入力から判断や予測を行います",
    ],
    "LLM": [
        "LLMは大規模言語モデルです",
        "大規模言語モデルは大量の文章から言語パターンを学びます",
        "LLMは文章を理解し生成する言語モデルです",
        "言語モデルは次の語を予測しながら文章を生成します",
        "LLMはTransformerを使った大規模な言語処理モデルです",
        "大規模言語モデルは対話や要約に利用されます",
    ],
    "CPU": [
        "CPUは中央処理装置です",
        "CPUは汎用的な命令実行を担当します",
        "中央処理装置はプログラムの命令を順番に実行します",
        "CPUは制御処理と汎用計算を行います",
        "CPUはコンピュータ全体の処理を制御します",
        "中央処理装置は分岐や逐次処理を得意とします",
    ],
    "GPU": [
        "GPUは画像処理装置です",
        "GPUは大量の並列計算を得意とします",
        "GPUは多数の演算を同時に処理します",
        "画像処理装置は並列演算に強いです",
        "GPUはグラフィックスと数値計算に使われます",
        "GPUは多数のコアで同じ種類の計算を並列実行します",
    ],
    "CUDA": [
        "CUDAはNVIDIAのGPU向け汎用計算技術です",
        "CUDAはGPU上でプログラムを実行するための基盤です",
        "CUDAはNVIDIA GPUを計算用途に利用する技術です",
        "CUDAプログラムはGPUカーネルを実行します",
        "CUDAはGPU並列計算のソフトウェア環境です",
        "CUDAはNVIDIAが提供する並列計算プラットフォームです",
    ],
}

HOLDOUT_SAMPLES = {
    "AI": [
        "人工知能とは知的な判断を機械に行わせる技術です",
        "機械が認識や推論を行う技術",
    ],
    "LLM": [
        "大量のテキストで学習した大規模言語モデル",
        "文章の理解と生成を行う言語モデル",
    ],
    "CPU": [
        "汎用処理を担当する中央演算装置",
        "命令実行と制御を行うプロセッサ",
    ],
    "GPU": [
        "並列演算を得意とする画像処理プロセッサ",
        "多数の計算を同時実行する演算装置",
    ],
    "CUDA": [
        "NVIDIA GPU向けの並列計算環境",
        "GPUで汎用計算を実行するNVIDIAの技術",
    ],
}

HARD_NEGATIVE_PAIRS = [
    ("AI", "LLM"),
    ("CPU", "GPU"),
    ("GPU", "CUDA"),
]


class ConceptProjection(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


@torch.no_grad()
def encode_samples(model, tokenizer, samples, device):
    xs, ys, texts = [], [], []
    for label_idx, label in enumerate(CONCEPTS):
        for text in samples[label]:
            xs.append(semantic_vector(model, tokenizer, text).detach())
            ys.append(label_idx)
            texts.append((label, text))
    return torch.stack(xs).to(device), torch.tensor(ys, dtype=torch.long, device=device), texts


def centroid_logits(z: torch.Tensor, centroids: torch.Tensor, temperature: float) -> torch.Tensor:
    return z @ F.normalize(centroids, dim=-1).T / temperature


def class_centroids(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    cs = []
    for i in range(len(CONCEPTS)):
        cs.append(F.normalize(z[y == i].mean(dim=0), dim=0))
    return torch.stack(cs)


def evaluate_space(x, y, projection=None):
    with torch.no_grad():
        z = F.normalize(x, dim=-1) if projection is None else projection(x)
        centroids = class_centroids(z, y)
        logits = z @ centroids.T
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        margins = []
        for i in range(len(y)):
            own = logits[i, y[i]]
            others = torch.cat((logits[i, :y[i]], logits[i, y[i]+1:]))
            margins.append(float((own - others.max()).item()))
        return acc, sum(margins) / len(margins), centroids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.08)
    ap.add_argument("--hard-negative-weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default="model/concept-calibration-v1510.pt")
    ap.add_argument("--results", default="results/concept_calibration_v1510")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    base, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    train_x, train_y, _ = encode_samples(base, tokenizer, TRAIN_SAMPLES, device)
    hold_x, hold_y, hold_meta = encode_samples(base, tokenizer, HOLDOUT_SAMPLES, device)

    projection = ConceptProjection(base.d_model).to(device)
    class_weights = nn.Parameter(torch.randn(len(CONCEPTS), 64, device=device))
    nn.init.normal_(class_weights, std=0.02)
    opt = torch.optim.AdamW(list(projection.parameters()) + [class_weights], lr=args.lr)

    raw_train_acc, raw_train_margin, _ = evaluate_space(train_x, train_y)
    raw_hold_acc, raw_hold_margin, _ = evaluate_space(hold_x, hold_y)

    pair_ids = [(CONCEPTS.index(a), CONCEPTS.index(b)) for a, b in HARD_NEGATIVE_PAIRS]

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        projection.train()
        z = projection(train_x)
        w = F.normalize(class_weights, dim=-1)
        logits = z @ w.T / args.temperature
        ce = F.cross_entropy(logits, train_y)

        # Explicitly separate the known hard-negative concept pairs.
        hard = torch.tensor(0.0, device=device)
        for a, b in pair_ids:
            ca = F.normalize(z[train_y == a].mean(dim=0), dim=0)
            cb = F.normalize(z[train_y == b].mean(dim=0), dim=0)
            hard = hard + F.relu(torch.dot(ca, cb) - 0.65)
        hard = hard / len(pair_ids)

        loss = ce + args.hard_negative_weight * hard
        opt.zero_grad()
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                "projection": {k: v.detach().cpu().clone() for k, v in projection.state_dict().items()},
                "class_weights": class_weights.detach().cpu().clone(),
                "epoch": epoch,
            }

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:4d} loss={loss.item():.6f} ce={ce.item():.6f} hard={hard.item():.6f}")

    projection.load_state_dict(best_state["projection"])
    projection.eval()

    cal_train_acc, cal_train_margin, train_centroids = evaluate_space(train_x, train_y, projection)
    with torch.no_grad():
        hold_z = projection(hold_x)
        logits = hold_z @ train_centroids.T
        hold_pred = logits.argmax(dim=1)
        cal_hold_acc = (hold_pred == hold_y).float().mean().item()
        hold_margins = []
        for i in range(len(hold_y)):
            own = logits[i, hold_y[i]]
            others = torch.cat((logits[i, :hold_y[i]], logits[i, hold_y[i]+1:]))
            hold_margins.append(float((own - others.max()).item()))
        cal_hold_margin = sum(hold_margins) / len(hold_margins)

    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    cp = Path(args.checkpoint)
    cp.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "version": "v1.5.10",
        "concepts": CONCEPTS,
        "input_dim": base.d_model,
        "hidden_dim": 128,
        "output_dim": 64,
        "projection_state": projection.state_dict(),
        "centroids": train_centroids.detach().cpu(),
        "base_model": args.model,
        "base_checkpoint_loss": checkpoint.get("loss"),
        "best_epoch": best_state["epoch"],
        "best_loss": best_loss,
    }, cp)

    with (out / "holdout.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "text", "predicted", "correct", "margin"])
        for i, (label, text) in enumerate(hold_meta):
            pred = CONCEPTS[int(hold_pred[i].item())]
            w.writerow([label, text, pred, int(pred == label), hold_margins[i]])

    print()
    print("=" * 76)
    print(" LLM_GPU v1.5.10 Contrastive Concept Calibration")
    print("=" * 76)
    print("Device                  :", device)
    print("Base checkpoint loss    :", checkpoint.get("loss"))
    print("Train samples           :", len(train_y))
    print("Holdout samples         :", len(hold_y))
    print("Concepts                :", ", ".join(CONCEPTS))
    print()
    print(f"Raw train accuracy      : {raw_train_acc*100:.1f}%")
    print(f"Raw train mean margin   : {raw_train_margin:.6f}")
    print(f"Raw holdout accuracy    : {raw_hold_acc*100:.1f}%")
    print(f"Raw holdout mean margin : {raw_hold_margin:.6f}")
    print()
    print(f"Cal train accuracy      : {cal_train_acc*100:.1f}%")
    print(f"Cal train mean margin   : {cal_train_margin:.6f}")
    print(f"Cal holdout accuracy    : {cal_hold_acc*100:.1f}%")
    print(f"Cal holdout mean margin : {cal_hold_margin:.6f}")
    print()
    print("Checkpoint              :", cp)
    print("Holdout detail          :", out / "holdout.csv")


if __name__ == "__main__":
    main()
