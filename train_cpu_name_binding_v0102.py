# train_cpu_name_binding_v0102.py
#
# v0.10.2 CPU Name-Meaning Binding
#
# Exact G05 is excluded from training.
# Base model + v0.8 semantic adapter are frozen.
# Only a small cross-lingual binding projection is trained.

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from cpu_name_binding_v0102 import (
    CPUNameBindingProjection,
    save_cpu_name_binding_checkpoint,
)
from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v08 import load_semantic_adapter_v08_checkpoint
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
SEED = 42


CPU_PHRASES = [
    "CPU",
    "Central Processing Unit",
    "CPUはCentral Processing Unitの略です。",
    "中央処理装置",
    "中央演算処理装置",
    "CPUは日本語で中央処理装置と呼ばれます。",
    "CPUは中央演算処理装置とも表現されます。",
    "Centralは中央を意味します。",
    "Processingは処理を意味します。",
    "Unitは装置を意味します。",
    "Central Processing Unitは中央処理装置です。",
    "Central Processing Unitは中央演算処理装置です。",
    "中央で処理を行う装置はCPUです。",
    "コンピュータの中央で命令処理を担う装置はCPUです。",
]

GPU_PHRASES = [
    "GPU",
    "Graphics Processing Unit",
    "画像処理装置",
    "GPUは並列計算を得意とします。",
    "多数の同種計算を並列処理する装置です。",
    "Graphicsは画像を意味します。",
]


def parse_args():
    p = argparse.ArgumentParser(description="Train CPU Name Binding v0.10.2.")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.30)
    p.add_argument("--binding-dim", type=int, default=64)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


@torch.no_grad()
def encode(model, adapter, tokenizer, text):
    device = next(model.parameters()).device
    ids = tokenizer.encode(f"人: {text}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = model.forward_hidden(x)[:, -1, :]
    return adapter(hidden)[0].detach()


def exact_dev_overlap():
    dev = {str(case["prompt"]) for case in CASES}
    return sorted(x for x in CPU_PHRASES + GPU_PHRASES if x in dev)


def pairwise_positive_loss(z):
    sim = z @ z.T
    n = z.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    return (1.0 - sim[mask]).mean()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model, args.semantic_adapter):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    overlap = exact_dev_overlap()
    if overlap:
        raise RuntimeError("Exact fixed-DEV overlap: " + repr(overlap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    adapter, heads, hierarchy_head, semantic_checkpoint = (
        load_semantic_adapter_v08_checkpoint(args.semantic_adapter, device)
    )

    model.eval()
    adapter.eval()
    heads.eval()
    hierarchy_head.eval()
    for module in (model, adapter, heads, hierarchy_head):
        for p in module.parameters():
            p.requires_grad_(False)

    cpu_hidden = torch.stack(
        [encode(model, adapter, tokenizer, x) for x in CPU_PHRASES], dim=0
    )
    gpu_hidden = torch.stack(
        [encode(model, adapter, tokenizer, x) for x in GPU_PHRASES], dim=0
    )

    projection = CPUNameBindingProjection(
        d_model=model.d_model,
        binding_dim=args.binding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.lr, weight_decay=0.01)

    print()
    print("====================================================")
    print(" CPU Name-Meaning Binding v0.10.2 Training")
    print("====================================================")
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss :", base_checkpoint.get("loss"))
    print("Semantic adapter loss:", semantic_checkpoint.get("loss"))
    print("Base model           : frozen")
    print("Semantic adapter     : frozen")
    print("Binding projection   : trainable")
    print("CPU phrases          :", len(CPU_PHRASES))
    print("GPU distractors      :", len(GPU_PHRASES))
    print("Exact DEV overlap    :", len(overlap))
    print("Binding dim          :", args.binding_dim)
    print("Margin               :", args.margin)
    print()

    best = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        projection.train()
        optimizer.zero_grad(set_to_none=True)

        z_cpu = projection(cpu_hidden)
        z_gpu = projection(gpu_hidden)

        positive = pairwise_positive_loss(z_cpu)

        cpu_centroid = F.normalize(z_cpu.mean(dim=0), dim=0)
        gpu_centroid = F.normalize(z_gpu.mean(dim=0), dim=0)
        cross = z_cpu @ gpu_centroid
        separation = F.relu(args.margin + cross - (z_cpu @ cpu_centroid)).mean()

        centroid_sep = F.relu(
            args.margin + torch.dot(cpu_centroid, gpu_centroid) - 0.20
        )

        total = positive + separation + 0.5 * centroid_sep
        total.backward()
        torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
        optimizer.step()

        value = float(total.item())
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"| loss={value:.4f} pos={positive.item():.4f} "
                f"sep={separation.item():.4f} csep={centroid_sep.item():.4f}"
            )

        if value < best - 1e-6:
            best = value
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in projection.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    projection.load_state_dict(best_state)
    save_cpu_name_binding_checkpoint(
        args.output,
        projection,
        epoch=best_epoch,
        loss=best,
        base_model=args.model,
        semantic_adapter=args.semantic_adapter,
        learning_rate=args.lr,
        margin=args.margin,
    )

    print()
    print("CPU Name-Meaning Binding training completed.")
    print("Best epoch       :", best_epoch)
    print("Best loss        :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
