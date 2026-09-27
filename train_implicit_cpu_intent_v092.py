# train_implicit_cpu_intent_v092.py
#
# LLM_GPU v0.9.2
# Implicit CPU Intent Correction
#
# Freeze v0.8 LM. Fine-tune only the existing v0.8 intent head on curated
# implicit CPU/GPU paraphrases plus protected replay. Exact fixed G05 prompts
# are excluded.

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluate_partial_intent_v09 import CASES
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INTENT_HEAD = "model/model-gpu-v0.8-intent-head.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.2-intent-head-implicit-cpu.pt"
SEED = 42

CPU_ROWS = [
    "中央で幅広い種類の命令を処理する演算装置は何ですか。",
    "コンピュータ全体の制御と多様な命令実行を担う装置は何ですか。",
    "分岐やメモリ操作を含むさまざまな命令を扱う中央の処理装置は何ですか。",
    "汎用処理を担当し、異なる種類の命令を順に実行するプロセッサは何ですか。",
    "OSやアプリから来る多様な命令を処理する中心的な装置は何ですか。",
    "大量の同種並列計算ではなく、汎用的な命令処理を担う装置は何ですか。",
    "制御処理や順序依存の強い処理を柔軟に実行するプロセッサは何ですか。",
    "コンピュータの中央で命令実行と汎用制御を担当する装置を答えてください。",
    "幅広い命令を解釈して実行し、システム全体を制御する装置は何ですか。",
    "多様な処理を担当する汎用プロセッサの名称を答えてください。",
    "中央でさまざまな命令を処理する装置の名前を答えてください。",
    "汎用的な命令処理を担当する中央の演算装置の名前は何ですか。",
]

GPU_ROWS = [
    "大量の同種計算を同時に処理するのが得意な装置は何ですか。",
    "多数の演算を並列に進める用途に適したプロセッサは何ですか。",
    "画像処理や行列計算を大量に並列実行する装置は何ですか。",
    "同じ種類の計算を多数のデータへ一斉に適用する装置は何ですか。",
    "スループット重視で大量の計算スレッドを動かす装置は何ですか。",
    "機械学習で大量の並列計算を高速化する演算装置は何ですか。",
]

ERROR_ROWS = [
    "エラー内容と関連するコードを確認したいです。",
    "プログラム実行に失敗した原因を調べたいです。",
    "不具合の切り分け手順を教えてください。",
    "例外メッセージから原因を探したいです。",
]

OTHER_REPLAY = [
    ("tech_cpu", "CPUとは何ですか。"),
    ("tech_gpu", "GPUとは何ですか。"),
    ("tech_cuda", "CUDAとは何ですか。"),
    ("tech_python", "Pythonとは何ですか。"),
    ("tech_transformer", "Transformerとは何ですか。"),
    ("tech_llm", "LLMとは何ですか。"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT_HEAD)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


def encode_prompt(model, tokenizer, text):
    device = next(model.parameters()).device
    ids = tokenizer.encode(f"人: {text}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        hidden = model.forward_hidden(x)[:, -1, :]
    return hidden[0]


def exact_overlap():
    dev = {str(case["prompt"]) for case in CASES}
    rows = CPU_ROWS + GPU_ROWS + ERROR_ROWS + [x[1] for x in OTHER_REPLAY]
    return sorted(x for x in rows if x in dev)


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model, args.intent_head):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    overlap = exact_overlap()
    if overlap:
        raise RuntimeError("Exact fixed-DEV overlap: " + repr(overlap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ckpt = LanguageModel.load_checkpoint(args.model, device=device)
    head, head_ckpt, labels = load_intent_head(args.intent_head, model, device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    label_to_id = {label: i for i, label in enumerate(labels)}
    required = {"tech_cpu", "tech_gpu", "debug_error"}
    missing = required - set(label_to_id)
    if missing:
        raise RuntimeError("Missing intent labels: " + repr(sorted(missing)))

    rows = []
    rows += [(text, "tech_cpu") for text in CPU_ROWS]
    rows += [(text, "tech_gpu") for text in GPU_ROWS]
    rows += [(text, "debug_error") for text in ERROR_ROWS]
    rows += [(text, label) for label, text in OTHER_REPLAY]

    hidden = torch.stack([encode_prompt(model, tokenizer, text) for text, _ in rows])
    targets = torch.zeros((len(rows), len(labels)), dtype=torch.float32, device=device)
    for i, (_text, label) in enumerate(rows):
        targets[i, label_to_id[label]] = 1.0

    # Protect the original head globally with a distillation target.
    with torch.no_grad():
        original_logits = head(hidden).detach()

    for p in head.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)

    print()
    print("===========================================")
    print(" v0.9.2 Implicit CPU Intent Correction")
    print("===========================================")
    print("Device             :", device)
    print("Base loss          :", base_ckpt.get("loss"))
    print("Source head loss   :", head_ckpt.get("loss"))
    print("Base model         : frozen")
    print("Intent head        : trainable")
    print("Exact DEV overlap  :", len(overlap))
    print("CPU rows           :", len(CPU_ROWS))
    print("GPU rows           :", len(GPU_ROWS))
    print("Error rows         :", len(ERROR_ROWS))
    print("Replay rows        :", len(OTHER_REPLAY))
    print("LR                 :", args.lr)
    print("Margin             :", args.margin)
    print()

    cpu_i = label_to_id["tech_cpu"]
    gpu_i = label_to_id["tech_gpu"]
    error_i = label_to_id["debug_error"]

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    cpu_n = len(CPU_ROWS)
    gpu_n = len(GPU_ROWS)

    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)

        logits = head(hidden)
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        cpu_rows = probs[:cpu_n]
        gpu_rows = probs[cpu_n:cpu_n + gpu_n]

        cpu_margin = F.relu(
            args.margin - (cpu_rows[:, cpu_i] - cpu_rows[:, gpu_i])
        ).mean()
        cpu_error_margin = F.relu(
            args.margin - (cpu_rows[:, cpu_i] - cpu_rows[:, error_i])
        ).mean()
        gpu_margin = F.relu(
            args.margin - (gpu_rows[:, gpu_i] - gpu_rows[:, cpu_i])
        ).mean()

        # Keep changes small outside the targeted correction.
        distill = F.mse_loss(logits, original_logits)

        loss = (
            bce
            + 1.0 * cpu_margin
            + 0.5 * cpu_error_margin
            + 0.5 * gpu_margin
            + 0.05 * distill
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        value = float(loss.item())
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"| loss={value:.5f} bce={bce.item():.5f} "
                f"cpu-gpu={cpu_margin.item():.5f} "
                f"cpu-err={cpu_error_margin.item():.5f} "
                f"gpu-cpu={gpu_margin.item():.5f} "
                f"distill={distill.item():.5f}"
            )

        if value < best - 1e-6:
            best = value
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in head.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("No v0.9.2 intent-head state produced.")

    head.load_state_dict(best_state)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "labels": labels,
            "multi_label": True,
            "threshold": float(head_ckpt.get("threshold", 0.5)),
            "state_dict": head.state_dict(),
            "d_model": model.d_model,
            "epoch": best_epoch,
            "loss": best,
            "source_intent_head": args.intent_head,
            "experiment": "v0.9.2 implicit CPU intent correction",
            "exact_dev_overlap": overlap,
            "margin": args.margin,
            "learning_rate": args.lr,
        },
        args.output,
    )

    print()
    print("Completed.")
    print("Best epoch       :", best_epoch)
    print("Best loss        :", f"{best:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
