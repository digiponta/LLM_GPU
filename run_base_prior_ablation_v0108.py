# run_base_prior_ablation_v0108.py
#
# Rebuild v0.8 from pretraining with the lexical prior ablated, then run
# v0.8 multi-task SFT and the fixed 30-case generalization benchmark.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PRETRAIN = "model/model-gpu-v0.8-pretrain-clean.pt"
CHAT = "model/model-gpu-v0.8-chat-clean.pt"
INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"

RESULTS_DIR = Path("results/base_prior_ablation_v0108")
PRETRAIN_LOG = RESULTS_DIR / "pretrain.log"
SFT_LOG = RESULTS_DIR / "sft.log"
EVAL_LOG = RESULTS_DIR / "evaluation.log"


def utf8_child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run(command, log_path):
    completed = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=utf8_child_env(),
    )
    output = completed.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    print(output, end="" if output.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return output


def audit_sft_sources():
    paths = [
        Path("data/conversation-ja.txt"),
        Path("data/instruction-ja.txt"),
        Path("augment_sft_v07.py"),
    ]
    print("SFT source audit:")
    total = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        count = text.count("多数")
        total += count
        print(f"  {path}: 多数={count}")
    if total != 0:
        raise RuntimeError(
            "v0.10.8 requires zero '多数' occurrences in SFT sources."
        )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    print()
    print("====================================================")
    print(" Base-Prior Ablation v0.10.8")
    print("====================================================")
    print("Rebuild start           : v0.8 pretraining from scratch")
    print("Tokenizer               : existing v0.7 BPE (fixed)")
    print("Pretraining ablation    : 多数 -> 多く in ALL text sources")
    print("SFT sources             : zero 多数 required")
    print("Architecture            : v0.8 unchanged")
    print("Pretrain output         :", PRETRAIN)
    print("Chat output             :", CHAT)
    print("Intent head output      :", INTENT)
    print()

    audit_sft_sources()

    print()
    print("[1/3] Rebuilding v0.8 clean-prior pretraining")
    run(
        [
            python,
            "train_mixed_v08_clean_prior.py",
            "--samples",
            "500000",
            "--epochs",
            "1",
            "--output",
            PRETRAIN,
        ],
        PRETRAIN_LOG,
    )

    print()
    print("[2/3] Rebuilding v0.8 multi-task SFT")
    run(
        [
            python,
            "train_sft_v08.py",
            "--base-model",
            PRETRAIN,
            "--output",
            CHAT,
            "--intent-head-output",
            INTENT,
            "--technical-repeat",
            "1",
            "--replay-repeat",
            "2",
        ],
        SFT_LOG,
    )

    print()
    print("[3/3] Evaluating rebuilt clean v0.8 base")
    run(
        [
            python,
            "evaluate_generalization_v07.py",
            "--model",
            CHAT,
        ],
        EVAL_LOG,
    )

    print()
    print("====================================================")
    print(" v0.10.8 completed")
    print("====================================================")
    print("Pretrain log :", PRETRAIN_LOG)
    print("SFT log      :", SFT_LOG)
    print("Eval log     :", EVAL_LOG)
    print("Pretrain     :", PRETRAIN)
    print("Chat model   :", CHAT)
    print("Intent head  :", INTENT)


if __name__ == "__main__":
    main()
