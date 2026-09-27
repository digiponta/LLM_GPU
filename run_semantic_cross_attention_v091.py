# run_semantic_cross_attention_v091.py
#
# End-to-end runner for Semantic-to-Generation Interface v0.9.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9.1-semantic-cross-attn-v09.pt"
RESULTS_DIR = Path("results/semantic_cross_attention_v091")
TRAIN_LOG = RESULTS_DIR / "train.log"
EVAL_LOG = RESULTS_DIR / "eval.log"


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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    train_cmd = [
        python,
        "train_semantic_cross_attention_v091.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9-semantic-adapter-v05.pt",
        "--balanced",
        "model/model-gpu-v0.9.1-semantic-token-balanced-v08.pt",
        "--output",
        CHECKPOINT,
        "--cross-lr",
        "1e-3",
        "--block-lr",
        "1e-5",
        "--cross-heads",
        "8",
        "--initial-cross-scale",
        "0.1",
        "--inject-after",
        "3",
    ]

    eval_cmd = [
        python,
        "evaluate_semantic_cross_attention_v091.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9-semantic-adapter-v05.pt",
        "--balanced",
        "model/model-gpu-v0.9.1-semantic-token-balanced-v08.pt",
        "--cross",
        CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" Semantic-to-Generation Interface v0.9 Experiment")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Semantic adapter     : v0.5 frozen")
    print("Balanced projector   : v0.8 frozen")
    print("Interface            : dedicated Semantic Cross-Attention")
    print("Q                    : text hidden after Block 3")
    print("K,V                  : balanced semantic tokens")
    print("Cross heads          : 8")
    print("Initial cross scale  : 0.1")
    print("Blocks 4-6           : trainable")
    print("FinalNorm            : trainable")
    print("LM Head              : frozen")
    print("Boundary data        : v1 only")
    print("Cross LR             : 1e-3")
    print("Block LR             : 1e-5")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
    print()

    print("[1/2] Training Semantic Cross-Attention")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating fixed 30-case benchmark")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary success signals:")
    print("  G05 becomes CPU-correct")
    print("  G08 Transformer is correct and fluent")
    print("  G09 CUDA remains correct")
    print("  semantic score >= 26/30")
    print("  strict score > 25/30 if possible")
    print("  cross-attention diagnostics remain interpretable")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
