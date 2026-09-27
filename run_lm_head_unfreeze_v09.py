# run_lm_head_unfreeze_v09.py
#
# LLM_GPU v0.9 LM-Head Partial Unfreeze experiment.
#
# Baseline:
#   Boundary v1 data (25/30 best stable boundary result)
#
# Trainable:
#   Projection      : lr=1e-3
#   Blocks 4-6      : lr=1e-5
#   FinalNorm       : lr=1e-5
#   LM Head         : lr=1e-6
#
# Frozen:
#   Blocks 1-3
#   intent model/head
#
# Purpose:
#   Test whether limited output-token adaptation can improve hard residuals
#   such as G05 CPU reverse lookup and G08 Transformer category completion
#   without introducing the replay interference seen in Boundary v2-v4.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9-partial-intent-lmhead.pt"
RESULTS_DIR = Path("results/lm_head_unfreeze_v09")
TRAIN_LOG = RESULTS_DIR / "train.log"
EVAL_LOG = RESULTS_DIR / "eval.log"


def utf8_child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run(command, log_path: Path) -> str:
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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    train_cmd = [
        python,
        "train_partial_intent_v09.py",
        "--block-lr", "1e-5",
        "--projection-lr", "1e-3",
        "--alpha", "0.1",
        "--inject-after", "3",
        "--targeted-boundary",
        "--unfreeze-lm-head",
        "--lm-head-lr", "1e-6",
        "--output", CHECKPOINT,
    ]

    eval_cmd = [
        python,
        "evaluate_partial_intent_v09.py",
        "--partial", CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" LLM_GPU v0.9 LM-Head Partial Unfreeze Experiment")
    print("====================================================")
    print("Boundary data       : v1 only")
    print("Block LR            : 1e-5")
    print("Projection LR       : 1e-3")
    print("LM Head             : trainable")
    print("LM Head LR          : 1e-6")
    print("Alpha               : 0.1")
    print("Inject after        : 3")
    print("Checkpoint          :", CHECKPOINT)
    print("Results dir         :", RESULTS_DIR)
    print()

    print("[1/2] Training")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Reference Boundary v1:")
    print("  Semantic : 25/30 (83.3%)")
    print("  Strict   : 25/30 (83.3%)")
    print()
    print("Primary hard residuals:")
    print("  G05 CPU reverse identification")
    print("  G08 Transformer category completion")
    print()
    print("Regression watch:")
    print("  Python, GPU/CPU, topic, end, repeat, compare")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
