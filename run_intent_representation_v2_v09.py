# run_intent_representation_v2_v09.py
#
# End-to-end runner for LLM_GPU v0.9 Intent Representation v2.
#
# Boundary configuration:
#   v1 enabled
#   v2/v3/v4 disabled
#
# Representation:
#   24-d intent probabilities
#   + 256-d frozen semantic hidden
#   = 280-d fused representation
#   -> Linear(280 -> 256)
#   -> inject after Block 3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9-intent-representation-v2.pt"
RESULTS_DIR = Path("results/intent_representation_v2_v09")
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
        "train_intent_representation_v2_v09.py",
        "--block-lr", "1e-5",
        "--projection-lr", "1e-3",
        "--alpha", "0.1",
        "--inject-after", "3",
        "--output", CHECKPOINT,
    ]
    eval_cmd = [
        python,
        "evaluate_intent_representation_v2_v09.py",
        "--checkpoint", CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" LLM_GPU v0.9 Intent Representation v2 Experiment")
    print("====================================================")
    print("Intent probabilities : 24 dims")
    print("Semantic hidden      : 256 dims")
    print("Fused representation : 280 dims")
    print("Projection output    : 256 dims")
    print("Block LR             : 1e-5")
    print("Projection LR        : 1e-3")
    print("Alpha                : 0.1")
    print("Inject after         : 3")
    print("Boundary data        : v1 only")
    print("LM Head              : frozen")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
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
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
