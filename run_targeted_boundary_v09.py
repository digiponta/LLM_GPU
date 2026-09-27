# run_targeted_boundary_v09.py
#
# LLM_GPU v0.9 Targeted Boundary Training experiment.
#
# Fixed experimental settings:
#   block LR      = 1e-5  (current robust local optimum)
#   projection LR = 1e-3
#   alpha         = 0.1
#   inject after  = Block 3
#   targeted boundary rows enabled
#
# Runs training and the unchanged fixed 30-case evaluation.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9-partial-intent-boundary.pt"
RESULTS_DIR = Path("results/targeted_boundary_v09")
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
        "--block-lr",
        "1e-5",
        "--projection-lr",
        "1e-3",
        "--alpha",
        "0.1",
        "--inject-after",
        "3",
        "--targeted-boundary",
        "--output",
        CHECKPOINT,
    ]

    eval_cmd = [
        python,
        "evaluate_partial_intent_v09.py",
        "--partial",
        CHECKPOINT,
    ]

    print()
    print("============================================")
    print(" LLM_GPU v0.9 Targeted Boundary Training")
    print("============================================")
    print("Block LR          : 1e-5")
    print("Projection LR     : 1e-3")
    print("Alpha             : 0.1")
    print("Inject after      : 3")
    print("Targeted boundary : enabled")
    print("Checkpoint        :", CHECKPOINT)
    print("Results dir       :", RESULTS_DIR)
    print()

    print("[1/2] Training")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating")
    output = run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Reference before Targeted Boundary Training:")
    print("  Semantic : 23/30 (76.7%)")
    print("  Strict   : 23/30 (76.7%)")
    print()
    print("Important residual intents:")
    print("  cpu, gpu_cpu, transformer, python, repeat, short, topic")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
