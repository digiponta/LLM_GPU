# run_targeted_boundary_v2_v09.py
#
# LLM_GPU v0.9 Targeted Boundary Training v2.
#
# Cumulative experiment:
#   v1 boundary rows + v2 boundary rows
#   block LR      = 1e-5
#   projection LR = 1e-3
#   alpha         = 0.1
#   inject after  = Block 3
#
# The goal is to preserve the v1 25/30 result while targeting the five
# remaining development failures: CPU, Transformer, short, repeat, end.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9-partial-intent-boundary-v2.pt"
RESULTS_DIR = Path("results/targeted_boundary_v2_v09")
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
        "--targeted-boundary-v2",
        "--output", CHECKPOINT,
    ]

    eval_cmd = [
        python,
        "evaluate_partial_intent_v09.py",
        "--partial", CHECKPOINT,
    ]

    print()
    print("============================================")
    print(" LLM_GPU v0.9 Targeted Boundary Training v2")
    print("============================================")
    print("Block LR          : 1e-5")
    print("Projection LR     : 1e-3")
    print("Alpha             : 0.1")
    print("Inject after      : 3")
    print("Boundary v1       : enabled")
    print("Boundary v2       : enabled")
    print("Checkpoint        :", CHECKPOINT)
    print("Results dir       :", RESULTS_DIR)
    print()

    print("[1/2] Training")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Reference before Boundary v2:")
    print("  Semantic : 25/30 (83.3%)")
    print("  Strict   : 25/30 (83.3%)")
    print()
    print("Primary v2 targets:")
    print("  G05 CPU")
    print("  G08 Transformer")
    print("  G12 short")
    print("  G15 repeat")
    print("  G18 end")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
