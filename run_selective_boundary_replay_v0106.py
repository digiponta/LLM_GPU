# run_selective_boundary_replay_v0106.py
#
# End-to-end runner for v0.10.6 Selective Alignment + Boundary Replay.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path("results/selective_boundary_replay_v0106")
TRAIN_LOG = RESULTS_DIR / "train.log"
EVAL_LOG = RESULTS_DIR / "evaluation.log"
CHECKPOINT = "model/model-gpu-v0.10.6-selective-boundary-replay.pt"


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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    print()
    print("====================================================")
    print(" Selective Alignment + Boundary Replay v0.10.6")
    print("====================================================")
    print("Branch                  : v0.10.6")
    print("Architecture            : same as v0.10.5")
    print("Targeted Boundary v2    : ON")
    print("Protected Boundary      : ON")
    print("Balanced Control Replay : OFF")
    print("Consistency weight      : 0.50")
    print("LM Head LR              : 3e-6")
    print("Block LR                : 1e-5")
    print("Projection LR           : 1e-3")
    print("Consistency LR          : 1e-3")
    print("Entity alignment weight : 0.05")
    print("Global alignment weight : 0.02")
    print("Continuation weight     : 0.02")
    print("Checkpoint              :", CHECKPOINT)
    print()

    print("[1/2] Training v0.10.6")
    run(
        [
            python,
            "train_selective_boundary_replay_v0106.py",
            "--output",
            CHECKPOINT,
            "--consistency-weight",
            "0.50",
            "--lm-head-lr",
            "3e-6",
            "--block-lr",
            "1e-5",
            "--projection-lr",
            "1e-3",
            "--consistency-lr",
            "1e-3",
            "--alignment-weight",
            "0.05",
            "--global-alignment-weight",
            "0.02",
            "--continuation-weight",
            "0.02",
        ],
        TRAIN_LOG,
    )

    print()
    print("[2/2] Evaluating fixed 30-case benchmark")
    run(
        [
            python,
            "evaluate_selective_boundary_replay_v0106.py",
            "--checkpoint",
            CHECKPOINT,
        ],
        EVAL_LOG,
    )

    print()
    print("====================================================")
    print(" v0.10.6 completed")
    print("====================================================")
    print("Train log :", TRAIN_LOG)
    print("Eval log  :", EVAL_LOG)
    print("Checkpoint:", CHECKPOINT)


if __name__ == "__main__":
    main()
