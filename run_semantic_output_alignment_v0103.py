# run_semantic_output_alignment_v0103.py
#
# End-to-end runner for v0.10.3 Semantic-to-Output Alignment.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path("results/semantic_output_alignment_v0103")
TRAIN_LOG = RESULTS_DIR / "train.log"
EVAL_LOG = RESULTS_DIR / "evaluation.log"
CHECKPOINT = "model/model-gpu-v0.10.3-semantic-output-aligned.pt"


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
    print(" Semantic-to-Output Alignment v0.10.3")
    print("====================================================")
    print("Branch               : v0.10.3")
    print("Consistency weight   : 0.50")
    print("LM Head LR           : 3e-6")
    print("Block LR             : 1e-5")
    print("Projection LR        : 1e-3")
    print("Consistency LR       : 1e-3")
    print("Alignment weight     : 0.10")
    print("Alignment margin     : 0.75")
    print("Alignment confidence : 0.70")
    print("Checkpoint           :", CHECKPOINT)
    print()

    print("[1/2] Training semantic-to-output alignment")
    run(
        [
            python,
            "train_semantic_output_alignment_v0103.py",
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
            "0.10",
            "--alignment-margin",
            "0.75",
            "--alignment-confidence",
            "0.70",
        ],
        TRAIN_LOG,
    )

    print()
    print("[2/2] Evaluating fixed 30-case benchmark")
    run(
        [
            python,
            "evaluate_semantic_output_alignment_v0103.py",
            "--checkpoint",
            CHECKPOINT,
        ],
        EVAL_LOG,
    )

    print()
    print("====================================================")
    print(" v0.10.3 completed")
    print("====================================================")
    print("Train log :", TRAIN_LOG)
    print("Eval log  :", EVAL_LOG)
    print("Checkpoint:", CHECKPOINT)


if __name__ == "__main__":
    main()
