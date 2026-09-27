# run_selective_semantic_output_alignment_v0105.py
#
# End-to-end runner for v0.10.5 Selective Semantic Output Alignment.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path("results/selective_semantic_output_alignment_v0105")
TRAIN_LOG = RESULTS_DIR / "train.log"
EVAL_LOG = RESULTS_DIR / "evaluation.log"
CHECKPOINT = "model/model-gpu-v0.10.5-selective-semantic-output-aligned.pt"


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
    print(" Selective Semantic Output Alignment v0.10.5")
    print("====================================================")
    print("Branch                  : v0.10.5")
    print("Gate                    : technical intents only")
    print("Consistency weight      : 0.50")
    print("LM Head LR              : 3e-6")
    print("Block LR                : 1e-5")
    print("Projection LR           : 1e-3")
    print("Consistency LR          : 1e-3")
    print("Entity alignment weight : 0.05")
    print("Entity alignment margin : 0.75")
    print("Global alignment weight : 0.02")
    print("Global alignment margin : 0.50")
    print("Continuation weight     : 0.02")
    print("Continuation tokens     : 4")
    print("Alignment confidence    : 0.70")
    print("Checkpoint              :", CHECKPOINT)
    print()

    print("[1/2] Training selective semantic output alignment")
    run(
        [
            python,
            "train_selective_semantic_output_alignment_v0105.py",
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
            "--alignment-margin",
            "0.75",
            "--alignment-confidence",
            "0.70",
            "--global-alignment-weight",
            "0.02",
            "--global-alignment-margin",
            "0.50",
            "--continuation-weight",
            "0.02",
            "--continuation-tokens",
            "4",
        ],
        TRAIN_LOG,
    )

    print()
    print("[2/2] Evaluating fixed 30-case benchmark")
    run(
        [
            python,
            "evaluate_selective_semantic_output_alignment_v0105.py",
            "--checkpoint",
            CHECKPOINT,
        ],
        EVAL_LOG,
    )

    print()
    print("====================================================")
    print(" v0.10.5 completed")
    print("====================================================")
    print("Train log :", TRAIN_LOG)
    print("Eval log  :", EVAL_LOG)
    print("Checkpoint:", CHECKPOINT)


if __name__ == "__main__":
    main()
