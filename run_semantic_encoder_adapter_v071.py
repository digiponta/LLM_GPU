# run_semantic_encoder_adapter_v071.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9.1-semantic-adapter-v071.pt"
RESULTS_DIR = Path("results/semantic_encoder_adapter_v071")
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
        "train_semantic_encoder_adapter_v071.py",
        "--init-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v07.pt",
        "--output",
        CHECKPOINT,
        "--lr",
        "5e-5",
        "--relation-margin",
        "0.15",
        "--relation-weight",
        "2.0",
        "--preservation-weight",
        "1.0",
    ]

    eval_cmd = [
        python,
        "evaluate_semantic_encoder_adapter_v071.py",
        "--adapter",
        CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.7.1 Experiment")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Initialization       : semantic adapter v0.7")
    print("Focus                : G05 instruction-computation relation")
    print("Exact G05 training   : no")
    print("Core target          : instruction_execution > computation")
    print("Base encoder         : frozen")
    print("Semantic adapter     : frozen")
    print("Concept/attr heads   : frozen")
    print("Hierarchy head       : trainable")
    print("Generation training  : none")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
    print()

    print("[1/2] Refining G05 relation")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating relation hierarchy")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary gate:")
    print("  G05 instruction_execution > computation")
    print("  G05 stays CPU")
    print("  G05 heterogeneous stream stays strong")
    print("  G08 Transformer retained")
    print("  GPU repeated computation behavior retained")
    print("  technical held-out accuracy does not regress")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
