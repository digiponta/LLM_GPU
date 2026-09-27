# run_semantic_encoder_adapter_v07.py

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

CHECKPOINT = "model/model-gpu-v0.9.1-semantic-adapter-v07.pt"
RESULTS_DIR = Path("results/semantic_encoder_adapter_v07")
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
        "train_semantic_encoder_adapter_v07.py",
        "--init-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v062.pt",
        "--output",
        CHECKPOINT,
        "--lr",
        "1e-4",
        "--hierarchy-weight",
        "1.0",
        "--relation-weight",
        "1.5",
        "--preservation-weight",
        "1.0",
    ]

    eval_cmd = [
        python,
        "evaluate_semantic_encoder_adapter_v07.py",
        "--adapter",
        CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.7 Experiment")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Initialization       : semantic adapter v0.6.2")
    print("Focus                : Instruction-Computation Hierarchy")
    print("Core relation        : computation is part of instruction execution")
    print("Base encoder         : frozen")
    print("Semantic adapter     : frozen")
    print("Concept/attr heads   : frozen")
    print("Hierarchy head       : trainable")
    print("Generation training  : none")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
    print()

    print("[1/2] Training Instruction-Computation hierarchy")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating semantic hierarchy")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary gate:")
    print("  computation <= instruction_execution")
    print("  arithmetic_logic <= computation")
    print("  control/memory/data movement <= instruction_execution")
    print("  G05 stays CPU and heterogeneous")
    print("  GPU repeated computation remains instruction execution")
    print("  G08 Transformer retained")
    print("  technical held-out accuracy does not regress")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
