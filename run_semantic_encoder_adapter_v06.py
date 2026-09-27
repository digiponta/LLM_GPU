# run_semantic_encoder_adapter_v06.py
#
# End-to-end runner for Semantic Encoder Adapter v0.6:
# Instruction Semantics.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9.1-semantic-adapter-v06.pt"
RESULTS_DIR = Path("results/semantic_encoder_adapter_v06")
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
        "train_semantic_encoder_adapter_v06.py",
        "--init-adapter",
        "model/model-gpu-v0.9-semantic-adapter-v05.pt",
        "--output",
        CHECKPOINT,
        "--lr",
        "1e-4",
        "--hierarchy-weight",
        "1.0",
        "--instruction-contrast-weight",
        "1.0",
        "--preservation-weight",
        "0.25",
    ]

    eval_cmd = [
        python,
        "evaluate_semantic_encoder_adapter_v06.py",
        "--adapter",
        CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.6 Experiment")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Initialization       : semantic adapter v0.5")
    print("Base encoder         : frozen v0.8 pairwise-best")
    print("New CPU semantic     : heterogeneous_instruction")
    print("New GPU semantic     : homogeneous_computation")
    print("CPU instruction set  : arithmetic + branch + memory + control + I/O")
    print("GPU emphasis         : many similar computations in parallel")
    print("Generation training  : none")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
    print()

    print("[1/2] Training instruction semantics")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating instruction semantics")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary gate:")
    print("  G05 adapted centroid -> CPU")
    print("  G05 heterogeneous_instruction > homogeneous_computation")
    print("  non-arithmetic instructions -> CPU-like")
    print("  heterogeneous instructions -> CPU-like")
    print("  homogeneous computation -> GPU-like")
    print("  G08 Transformer retained")
    print("  held-out technical accuracy does not regress")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
