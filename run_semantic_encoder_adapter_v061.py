# run_semantic_encoder_adapter_v061.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9.1-semantic-adapter-v061.pt"
RESULTS_DIR = Path("results/semantic_encoder_adapter_v061")
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
        "train_semantic_encoder_adapter_v061.py",
        "--init-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v06.pt",
        "--output",
        CHECKPOINT,
        "--lr",
        "5e-5",
        "--hierarchy-weight",
        "0.75",
        "--direct-binding-weight",
        "1.5",
        "--direct-binding-margin",
        "0.30",
        "--hierarchy-preservation-weight",
        "0.50",
        "--representation-preservation-weight",
        "0.50",
    ]

    eval_cmd = [
        python,
        "evaluate_semantic_encoder_adapter_v061.py",
        "--adapter",
        CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.6.1 Experiment")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Initialization       : semantic adapter v0.6")
    print("Focus                : instruction binding refinement")
    print("Direct target        : heterogeneous_instruction > homogeneous_computation")
    print("Exact G05 training   : no")
    print("Concept/attr heads   : frozen")
    print("Generation training  : none")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
    print()

    print("[1/2] Refining instruction binding")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating binding refinement")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary gate:")
    print("  G05 centroid -> CPU")
    print("  G05 heterogeneous_instruction > homogeneous_computation")
    print("  G08 Transformer retained")
    print("  technical held-out accuracy does not regress")
    print("  diverse/mixed instructions -> CPU-like")
    print("  repeated homogeneous computation -> GPU-like")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
