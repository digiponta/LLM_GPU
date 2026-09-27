# run_cpu_gpu_local_margin_v0123.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9.1-cpu-gpu-local-margin-v0123.pt"
RESULTS_DIR = Path("results/cpu_gpu_local_margin_v0123")
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
        "train_cpu_gpu_local_margin_v0123.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--name-binding",
        "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt",
        "--v011-checkpoint",
        "model/model-gpu-v0.9.1-lexical-generation-v011.pt",
        "--source-checkpoint",
        "model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt",
        "--output",
        CHECKPOINT,
        "--lr",
        "1e-4",
        "--margin",
        "1.0",
        "--classification-weight",
        "0.25",
        "--preservation-weight",
        "0.20",
    ]
    eval_cmd = [
        python,
        "evaluate_cpu_gpu_local_margin_v0123.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--name-binding",
        "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt",
        "--v011-checkpoint",
        "model/model-gpu-v0.9.1-lexical-generation-v011.pt",
        "--v0122-checkpoint",
        "model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt",
        "--v0123-checkpoint",
        CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" CPU-GPU Local Margin Refinement v0.12.3")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Source               : v0.12.2")
    print("Objective            : final CPU/GPU local margin")
    print("CPU target           : CPU > GPU + 1.0")
    print("GPU target           : GPU > CPU + 1.0")
    print("Exact G05            : excluded")
    print("Gate                 : frozen")
    print("Adapter              : local fine-tune")
    print("Preservation replay  : enabled")
    print("Learning rate        : 1e-4")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
    print()

    print("[1/2] Refining CPU/GPU local final-logit margin")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating fixed 30-case benchmark + G05 margin delta")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary signals:")
    print("  Original G05 final margin: CPU - GPU > 0")
    print("  Name-request G05 final margin: CPU - GPU > 0")
    print("  G05 generation -> CPU")
    print("  G27 -> GPU")
    print("  G28 -> CPU")
    print("  G08/G09/G10 retained")
    print("  Non-entity behavior retained")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
