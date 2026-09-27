# run_entity_contrastive_logit_alignment_v0122.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt"
RESULTS_DIR = Path("results/entity_contrastive_logit_v0122")
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
        "train_entity_contrastive_logit_alignment_v0122.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--name-binding",
        "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt",
        "--v011-checkpoint",
        "model/model-gpu-v0.9.1-lexical-generation-v011.pt",
        "--output",
        CHECKPOINT,
        "--lr",
        "3e-4",
        "--rank",
        "64",
        "--beta",
        "0.30",
        "--margin",
        "1.50",
        "--gate-weight",
        "0.50",
        "--l2-weight",
        "1e-4",
    ]
    eval_cmd = [
        python,
        "evaluate_entity_contrastive_logit_alignment_v0122.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--name-binding",
        "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt",
        "--v011-checkpoint",
        "model/model-gpu-v0.9.1-lexical-generation-v011.pt",
        "--contrastive-checkpoint",
        CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" Entity Contrastive Logit Alignment v0.12.2")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Objective            : correct entity > competing entities + margin")
    print("CPU/GPU contrast     : explicit")
    print("Entity gate          : enabled")
    print("Exact fixed DEV rows : excluded")
    print("v0.11 generation     : frozen")
    print("Beta                 : 0.30")
    print("Margin               : 1.50")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
    print()

    print("[1/2] Training contrastive entity logit adapter + gate")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating fixed 30-case benchmark + G05 CPU/GPU margins")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary signals:")
    print("  Original G05: CPU final logit > GPU final logit")
    print("  Name-request G05: CPU final logit > GPU final logit")
    print("  G05 -> CPU")
    print("  Non-entity gate low on error/compare/topic/repeat")
    print("  G08 -> Transformer")
    print("  G09 -> CUDA")
    print("  G10 -> Python")
    print("  G27 -> GPU")
    print("  G28 -> CPU")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
