# run_semantic_lexical_logit_alignment_v012.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9.1-semantic-lexical-logit-v012.pt"
RESULTS_DIR = Path("results/semantic_lexical_logit_v012")
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
        "train_semantic_lexical_logit_alignment_v012.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--name-binding",
        "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt",
        "--v011-checkpoint",
        "model/model-gpu-v0.9.1-lexical-generation-v011.pt",
        "--output",
        CHECKPOINT,
        "--lr",
        "5e-4",
        "--rank",
        "64",
        "--beta",
        "0.10",
        "--l2-weight",
        "1e-4",
    ]
    eval_cmd = [
        python,
        "evaluate_semantic_lexical_logit_alignment_v012.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--name-binding",
        "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt",
        "--v011-checkpoint",
        "model/model-gpu-v0.9.1-lexical-generation-v011.pt",
        "--logit-checkpoint",
        CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" Semantic/Lexical -> Logit Alignment v0.12")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("v0.11 generation     : frozen")
    print("Semantic teacher     : v0.8 frozen")
    print("CPU Name Binding     : v0.10.2 frozen")
    print("Condition            : 345 dims")
    print("Direct logit adapter : 345 -> 64 -> vocab")
    print("Beta                 : 0.10")
    print("Application          : first assistant token only")
    print("LM Head              : frozen")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
    print()

    print("[1/2] Training direct semantic/lexical logit adapter")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating fixed 30-case benchmark + G05 wording")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary signals:")
    print("  Original G05 -> CPU name")
    print("  Name-request G05 -> CPU name")
    print("  G04/G06 CPU retained")
    print("  G08 Transformer retained")
    print("  G09 CUDA retained")
    print("  G27 GPU retained")
    print("  G28 CPU retained")
    print("  semantic >= 26/30")
    print("  strict >= 26/30")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
