# run_cpu_name_binding_v0102.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
RESULTS_DIR = Path("results/cpu_name_binding_v0102")
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
        "train_cpu_name_binding_v0102.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--output",
        CHECKPOINT,
        "--lr",
        "1e-4",
        "--margin",
        "0.30",
        "--binding-dim",
        "64",
    ]
    eval_cmd = [
        python,
        "evaluate_cpu_name_binding_v0102.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--binding",
        CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" CPU Name-Meaning Binding v0.10.2 Experiment")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Purpose              : bind CPU English/Japanese names directly")
    print("English              : Central Processing Unit")
    print("Japanese             : 中央処理装置 / 中央演算処理装置")
    print("Word binding         : Central↔中央, Processing↔処理, Unit↔装置")
    print("Exact G05 training   : no")
    print("Base model           : frozen")
    print("Semantic adapter     : v0.8 frozen")
    print("Generation training  : none")
    print("Binding projection   : 256 -> 64")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
    print()

    print("[1/2] Training cross-lingual CPU name binding")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating name equivalence + G05 holdout")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary signals:")
    print("  Central Processing Unit ~= 中央処理装置")
    print("  Central Processing Unit ~= 中央演算処理装置")
    print("  Central/Processing/Unit Japanese mappings -> CPU")
    print("  exact G05 holdout -> CPU name centroid")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
