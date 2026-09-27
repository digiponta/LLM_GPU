# run_semantic_consistency_v09.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CHECKPOINT = "model/model-gpu-v0.9.1-semantic-consistency-v09.pt"
RESULTS_DIR = Path("results/semantic_consistency_v09")
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
        "train_semantic_consistency_v09.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--output",
        CHECKPOINT,
        "--projection-lr",
        "1e-3",
        "--block-lr",
        "1e-5",
        "--consistency-lr",
        "1e-3",
        "--consistency-weight",
        "0.35",
        "--alpha",
        "0.1",
        "--inject-after",
        "3",
    ]
    eval_cmd = [
        python,
        "evaluate_semantic_consistency_v09.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--checkpoint",
        CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" Semantic Consistency Training v0.9 Experiment")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Semantic teacher     : v0.8 constrained, frozen")
    print("Generation interface : additive 281 -> 256")
    print("Consistency target   : 6 concept + 4 attribute + 15 hierarchy")
    print("Consistency head     : generation hidden 256 -> semantic 25")
    print("Loss                 : LM + 0.35 * semantic consistency")
    print("Blocks 1-3           : frozen")
    print("Blocks 4-6           : trainable")
    print("FinalNorm            : trainable")
    print("LM Head              : frozen")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
    print()

    print("[1/2] Training semantic consistency")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating fixed 30-case benchmark + consistency")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary signals:")
    print("  G05 generation -> CPU")
    print("  G05 predicted semantic concept -> CPU")
    print("  G08 -> Transformer")
    print("  G09 -> CUDA")
    print("  G28 -> CPU")
    print("  semantic >= 26/30")
    print("  strict >= 26/30")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
