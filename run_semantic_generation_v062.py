# run_semantic_generation_v062.py
#
# End-to-end controlled comparison:
# Semantic Adapter v0.6.2 -> original v0.9 integration architecture.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9.1-semantic-generation-v062.pt"
RESULTS_DIR = Path("results/semantic_generation_v062")
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
        "train_semantic_generation_integration_v062.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v062.pt",
        "--output",
        CHECKPOINT,
        "--projection-lr",
        "1e-3",
        "--block-lr",
        "1e-5",
        "--alpha",
        "0.1",
        "--inject-after",
        "3",
    ]

    eval_cmd = [
        python,
        "evaluate_semantic_generation_v062.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v062.pt",
        "--integration",
        CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" Semantic v0.6.2 -> v0.9 Integration Experiment")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Semantic adapter     : v0.6.2 frozen")
    print("Generation interface : original v0.9 additive integration")
    print("Semantic features    : 256 hidden + 6 concept + 4 attribute + 7 hierarchy")
    print("Projection           : 273 -> 256")
    print("Inject after         : Block 3")
    print("Blocks 1-3           : frozen")
    print("Blocks 4-6           : trainable")
    print("FinalNorm            : trainable")
    print("LM Head              : frozen")
    print("Boundary data        : v1 only")
    print("Projection LR        : 1e-3")
    print("Block LR             : 1e-5")
    print("Alpha                : 0.1")
    print("Checkpoint           :", CHECKPOINT)
    print("Results dir          :", RESULTS_DIR)
    print()

    print("[1/2] Training controlled integration")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating fixed 30-case benchmark")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary success signals:")
    print("  G05 generation becomes CPU-correct")
    print("  G05 hierarchy keeps heterogeneous > homogeneous")
    print("  G08 remains Transformer-correct")
    print("  G09 CUDA remains correct")
    print("  semantic >= original v0.9 integration 26/30 if possible")
    print("  strict >= 25/30")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
