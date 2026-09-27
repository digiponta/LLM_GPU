# run_semantic_encoder_adapter_v01.py
#
# End-to-end runner for Semantic Encoder Adapter v0.1.
#
# Stage 1: train adapter + supervision heads on frozen v0.8 hidden states.
# Stage 2: evaluate G05/G08 before/after semantic geometry.
#
# No generation model is updated in this experiment.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9-semantic-adapter-v01.pt"
RESULTS_DIR = Path("results/semantic_encoder_adapter_v01")
TRAIN_LOG = RESULTS_DIR / "train.log"
EVAL_LOG = RESULTS_DIR / "eval.log"


def utf8_child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run(command, log_path: Path) -> str:
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
    return output


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    train_cmd = [
        python,
        "train_semantic_encoder_adapter_v01.py",
        "--adapter-hidden", "64",
        "--residual-scale", "0.25",
        "--lr", "5e-4",
        "--concept-weight", "1.0",
        "--attribute-weight", "0.75",
        "--preservation-weight", "0.25",
        "--output", CHECKPOINT,
    ]

    eval_cmd = [
        python,
        "evaluate_semantic_encoder_adapter_v01.py",
        "--adapter", CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.1 Experiment")
    print("====================================================")
    print("Base encoder          : frozen v0.8 pairwise-best")
    print("Adapter               : 256 -> 64 -> 256 residual")
    print("Residual scale        : 0.25")
    print("Concept supervision   : 6 technical classes")
    print("Attribute supervision : 4 attributes")
    print("Concept loss weight   : 1.0")
    print("Attribute loss weight : 0.75")
    print("Preservation weight   : 0.25")
    print("Learning rate         : 5e-4")
    print("Generation training   : none")
    print("Checkpoint            :", CHECKPOINT)
    print("Results dir           :", RESULTS_DIR)
    print()

    print("[1/2] Training adapter")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating semantic geometry")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary success conditions:")
    print("  G05 adapted centroid -> CPU")
    print("  G08 adapted centroid -> Transformer")
    print("  G08 property_attention_structure probability increases")
    print("  representation drift remains small")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
