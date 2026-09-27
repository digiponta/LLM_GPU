# run_semantic_encoder_adapter_v04.py
#
# End-to-end runner for Semantic Encoder Adapter v0.4.
#
# Continues from v0.3 and adds acronym/alias contrastive alignment.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT = "model/model-gpu-v0.9-semantic-adapter-v04.pt"
RESULTS_DIR = Path("results/semantic_encoder_adapter_v04")
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
        "train_semantic_encoder_adapter_v04.py",
        "--init-adapter",
        "model/model-gpu-v0.9-semantic-adapter-v03.pt",
        "--lr", "1e-4",
        "--concept-weight", "1.0",
        "--attribute-weight", "0.75",
        "--centroid-margin-weight", "0.20",
        "--centroid-margin", "0.05",
        "--pairwise-weight", "0.50",
        "--pairwise-margin", "0.08",
        "--alignment-weight", "1.00",
        "--alignment-temperature", "0.10",
        "--preservation-weight", "0.25",
        "--output", CHECKPOINT,
    ]

    eval_cmd = [
        python,
        "evaluate_semantic_encoder_adapter_v04.py",
        "--adapter", CHECKPOINT,
    ]

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.4 Experiment")
    print("====================================================")
    print("Base encoder          : frozen v0.8 pairwise-best")
    print("Initialization        : semantic adapter v0.3")
    print("Adapter               : 256 -> 64 -> 256 residual")
    print("Alignment groups      : CPU, GPU, LLM, CUDA")
    print("Concept loss weight   : 1.0")
    print("Attribute loss weight : 0.75")
    print("Centroid loss weight  : 0.20")
    print("Pairwise loss weight  : 0.50")
    print("Alignment loss weight : 1.00")
    print("Alignment temperature : 0.10")
    print("Preservation weight   : 0.25")
    print("Learning rate         : 1e-4")
    print("Generation training   : none")
    print("Checkpoint            :", CHECKPOINT)
    print("Results dir           :", RESULTS_DIR)
    print()

    print("[1/2] Training adapter v0.4")
    run(train_cmd, TRAIN_LOG)

    print()
    print("[2/2] Evaluating semantic geometry")
    run(eval_cmd, EVAL_LOG)

    print()
    print("Experiment complete.")
    print("Primary integration gate:")
    print("  G05 adapted centroid -> CPU")
    print("  G08 adapted centroid -> Transformer")
    print("  G09 CUDA improves or at least does not regress")
    print("  technical held-out centroid accuracy does not regress")
    print("  unseen CPU/GPU/LLM/CUDA alias probes pass")
    print()
    print("Logs:")
    print(" ", TRAIN_LOG)
    print(" ", EVAL_LOG)


if __name__ == "__main__":
    main()
