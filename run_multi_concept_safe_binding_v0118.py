# run_multi_concept_safe_binding_v0118.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results/multi_concept_safe_binding_v0118")
TRAIN_LOG = RESULTS / "train.log"
EVAL_LOG = RESULTS / "evaluation.log"
CHECKPOINT = "model/model-gpu-v0.11.8-multi-concept-safe-binding.pt"

def child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env

def run(cmd, log):
    p = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=child_env(),
    )
    out = p.stdout or ""
    log.write_text(out, encoding="utf-8")
    print(out, end="" if out.endswith("\n") else "\n")
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    print("====================================================")
    print(" Multi-Concept Safe Binding v0.11.8")
    print("====================================================")
    print("Concepts: GPU / CPU / LLM / Transformer / CUDA / Python")
    print("Initial boost: 0.20")
    print("Non-pair safety threshold: 0.70")
    print("Checkpoint:", CHECKPOINT)
    print()

    print("[1/2] Training multi-concept hardest-competitor boost")
    run(
        [
            py,
            "train_multi_concept_safe_binding_v0118.py",
            "--output",
            CHECKPOINT,
            "--max-boost",
            "12",
            "--initial-boost",
            "0.20",
            "--target-margin",
            "0.5",
            "--lr",
            "0.001",
        ],
        TRAIN_LOG,
    )

    print()
    print("[2/2] Evaluating fixed 30-case benchmark")
    run(
        [
            py,
            "evaluate_multi_concept_safe_binding_v0118.py",
            "--binding",
            CHECKPOINT,
        ],
        EVAL_LOG,
    )

    print()
    print("====================================================")
    print(" v0.11.8 completed")
    print("====================================================")
    print("Train log :", TRAIN_LOG)
    print("Eval log  :", EVAL_LOG)
    print("Checkpoint:", CHECKPOINT)

if __name__ == "__main__":
    main()
