# run_conditional_cpu_suppression_v0124.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path("results/conditional_cpu_suppression_v0124")
LOG = RESULTS_DIR / "sweep.log"


def utf8_child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "evaluate_conditional_cpu_suppression_v0124.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--name-binding",
        "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt",
        "--v011-checkpoint",
        "model/model-gpu-v0.9.1-lexical-generation-v011.pt",
        "--v0122-checkpoint",
        "model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt",
    ]

    print()
    print("====================================================")
    print(" v0.12.4 Conditional CPU-Semantic Suppression")
    print("====================================================")
    print("Current best          : v0.12.2")
    print("Training              : none")
    print("CPU condition         : frozen P(tech_cpu)")
    print("Direct gain           : 1.0")
    print("Suppression           : lambda * P(tech_cpu)")
    print("Lambda                : 0, 2, 4, 6, 8, 10")
    print("Blockers              : 多数の / 特集 / エ / 学習 / 読み")
    print("Evaluation            : fixed 30-case + both G05 variants")
    print("Output                :", LOG)
    print()

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
    LOG.write_text(output, encoding="utf-8")
    print(output, end="" if output.endswith("\n") else "\n")

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
