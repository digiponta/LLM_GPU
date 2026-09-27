# run_g05_after_cpu_definition_v091.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path("results/g05_after_cpu_definition_v091")
LOG = RESULTS_DIR / "evaluation.log"


def utf8_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "evaluate_g05_after_cpu_definition_v091.py",
        "--projection",
        "model/model-gpu-v0.9.1-soft-intent-cpu-definition.pt",
    ]

    print()
    print("================================================")
    print(" G05 Re-evaluation after CPU Definition v0.9.1")
    print("================================================")
    print("Training              : none")
    print("Projection            : corrected v0.9.1 CPU-definition checkpoint")
    print("Generation            : greedy / temperature=0 / history=0")
    print("Includes              : CPU/GPU basics + both G05 variants")
    print("Output                :", LOG)
    print()

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=utf8_env(),
        check=False,
    )
    output = completed.stdout or ""
    LOG.write_text(output, encoding="utf-8")
    print(output, end="" if output.endswith("\n") else "\n")

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
