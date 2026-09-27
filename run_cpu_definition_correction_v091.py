# run_cpu_definition_correction_v091.py
#
# Data-audit -> projection-only retraining -> deterministic direct-definition
# evaluation. Existing v0.9 checkpoint is not overwritten.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


OUTPUT = "model/model-gpu-v0.9.1-soft-intent-cpu-definition.pt"
RESULTS = Path("results/cpu_definition_correction_v091")
LOG = RESULTS / "run.log"


def env_utf8():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run(command):
    print("$", " ".join(command))
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=env_utf8(),
        check=False,
    )
    output = completed.stdout or ""
    print(output, end="" if output.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )
    return output


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    chunks = []

    print()
    print("============================================")
    print(" CPU/GPU Definition Correction v0.9.1")
    print("============================================")
    print("Base model          : frozen v0.8 pairwise-best")
    print("Intent head         : frozen v0.8")
    print("Trainable           : v0.9 soft-intent projection only")
    print("Existing v0.9 ckpt  : preserved")
    print("New checkpoint      :", OUTPUT)
    print()

    chunks.append(run([
        sys.executable,
        "audit_cpu_gpu_training_data_v091.py",
    ]))

    chunks.append(run([
        sys.executable,
        "train_sft_v09.py",
        "--output", OUTPUT,
        "--technical-repeat", "1",
    ]))

    chunks.append(run([
        sys.executable,
        "evaluate_cpu_definition_projection_v091.py",
        "--projection", OUTPUT,
    ]))

    LOG.write_text("\n".join(chunks), encoding="utf-8")
    print("Combined log:", LOG)


if __name__ == "__main__":
    main()
