# run_local_logit_surface_diagnostic_v0122.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path("results/local_logit_surface_v0122")
LOG = RESULTS_DIR / "diagnostic.log"


def utf8_child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "diagnose_local_logit_surface_v0122.py",
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
    print(" v0.12.2 Local Logit Surface / Jacobian Diagnostic")
    print("====================================================")
    print("Current best          : v0.12.2")
    print("Training              : none")
    print("D(z)                  : logit(CPU) - logit(GPU)")
    print("Gradient              : dD/dz in frozen 345-d condition space")
    print("Directions            : nearest/centroid CPU and GPU")
    print("Finite steps          : 1%, 5%, 10%, 25%, 50%, 100%")
    print("Failed v0.12.3 model  : not loaded")
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
