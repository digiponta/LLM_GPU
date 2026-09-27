# run_g05_topk_logit_diagnostic_v0122.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path("results/g05_topk_logits_v0122")
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
        "diagnose_g05_topk_logits_v0122.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--name-binding",
        "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt",
        "--v011-checkpoint",
        "model/model-gpu-v0.9.1-lexical-generation-v011.pt",
        "--v0122-checkpoint",
        "model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt",
        "--top-k",
        "10",
    ]

    print()
    print("====================================================")
    print(" v0.12.2 G05 Top-K Logit Diagnostic")
    print("====================================================")
    print("Current best          : v0.12.2")
    print("Training              : none")
    print("Gamma                 : 1.00, 1.50, 2.00, 2.50, 3.00")
    print("Top K                 : 10")
    print("Both G05 variants     : yes")
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
