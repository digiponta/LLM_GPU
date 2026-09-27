# run_semantic_attention_probe_v091.py
#
# Runner for Semantic Attention Probe.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path("results/semantic_attention_probe_v091")
LOG = RESULTS_DIR / "probe.log"


def utf8_child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "semantic_attention_probe_v091.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9-semantic-adapter-v05.pt",
        "--balanced",
        "model/model-gpu-v0.9.1-semantic-token-balanced-v08.pt",
        "--cases",
        "5,8,9",
    ]

    print()
    print("====================================================")
    print(" Semantic Attention Probe Experiment")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Checkpoint           : balanced semantic tokens v0.8")
    print("Cases                : G05 / G08 / G09")
    print("Probe                : Blocks 4-6, all attention heads")
    print("Metrics              : text -> semantic attention")
    print("Training             : none")
    print("Output               :", LOG)
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
