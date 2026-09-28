# run_clean_intent_diagnostic_v0109.py
#
# Run full clean-v0.8 intent diagnosis plus focused G05 diagnosis.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path("results/clean_intent_diagnostic_v0109")
FULL_LOG = RESULTS_DIR / "intent_full.log"
G05_LOG = RESULTS_DIR / "g05_focus.log"

MODEL = "model/model-gpu-v0.8-chat-clean.pt"
INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"


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

    print()
    print("====================================================")
    print(" Clean Intent Diagnostic v0.10.9")
    print("====================================================")
    print("Model       :", MODEL)
    print("Intent head :", INTENT)
    print()

    print("[1/2] Full 30-case intent diagnostic")
    run(
        [
            python,
            "evaluate_intent_v09.py",
            "--model",
            MODEL,
            "--intent-head",
            INTENT,
            "--top-k-tags",
            "10",
        ],
        FULL_LOG,
    )

    print()
    print("[2/2] Focused G05 CPU/GPU intent diagnostic")
    run(
        [python, "diagnose_clean_g05_intent_v0109.py"],
        G05_LOG,
    )

    print()
    print("====================================================")
    print(" v0.10.9 completed")
    print("====================================================")
    print("Full log :", FULL_LOG)
    print("G05 log  :", G05_LOG)


if __name__ == "__main__":
    main()
