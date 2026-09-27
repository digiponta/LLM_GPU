# run_intent_projection_realignment_v093.py
#
# v0.9.3 Intent-Projection Realignment
#
# Freeze v0.8 base model and corrected v0.9.2 intent head.
# Retrain only the v0.9 soft-intent projection, then compare it against the
# old v0.9.1 projection under the same corrected head.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


HEAD = "model/model-gpu-v0.9.2-intent-head-implicit-cpu.pt"
OUTPUT = "model/model-gpu-v0.9.3-soft-intent-realigned.pt"
RESULTS = Path("results/intent_projection_realignment_v093")
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
    out = completed.stdout or ""
    print(out, end="" if out.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print()
    print("============================================")
    print(" v0.9.3 Intent-Projection Realignment")
    print("============================================")
    print("Base LM             : frozen v0.8 pairwise-best")
    print("Intent head         : frozen corrected v0.9.2")
    print("Trainable           : soft-intent projection only")
    print("Old projection      : preserved")
    print("New projection      :", OUTPUT)
    print("Generation eval     : greedy / temperature=0 / history=0")
    print()

    chunks = []
    chunks.append(run([
        sys.executable,
        "train_sft_v09.py",
        "--intent-head", HEAD,
        "--output", OUTPUT,
        "--technical-repeat", "1",
        "--replay-repeat", "2",
        "--alpha", "0.1",
    ]))

    chunks.append(run([
        sys.executable,
        "evaluate_intent_projection_realignment_v093.py",
        "--intent-head", HEAD,
        "--new-projection", OUTPUT,
    ]))

    LOG.write_text("\n".join(chunks), encoding="utf-8")
    print("Combined log:", LOG)


if __name__ == "__main__":
    main()
