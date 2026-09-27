# run_semantic_intent_bottleneck_v095.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS = Path("results/semantic_intent_bottleneck_v095")
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
    print("==========================================")
    print(" v0.9.5 Semantic Intent Bottleneck")
    print("==========================================")
    print("Base LM        : frozen v0.8 pairwise-best")
    print("Intent head    : frozen corrected v0.9.2")
    print("Semantic path  : 256 -> 64 -> 24 supervised tags")
    print("Fusion         : 64 + learned24 + frozen24")
    print("Generation     : FiLM after Block 3")
    print("Training       : bottleneck + semantic head + FiLM only")
    print()

    chunks = []
    chunks.append(run([
        sys.executable,
        "train_semantic_intent_bottleneck_v095.py",
    ]))
    chunks.append(run([
        sys.executable,
        "evaluate_semantic_intent_bottleneck_v095.py",
    ]))

    LOG.write_text("\n".join(chunks), encoding="utf-8")
    print("Combined log:", LOG)


if __name__ == "__main__":
    main()
