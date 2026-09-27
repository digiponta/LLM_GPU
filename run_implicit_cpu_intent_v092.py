# run_implicit_cpu_intent_v092.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS = Path("results/implicit_cpu_intent_v092")
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
    print("===========================================")
    print(" v0.9.2 Implicit CPU Intent Correction")
    print("===========================================")
    print("Base LM             : frozen")
    print("Source intent head  : v0.8")
    print("Trainable           : intent head only")
    print("Projection          : frozen corrected v0.9.1")
    print("Exact G05 training  : excluded")
    print()

    chunks = []
    chunks.append(run([
        sys.executable,
        "train_implicit_cpu_intent_v092.py",
    ]))
    chunks.append(run([
        sys.executable,
        "evaluate_implicit_cpu_intent_v092.py",
    ]))

    LOG.write_text("\n".join(chunks), encoding="utf-8")
    print("Combined log:", LOG)


if __name__ == "__main__":
    main()
