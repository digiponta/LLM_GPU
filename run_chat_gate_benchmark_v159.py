# run_chat_gate_benchmark_v159.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results/chat_gate_benchmark_v159")
LOG = RESULTS / "diagnostic.log"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    p = subprocess.run(
        [sys.executable, "chat_gate_benchmark_v159.py"],
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    output = p.stdout or ""
    LOG.write_text(output, encoding="utf-8")
    print(output, end="" if output.endswith("\n") else "\n")

    if p.returncode != 0:
        raise SystemExit(p.returncode)

    print("Log      :", LOG)
    print("Summary  :", RESULTS / "summary.csv")
    print("Per-case :", RESULTS / "per_case.csv")


if __name__ == "__main__":
    main()
