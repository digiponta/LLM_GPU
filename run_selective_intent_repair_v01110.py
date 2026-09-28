# run_selective_intent_repair_v01110.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results/selective_intent_repair_v01110")
TRAIN_LOG = RESULTS / "train.log"
EVAL_LOG = RESULTS / "evaluation.log"
REPAIR = "model/model-gpu-v0.11.10-selective-intent-repair.pt"

def child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env

def run(cmd, log):
    p = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=child_env(),
    )
    out = p.stdout or ""
    log.write_text(out, encoding="utf-8")
    print(out, end="" if out.endswith("\n") else "\n")
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    print("====================================================")
    print(" Selective Intent Repair v0.11.10")
    print("====================================================")
    print("Base intent : frozen v0.8 clean")
    print("Binding     : frozen v0.11.8 multi-concept")
    print("Repair      :", REPAIR)
    print()

    print("[1/2] Training selective repair + technical scope gate")
    run(
        [py, "train_selective_intent_repair_v01110.py", "--output", REPAIR],
        TRAIN_LOG,
    )

    print()
    print("[2/2] Evaluating selective repair with frozen binding")
    run(
        [py, "evaluate_selective_intent_repair_v01110.py", "--repair", REPAIR],
        EVAL_LOG,
    )

    print()
    print("====================================================")
    print(" v0.11.10 completed")
    print("====================================================")
    print("Train log :", TRAIN_LOG)
    print("Eval log  :", EVAL_LOG)
    print("Repair    :", REPAIR)

if __name__ == "__main__":
    main()
