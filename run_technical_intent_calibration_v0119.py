# run_technical_intent_calibration_v0119.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results/technical_intent_calibration_v0119")
TRAIN_LOG = RESULTS / "train.log"
EVAL_LOG = RESULTS / "evaluation.log"
CALIBRATOR = "model/model-gpu-v0.11.9-technical-intent-calibrator.pt"

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
    print(" Semantic Intent Repair / Calibration v0.11.9")
    print("====================================================")
    print("Base intent head : model/model-gpu-v0.8-intent-head-clean.pt (frozen)")
    print("Binding adapter  : model/model-gpu-v0.11.8-multi-concept-safe-binding.pt (frozen)")
    print("Calibrator       :", CALIBRATOR)
    print()

    print("[1/2] Training six-way technical intent calibrator")
    run(
        [
            py,
            "train_technical_intent_calibrator_v0119.py",
            "--output",
            CALIBRATOR,
        ],
        TRAIN_LOG,
    )

    print()
    print("[2/2] Evaluating calibrated gate + frozen v0.11.8 binding")
    run(
        [
            py,
            "evaluate_calibrated_multi_concept_binding_v0119.py",
            "--calibrator",
            CALIBRATOR,
        ],
        EVAL_LOG,
    )

    print()
    print("====================================================")
    print(" v0.11.9 completed")
    print("====================================================")
    print("Train log :", TRAIN_LOG)
    print("Eval log  :", EVAL_LOG)
    print("Calibrator:", CALIBRATOR)

if __name__ == "__main__":
    main()
