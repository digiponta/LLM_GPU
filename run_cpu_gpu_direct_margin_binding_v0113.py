# run_cpu_gpu_direct_margin_binding_v0113.py
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results/cpu_gpu_direct_margin_binding_v0113")
TRAIN_LOG = RESULTS / "train.log"
EVAL_LOG = RESULTS / "evaluation.log"
CHECKPOINT = "model/model-gpu-v0.11.3-direct-cpu-gpu-gap-binding.pt"

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
    print(" Direct CPU/GPU Logit Margin Binding v0.11.3")
    print("====================================================")
    print("Base model  : model/model-gpu-v0.8-chat-clean.pt")
    print("Intent head : model/model-gpu-v0.8-intent-head-clean.pt")
    print("Role teacher: model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt")
    print("Checkpoint  :", CHECKPOINT)
    print()

    print("[1/2] Training direct signed CPU/GPU gap correction")
    run(
        [
            py,
            "train_cpu_gpu_direct_margin_binding_v0113.py",
            "--output",
            CHECKPOINT,
        ],
        TRAIN_LOG,
    )

    print()
    print("[2/2] Evaluating fixed 30-case benchmark")
    run(
        [
            py,
            "evaluate_cpu_gpu_direct_margin_binding_v0113.py",
            "--binding",
            CHECKPOINT,
        ],
        EVAL_LOG,
    )

    print()
    print("====================================================")
    print(" v0.11.3 completed")
    print("====================================================")
    print("Train log :", TRAIN_LOG)
    print("Eval log  :", EVAL_LOG)
    print("Checkpoint:", CHECKPOINT)

if __name__ == "__main__":
    main()
