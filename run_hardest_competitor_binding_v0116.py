# run_hardest_competitor_binding_v0116.py
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS = Path("results/hardest_competitor_binding_v0116")
TRAIN_LOG = RESULTS / "train.log"
EVAL_LOG = RESULTS / "evaluation.log"
CHECKPOINT = "model/model-gpu-v0.11.6-hardest-competitor-binding.pt"

def env():
    e=os.environ.copy()
    e["PYTHONIOENCODING"]="utf-8"
    e["PYTHONUTF8"]="1"
    return e

def run(cmd, log):
    p=subprocess.run(cmd,text=True,encoding="utf-8",errors="strict",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env())
    out=p.stdout or ""
    log.write_text(out,encoding="utf-8")
    print(out,end="" if out.endswith("\n") else "\n")
    if p.returncode!=0:
        raise SystemExit(p.returncode)

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    py=sys.executable

    print("====================================================")
    print(" Hardest-Competitor Margin Binding v0.11.6")
    print("====================================================")
    print("Base model  : model/model-gpu-v0.8-chat-clean.pt")
    print("Intent head : model/model-gpu-v0.8-intent-head-clean.pt")
    print("Role teacher: model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt")
    print("Checkpoint  :",CHECKPOINT)
    print()

    print("[1/2] Training full-vocabulary hardest-competitor margin")
    run([
        py,"train_hardest_competitor_binding_v0116.py",
        "--output",CHECKPOINT,
        "--max-boost","12",
        "--target-margin","0.5",
        "--lr","0.001",
    ],TRAIN_LOG)

    print()
    print("[2/2] Evaluating fixed 30-case benchmark")
    run([
        py,"evaluate_hardest_competitor_binding_v0116.py",
        "--binding",CHECKPOINT,
    ],EVAL_LOG)

    print()
    print("====================================================")
    print(" v0.11.6 completed")
    print("====================================================")
    print("Train log :",TRAIN_LOG)
    print("Eval log  :",EVAL_LOG)
    print("Checkpoint:",CHECKPOINT)

if __name__=="__main__":
    main()
