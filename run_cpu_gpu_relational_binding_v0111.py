# run_cpu_gpu_relational_binding_v0111.py
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS=Path("results/cpu_gpu_relational_binding_v0111")
TRAIN_LOG=RESULTS/"train.log"
EVAL_LOG=RESULTS/"evaluation.log"
CHECKPOINT="model/model-gpu-v0.11.1-cpu-gpu-relational-binding.pt"

def env():
    e=os.environ.copy(); e["PYTHONIOENCODING"]="utf-8"; e["PYTHONUTF8"]="1"; return e

def run(cmd,log):
    p=subprocess.run(cmd,text=True,encoding="utf-8",errors="strict",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env())
    out=p.stdout or ""
    log.write_text(out,encoding="utf-8")
    print(out,end="" if out.endswith("\n") else "\n")
    if p.returncode!=0: raise SystemExit(p.returncode)

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    py=sys.executable

    print("====================================================")
    print(" CPU-GPU Relational Binding v0.11.1")
    print("====================================================")
    print("Base model  : model/model-gpu-v0.8-chat-clean.pt")
    print("Intent head : model/model-gpu-v0.8-intent-head-clean.pt")
    print("Relations   : cpu_controls_gpu / gpu_controlled_by_cpu /")
    print("              cpu_assigns_work_gpu / gpu_executes_for_cpu")
    print("Checkpoint  :",CHECKPOINT)
    print()

    print("[1/2] Training directional relation + entity binding")
    run([py,"train_cpu_gpu_relational_binding_v0111.py","--output",CHECKPOINT],TRAIN_LOG)

    print()
    print("[2/2] Evaluating fixed 30-case benchmark")
    run([py,"evaluate_cpu_gpu_relational_binding_v0111.py","--relational-binding",CHECKPOINT],EVAL_LOG)

    print()
    print("====================================================")
    print(" v0.11.1 completed")
    print("====================================================")
    print("Train log :",TRAIN_LOG)
    print("Eval log  :",EVAL_LOG)
    print("Checkpoint:",CHECKPOINT)

if __name__=="__main__":
    main()
