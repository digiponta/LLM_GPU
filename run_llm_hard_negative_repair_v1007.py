# run_llm_hard_negative_repair_v1007.py
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS=Path("results/llm_hard_negative_repair_v1007")
LOG=RESULTS/"evaluation.log"
CKPT="model/model-gpu-v1.0.7-llm-detector-hard-negative.pt"

def run(cmd,env):
    p=subprocess.run(cmd,text=True,encoding="utf-8",errors="strict",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    out=p.stdout or ""
    print(out,end="" if out.endswith("\n") else "\n")
    if p.returncode!=0: raise SystemExit(p.returncode)
    return out

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env["PYTHONIOENCODING"]="utf-8"; env["PYTHONUTF8"]="1"
    print("[train] v1.0.7 LLM hard-negative detector")
    train_out=run([sys.executable,"train_llm_semantic_detector_v1007.py"],env)
    print()
    print("[evaluate] v1.0.7 using v1.0.6 integrated evaluator")
    eval_out=run([
        sys.executable,
        "evaluate_llm_semantic_repair_v1006.py",
        "--llm-detector",CKPT,
    ],env)
    LOG.write_text(train_out+"\n"+eval_out,encoding="utf-8")
    print("Log:",LOG)

if __name__=="__main__":
    main()
