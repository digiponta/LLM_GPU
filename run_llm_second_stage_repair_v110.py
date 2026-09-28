# run_llm_second_stage_repair_v110.py
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS=Path("results/llm_second_stage_repair_v110")
LOG=RESULTS/"evaluation.log"

def run(cmd,env):
    p=subprocess.run(cmd,text=True,encoding="utf-8",errors="strict",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    out=p.stdout or ""
    print(out,end="" if out.endswith("\n") else "\n")
    if p.returncode!=0: raise SystemExit(p.returncode)
    return out

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env["PYTHONIOENCODING"]="utf-8"; env["PYTHONUTF8"]="1"
    print("[train] v1.1.0 LLM second-stage detector")
    train_out=run([sys.executable,"train_llm_second_stage_detector_v110.py"],env)
    print()
    print("[evaluate] v1.1.0 integrated")
    eval_out=run([sys.executable,"evaluate_llm_second_stage_repair_v110.py"],env)
    LOG.write_text(train_out+"\n"+eval_out,encoding="utf-8")
    print("Log:",LOG)

if __name__=="__main__":
    main()
