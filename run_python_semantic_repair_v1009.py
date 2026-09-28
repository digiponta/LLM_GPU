# run_python_semantic_repair_v1009.py
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS=Path("results/python_semantic_repair_v1009")
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
    print("[train] v1.0.9 Python semantic detector")
    train_out=run([sys.executable,"train_python_semantic_detector_v1009.py"],env)
    print()
    print("[evaluate] v1.0.9 integrated")
    eval_out=run([sys.executable,"evaluate_python_semantic_repair_v1009.py"],env)
    LOG.write_text(train_out+"\n"+eval_out,encoding="utf-8")
    print("Log:",LOG)

if __name__=="__main__":
    main()
