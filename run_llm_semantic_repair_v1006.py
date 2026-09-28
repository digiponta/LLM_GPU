# run_llm_semantic_repair_v1006.py
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS=Path("results/llm_semantic_repair_v1006")
LOG=RESULTS/"evaluation.log"

def run(cmd,env):
    p=subprocess.run(cmd,text=True,encoding="utf-8",errors="strict",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    print(p.stdout or "",end="" if (p.stdout or "").endswith("\n") else "\n")
    if p.returncode!=0: raise SystemExit(p.returncode)
    return p.stdout or ""

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env["PYTHONIOENCODING"]="utf-8"; env["PYTHONUTF8"]="1"
    print("[train] LLM semantic detector")
    run([sys.executable,"train_llm_semantic_detector_v1006.py"],env)
    print()
    print("[evaluate] integrated v1.0.6")
    out=run([sys.executable,"evaluate_llm_semantic_repair_v1006.py"],env)
    LOG.write_text(out,encoding="utf-8")
    print("Log:",LOG)

if __name__=="__main__":
    main()
