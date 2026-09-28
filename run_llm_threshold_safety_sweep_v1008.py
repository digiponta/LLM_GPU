# run_llm_threshold_safety_sweep_v1008.py
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS=Path("results/llm_threshold_safety_sweep_v1008")
LOG=RESULTS/"threshold_sweep.log"

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env["PYTHONIOENCODING"]="utf-8"; env["PYTHONUTF8"]="1"
    p=subprocess.run(
        [sys.executable,"llm_threshold_safety_sweep_v1008.py"],
        text=True,encoding="utf-8",errors="strict",
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,
    )
    out=p.stdout or ""
    LOG.write_text(out,encoding="utf-8")
    print(out,end="" if out.endswith("\n") else "\n")
    if p.returncode!=0: raise SystemExit(p.returncode)
    print("Log:",LOG)
    print("CSV:",RESULTS/"threshold_sweep.csv")

if __name__=="__main__":
    main()
