# run_family_invariant_gpu_cpu_v132.py
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS=Path("results/family_invariant_gpu_cpu_v132")
LOG=RESULTS/"family_invariant.log"

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env["PYTHONIOENCODING"]="utf-8"
    env["PYTHONUTF8"]="1"
    p=subprocess.run(
        [sys.executable,"family_invariant_gpu_cpu_v132.py"],
        text=True,encoding="utf-8",errors="strict",
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,
    )
    out=p.stdout or ""
    LOG.write_text(out,encoding="utf-8")
    print(out,end="" if out.endswith("\n") else "\n")
    if p.returncode!=0:
        raise SystemExit(p.returncode)
    print("Log:",LOG)
    print("Summary:",RESULTS/"summary.csv")
    print("Matrix:",RESULTS/"leave_pair_out_accuracy.csv")

if __name__=="__main__":
    main()
