# run_gpu_semantic_failure_analysis_v129.py
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS=Path("results/gpu_semantic_failure_analysis_v129")
LOG=RESULTS/"analysis.log"

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env["PYTHONIOENCODING"]="utf-8"
    env["PYTHONUTF8"]="1"
    p=subprocess.run(
        [sys.executable,"gpu_semantic_failure_analysis_v129.py"],
        text=True,encoding="utf-8",errors="strict",
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env
    )
    out=p.stdout or ""
    LOG.write_text(out,encoding="utf-8")
    print(out,end="" if out.endswith("\n") else "\n")
    if p.returncode!=0:
        raise SystemExit(p.returncode)
    print("Log:",LOG)
    print("Multiclass:",RESULTS/"gpu_multiclass_by_layer.csv")
    print("Family failures:",RESULTS/"gpu_family_failure_modes.csv")
    print("Pairwise:",RESULTS/"gpu_pairwise_separability.csv")
    print("Best pair stage:",RESULTS/"gpu_pairwise_best_stage.csv")

if __name__=="__main__":
    main()
