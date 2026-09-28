# run_first_token_competitor_diagnostic_v0115.py
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS = Path("results/first_token_competitor_diagnostic_v0115")
LOG = RESULTS / "g05_competitors.log"

def env():
    e=os.environ.copy()
    e["PYTHONIOENCODING"]="utf-8"
    e["PYTHONUTF8"]="1"
    return e

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    cmd=[
        sys.executable,
        "diagnose_first_token_competitors_v0115.py",
        "--binding","model/model-gpu-v0.11.4-d16_lr2e3_m15.pt",
        "--top-k","20",
    ]
    p=subprocess.run(cmd,text=True,encoding="utf-8",errors="strict",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env())
    out=p.stdout or ""
    LOG.write_text(out,encoding="utf-8")
    print(out,end="" if out.endswith("\n") else "\n")
    if p.returncode!=0:
        raise SystemExit(p.returncode)
    print()
    print("Log:",LOG)

if __name__=="__main__":
    main()
