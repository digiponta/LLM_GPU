from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS = Path("results/entity_logit_alignment_v098")
LOG = RESULTS / "run.log"

def run(cmd):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", env=env, check=False
    )
    out = p.stdout or ""
    print(out, end="" if out.endswith("\n") else "\n")
    if p.returncode:
        raise SystemExit(p.returncode)
    return out

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("v0.9.8 Semantic Entity Logit Alignment")
    train = run([sys.executable, "train_entity_logit_alignment_v098.py"])
    evaluate = run([sys.executable, "evaluate_entity_logit_alignment_v098.py"])
    LOG.write_text(train + "\n" + evaluate, encoding="utf-8")
    print("Combined log:", LOG)

if __name__ == "__main__":
    main()
