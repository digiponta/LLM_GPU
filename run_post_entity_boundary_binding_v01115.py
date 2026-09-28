# run_post_entity_boundary_binding_v01115.py
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results/post_entity_boundary_binding_v01115")
LOG = RESULTS / "evaluation.log"

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    p = subprocess.run(
        [sys.executable, "evaluate_post_entity_boundary_binding_v01115.py"],
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    out = p.stdout or ""
    LOG.write_text(out, encoding="utf-8")
    print(out, end="" if out.endswith("\n") else "\n")
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    print("Log:", LOG)

if __name__ == "__main__":
    main()
