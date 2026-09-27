# run_g05_name_request_v0111.py

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path("results/g05_name_request_v0111")
EVAL_LOG = RESULTS_DIR / "eval.log"


def utf8_child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "evaluate_g05_name_request_v0111.py",
        "--semantic-adapter",
        "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
        "--name-binding",
        "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt",
        "--checkpoint",
        "model/model-gpu-v0.9.1-lexical-generation-v011.pt",
    ]

    print()
    print("====================================================")
    print(" G05 Name-Request Evaluation v0.11.1")
    print("====================================================")
    print("Training          : none")
    print("Original G05      : コンピュータの中心で多様な命令を処理する装置は何ですか。")
    print("Name-request G05  : コンピュータの中心で多様な命令を処理する装置の名前は何ですか。")
    print("Model             : v0.11 lexical generation alignment")
    print("Output            :", EVAL_LOG)
    print()

    completed = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=utf8_child_env(),
    )
    output = completed.stdout or ""
    EVAL_LOG.write_text(output, encoding="utf-8")
    print(output, end="" if output.endswith("\n") else "\n")

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
