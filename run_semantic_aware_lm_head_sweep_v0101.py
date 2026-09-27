# run_semantic_aware_lm_head_sweep_v0101.py
#
# v0.10.1 LM Head LR Sweep
#
# Controlled sweep over LM-head learning rate only.
# All other v0.10 settings remain unchanged.

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path


LM_HEAD_LRS = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5]

RESULTS_DIR = Path("results/semantic_aware_lm_head_v0101_sweep")
SUMMARY_CSV = RESULTS_DIR / "summary.csv"


def utf8_child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def lr_tag(lr: float) -> str:
    text = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    text = text.replace("+", "")
    return text


def run(command, log_path):
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
    log_path.write_text(output, encoding="utf-8")
    print(output, end="" if output.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return output


def search_float(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def search_int(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def case_pass(eval_text, case_no):
    pattern = (
        rf"\[G{case_no:02d}\].*?"
        rf"semantic-content=(PASS|MISS).*?strict=(PASS|MISS)"
    )
    match = re.search(pattern, eval_text, flags=re.S)
    if not match:
        return None, None
    return match.group(1) == "PASS", match.group(2) == "PASS"


def generated_answer(eval_text, case_no):
    pattern = rf"\[G{case_no:02d}\].*?\n\s+AI:\s*(.*?)\n"
    match = re.search(pattern, eval_text, flags=re.S)
    return match.group(1).strip() if match else ""


def summary_score(label, eval_text):
    match = re.search(rf"{re.escape(label)}\s*:\s*(\d+)/30", eval_text)
    return int(match.group(1)) if match else None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    rows = []

    print()
    print("====================================================")
    print(" Semantic-Aware LM Head v0.10.1 LR Sweep")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Sweep parameter      : LM Head LR only")
    print("LM Head LRs          :", ", ".join(f"{x:.0e}" for x in LM_HEAD_LRS))
    print("Semantic teacher     : v0.8 constrained, frozen")
    print("Projection LR        : 1e-3")
    print("Block LR             : 1e-5")
    print("Consistency LR       : 1e-3")
    print("Consistency weight   : 0.35")
    print("Base checkpoint      : v0.8 pairwise-best")
    print("Summary CSV          :", SUMMARY_CSV)
    print()

    for index, lr in enumerate(LM_HEAD_LRS, start=1):
        tag = lr_tag(lr)
        checkpoint = (
            f"model/model-gpu-v0.9.1-semantic-aware-lm-head-v0101-{tag}.pt"
        )
        train_log = RESULTS_DIR / f"train-{tag}.log"
        eval_log = RESULTS_DIR / f"eval-{tag}.log"

        print()
        print("====================================================")
        print(f" Sweep {index}/{len(LM_HEAD_LRS)} | LM Head LR={lr:.0e}")
        print("====================================================")

        train_cmd = [
            python,
            "train_semantic_aware_lm_head_v010.py",
            "--semantic-adapter",
            "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
            "--output",
            checkpoint,
            "--projection-lr",
            "1e-3",
            "--block-lr",
            "1e-5",
            "--consistency-lr",
            "1e-3",
            "--lm-head-lr",
            str(lr),
            "--consistency-weight",
            "0.35",
            "--alpha",
            "0.1",
            "--inject-after",
            "3",
        ]
        train_text = run(train_cmd, train_log)

        eval_cmd = [
            python,
            "evaluate_semantic_aware_lm_head_v010.py",
            "--semantic-adapter",
            "model/model-gpu-v0.9.1-semantic-adapter-v08.pt",
            "--checkpoint",
            checkpoint,
        ]
        eval_text = run(eval_cmd, eval_log)

        g05_sem, g05_strict = case_pass(eval_text, 5)
        g08_sem, g08_strict = case_pass(eval_text, 8)
        g09_sem, g09_strict = case_pass(eval_text, 9)
        g28_sem, g28_strict = case_pass(eval_text, 28)

        row = {
            "lm_head_lr": lr,
            "checkpoint": checkpoint,
            "best_epoch": search_int(r"Best epoch\s*:\s*(\d+)", train_text),
            "best_val_loss": search_float(
                r"Best val loss\s*:\s*([0-9.]+)", train_text
            ),
            "best_val_lm": search_float(
                r"Best val LM\s*:\s*([0-9.]+)", train_text
            ),
            "best_val_semantic": search_float(
                r"Best val semantic\s*:\s*([0-9.]+)", train_text
            ),
            "semantic_30": summary_score("Semantic-content rate", eval_text),
            "strict_30": summary_score("Strict composite rate", eval_text),
            "legacy_30": summary_score("Legacy rule rate", eval_text),
            "g05_semantic": g05_sem,
            "g05_strict": g05_strict,
            "g05_answer": generated_answer(eval_text, 5),
            "g08_semantic": g08_sem,
            "g08_strict": g08_strict,
            "g08_answer": generated_answer(eval_text, 8),
            "g09_semantic": g09_sem,
            "g09_strict": g09_strict,
            "g09_answer": generated_answer(eval_text, 9),
            "g28_semantic": g28_sem,
            "g28_strict": g28_strict,
            "g28_answer": generated_answer(eval_text, 28),
        }
        rows.append(row)

        print()
        print(
            "Sweep row:",
            f"LR={lr:.0e}",
            f"val={row['best_val_loss']}",
            f"semantic={row['semantic_30']}/30",
            f"strict={row['strict_30']}/30",
            f"G05={'PASS' if g05_sem else 'MISS'}",
            f"G08={'PASS' if g08_sem else 'MISS'}",
            f"G09={'PASS' if g09_sem else 'MISS'}",
            f"G28={'PASS' if g28_sem else 'MISS'}",
        )

    fieldnames = list(rows[0].keys())
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("====================================================")
    print(" v0.10.1 LM Head LR Sweep Summary")
    print("====================================================")
    print(
        f"{'LM LR':>10s} {'Val':>9s} {'LM':>9s} {'SemLoss':>9s} "
        f"{'Semantic':>9s} {'Strict':>7s} "
        f"{'G05':>5s} {'G08':>5s} {'G09':>5s} {'G28':>5s}"
    )
    print("-" * 88)
    for row in rows:
        def pm(value):
            return "PASS" if value else "MISS"
        print(
            f"{row['lm_head_lr']:10.0e} "
            f"{row['best_val_loss']:9.6f} "
            f"{row['best_val_lm']:9.6f} "
            f"{row['best_val_semantic']:9.6f} "
            f"{row['semantic_30']:>7d}/30 "
            f"{row['strict_30']:>5d}/30 "
            f"{pm(row['g05_semantic']):>5s} "
            f"{pm(row['g08_semantic']):>5s} "
            f"{pm(row['g09_semantic']):>5s} "
            f"{pm(row['g28_semantic']):>5s}"
        )

    # Reporting only; do not auto-declare a winner. The important scientific
    # evidence is the tradeoff between held-out loss, benchmark score, and
    # hard-case behavior.
    print()
    print("Reference:")
    print("  v0.9 frozen LM head     : semantic 26/30, strict 26/30")
    print("  v0.10 LM LR=1e-6        : semantic 26/30, strict 26/30")
    print()
    print("Summary written:", SUMMARY_CSV)


if __name__ == "__main__":
    main()
