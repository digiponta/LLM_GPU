# run_semantic_consistency_weight_sweep_v0102.py
#
# v0.10.2 Semantic Consistency Weight Sweep
#
# v0.10.1 showed that LM-head LR changes alone do not recover G05/G08.
# This experiment fixes LM-head LR at the strongest non-regressing value
# (3e-6) and sweeps only semantic-consistency weight.

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path


CONSISTENCY_WEIGHTS = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
LM_HEAD_LR = 3e-6

RESULTS_DIR = Path("results/semantic_consistency_weight_v0102_sweep")
SUMMARY_CSV = RESULTS_DIR / "summary.csv"


def utf8_child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def weight_tag(weight: float) -> str:
    return f"{weight:.2f}".replace(".", "p")


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
    print(" Semantic Consistency Weight Sweep v0.10.2")
    print("====================================================")
    print("Branch               : v0.10.2")
    print("Sweep parameter      : consistency weight only")
    print("Weights              :", ", ".join(f"{x:.2f}" for x in CONSISTENCY_WEIGHTS))
    print("LM Head LR           : 3e-6")
    print("Projection LR        : 1e-3")
    print("Block LR             : 1e-5")
    print("Consistency-head LR  : 1e-3")
    print("Semantic teacher     : v0.8 constrained, frozen")
    print("Base checkpoint      : v0.8 pairwise-best")
    print("Summary CSV          :", SUMMARY_CSV)
    print()

    for index, weight in enumerate(CONSISTENCY_WEIGHTS, start=1):
        tag = weight_tag(weight)
        checkpoint = (
            f"model/model-gpu-v0.10.2-consistency-weight-{tag}.pt"
        )
        train_log = RESULTS_DIR / f"train-{tag}.log"
        eval_log = RESULTS_DIR / f"eval-{tag}.log"

        print()
        print("====================================================")
        print(
            f" Sweep {index}/{len(CONSISTENCY_WEIGHTS)} | "
            f"consistency weight={weight:.2f}"
        )
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
            str(LM_HEAD_LR),
            "--consistency-weight",
            str(weight),
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
            "consistency_weight": weight,
            "lm_head_lr": LM_HEAD_LR,
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

        def pm(value):
            return "PASS" if value else "MISS"

        print()
        print(
            "Sweep row:",
            f"W={weight:.2f}",
            f"val={row['best_val_loss']}",
            f"sem-loss={row['best_val_semantic']}",
            f"semantic={row['semantic_30']}/30",
            f"strict={row['strict_30']}/30",
            f"G05={pm(g05_sem)}",
            f"G08={pm(g08_sem)}",
            f"G09={pm(g09_sem)}",
            f"G28={pm(g28_sem)}",
        )

    fieldnames = list(rows[0].keys())
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("====================================================")
    print(" v0.10.2 Consistency Weight Sweep Summary")
    print("====================================================")
    print(
        f"{'Weight':>8s} {'Val':>9s} {'LM':>9s} {'SemLoss':>9s} "
        f"{'Semantic':>9s} {'Strict':>7s} "
        f"{'G05':>5s} {'G08':>5s} {'G09':>5s} {'G28':>5s}"
    )
    print("-" * 90)

    for row in rows:
        def pm(value):
            return "PASS" if value else "MISS"

        print(
            f"{row['consistency_weight']:8.2f} "
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

    print()
    print("Reference:")
    print("  v0.9 frozen LM head      : semantic 26/30, strict 26/30")
    print("  v0.10.1 LM LR=3e-6       : semantic 26/30, strict 26/30")
    print("  v0.10.1 LM LR=1e-5       : semantic 25/30, strict 25/30")
    print()
    print("Primary success criteria:")
    print("  - recover G05 and/or G08 without losing G09/G28")
    print("  - semantic/strict >= 26/30, preferably > 26/30")
    print("  - lower semantic-consistency loss without benchmark regression")
    print()
    print("Summary written:", SUMMARY_CSV)


if __name__ == "__main__":
    main()
