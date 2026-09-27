# run_partial_lr_sweep_v09.py
#
# Automated block-learning-rate sweep for LLM_GPU v0.9 Partial Fine-Tuning.
#
# For each block LR:
#   1) train an independent partial-intent checkpoint
#   2) evaluate it on the same fixed 30-case benchmark
#   3) save train/eval logs
#   4) print and save a compact comparison table
#
# Projection LR stays fixed at 1e-3 by default.

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_BLOCK_LRS = [2e-6, 5e-6, 1e-5, 2e-5]
DEFAULT_PROJECTION_LR = 1e-3
DEFAULT_RESULTS_DIR = "results/partial_lr_sweep_v09"
DEFAULT_MODEL_DIR = "model"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run v0.9 Partial Fine-Tuning block-LR sweep."
    )
    p.add_argument(
        "--block-lrs",
        type=float,
        nargs="+",
        default=DEFAULT_BLOCK_LRS,
        help="Block/FinalNorm learning rates to sweep.",
    )
    p.add_argument(
        "--projection-lr",
        type=float,
        default=DEFAULT_PROJECTION_LR,
    )
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--inject-after", type=int, default=3)
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Evaluate existing sweep checkpoints without retraining.",
    )
    p.add_argument(
        "--show-output",
        action="store_true",
        help="Echo full child-process output while still saving logs.",
    )
    return p.parse_args()


def lr_tag(value: float) -> str:
    # Stable filename tag: 2e-06 -> 2e-6, 1e-05 -> 1e-5
    text = f"{value:.0e}"
    return text.replace("e-0", "e-").replace("e+0", "e+")


def run_command(
    command: List[str],
    log_path: Path,
    show_output: bool,
) -> str:
    process = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    output = process.stdout or ""
    log_path.write_text(output, encoding="utf-8")

    if show_output:
        print(output, end="" if output.endswith("\n") else "\n")

    if process.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{process.returncode}: {' '.join(command)}\n"
            f"See log: {log_path}"
        )

    return output


def parse_fraction(output: str, label: str):
    pattern = (
        rf"{re.escape(label)}\s*:\s*"
        r"(\d+)/(\d+)\s*\(([0-9.]+)%\)"
    )
    match = re.search(pattern, output)
    if not match:
        return None
    return {
        "pass": int(match.group(1)),
        "total": int(match.group(2)),
        "percent": float(match.group(3)),
    }


def parse_number(output: str, label: str):
    match = re.search(
        rf"{re.escape(label)}\s*:\s*([-+0-9.eE]+)",
        output,
    )
    return float(match.group(1)) if match else None


def parse_eval(output: str) -> Dict[str, object]:
    semantic = parse_fraction(output, "Semantic-content rate")
    strict = parse_fraction(output, "Strict composite rate")
    fluency = parse_fraction(output, "Fluency rate")
    legacy = parse_fraction(output, "Legacy rule rate")

    entity_match = re.search(
        r"Entity-explicit rate\s*:\s*"
        r"(\d+)/(\d+)\s*\(([0-9.]+)%\)",
        output,
    )
    entity = None
    if entity_match:
        entity = {
            "pass": int(entity_match.group(1)),
            "total": int(entity_match.group(2)),
            "percent": float(entity_match.group(3)),
        }

    per_intent = {}
    per_intent_pattern = re.compile(
        r"^([a-z_]+)\s*:\s*"
        r"(\d+)/(\d+)\s*\(([0-9.]+)%\)\s+"
        r"(\d+)/(\d+)\s*\(([0-9.]+)%\)\s*$",
        re.MULTILINE,
    )
    for match in per_intent_pattern.finditer(output):
        per_intent[match.group(1)] = {
            "semantic": int(match.group(2)),
            "n": int(match.group(3)),
            "strict": int(match.group(5)),
        }

    return {
        "semantic": semantic,
        "strict": strict,
        "entity": entity,
        "fluency": fluency,
        "legacy": legacy,
        "per_intent": per_intent,
    }


def read_checkpoint_metadata(checkpoint_path: Path):
    # Keep this script lightweight: torch is imported only here.
    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    return {
        "val_loss": float(checkpoint.get("loss")),
        "epoch": int(checkpoint.get("epoch", 0)),
        "block_lr": float(checkpoint.get("block_learning_rate", 0.0)),
        "projection_lr": float(
            checkpoint.get("projection_learning_rate", 0.0)
        ),
    }


def format_score(item) -> str:
    if not item:
        return "N/A"
    return f"{item['pass']}/{item['total']} ({item['percent']:.1f}%)"


def main() -> None:
    args = parse_args()

    if not args.block_lrs:
        raise ValueError("--block-lrs must contain at least one value.")

    results_dir = Path(args.results_dir)
    model_dir = Path(args.model_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    results = []

    print()
    print("============================================")
    print(" LLM_GPU v0.9 Partial Fine-Tuning LR Sweep")
    print("============================================")
    print(
        "Block LRs       :",
        ", ".join(f"{lr:g}" for lr in args.block_lrs),
    )
    print("Projection LR  :", f"{args.projection_lr:g}")
    print("Alpha          :", args.alpha)
    print("Inject after   :", args.inject_after)
    print("Epochs         :", args.epochs)
    print("Batch size     :", args.batch_size)
    print("Results dir    :", results_dir)
    print()

    for index, block_lr in enumerate(args.block_lrs, start=1):
        tag = lr_tag(block_lr)
        checkpoint_path = (
            model_dir / f"model-gpu-v0.9-partial-intent-blocklr-{tag}.pt"
        )
        train_log = results_dir / f"train-blocklr-{tag}.log"
        eval_log = results_dir / f"eval-blocklr-{tag}.log"

        print(
            f"[{index}/{len(args.block_lrs)}] "
            f"block-lr={block_lr:g}"
        )

        if not args.skip_training:
            train_cmd = [
                python,
                "train_partial_intent_v09.py",
                "--block-lr",
                str(block_lr),
                "--projection-lr",
                str(args.projection_lr),
                "--output",
                str(checkpoint_path),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--patience",
                str(args.patience),
                "--alpha",
                str(args.alpha),
                "--inject-after",
                str(args.inject_after),
            ]

            print("  training...")
            run_command(
                train_cmd,
                train_log,
                args.show_output,
            )
        elif not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for --skip-training: {checkpoint_path}"
            )

        print("  evaluating...")
        eval_cmd = [
            python,
            "evaluate_partial_intent_v09.py",
            "--partial",
            str(checkpoint_path),
        ]
        eval_output = run_command(
            eval_cmd,
            eval_log,
            args.show_output,
        )

        metadata = read_checkpoint_metadata(checkpoint_path)
        metrics = parse_eval(eval_output)

        result = {
            "block_lr": block_lr,
            "checkpoint": str(checkpoint_path),
            **metadata,
            **metrics,
        }
        results.append(result)

        print(
            "  -> val="
            f"{metadata['val_loss']:.6f}, "
            "semantic="
            f"{format_score(metrics['semantic'])}, "
            "strict="
            f"{format_score(metrics['strict'])}"
        )
        print()

    csv_path = results_dir / "partial_lr_sweep_v09.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "block_lr",
            "projection_lr",
            "best_epoch",
            "val_loss",
            "semantic_pass",
            "semantic_total",
            "semantic_percent",
            "strict_pass",
            "strict_total",
            "strict_percent",
            "entity_pass",
            "entity_total",
            "entity_percent",
            "fluency_pass",
            "fluency_total",
            "fluency_percent",
            "checkpoint",
        ])
        for r in results:
            semantic = r["semantic"] or {}
            strict = r["strict"] or {}
            entity = r["entity"] or {}
            fluency = r["fluency"] or {}
            writer.writerow([
                r["block_lr"],
                r["projection_lr"],
                r["epoch"],
                r["val_loss"],
                semantic.get("pass"),
                semantic.get("total"),
                semantic.get("percent"),
                strict.get("pass"),
                strict.get("total"),
                strict.get("percent"),
                entity.get("pass"),
                entity.get("total"),
                entity.get("percent"),
                fluency.get("pass"),
                fluency.get("total"),
                fluency.get("percent"),
                r["checkpoint"],
            ])

    print("Overall comparison")
    print("------------------")
    print(
        f"{'block-lr':>10}  {'val loss':>10}  "
        f"{'semantic':>17}  {'strict':>17}  "
        f"{'entity':>17}  {'fluency':>17}"
    )

    for r in results:
        print(
            f"{r['block_lr']:10.1e}  "
            f"{r['val_loss']:10.6f}  "
            f"{format_score(r['semantic']):>17}  "
            f"{format_score(r['strict']):>17}  "
            f"{format_score(r['entity']):>17}  "
            f"{format_score(r['fluency']):>17}"
        )

    print()
    print("Per-intent strict comparison")
    print("----------------------------")

    intents = sorted({
        intent
        for r in results
        for intent in r["per_intent"].keys()
    })

    header = "intent".ljust(18)
    for r in results:
        header += f"{r['block_lr']:.1e}".rjust(12)
    print(header)

    for intent in intents:
        row = intent.ljust(18)
        for r in results:
            stats = r["per_intent"].get(intent)
            cell = (
                f"{stats['strict']}/{stats['n']}"
                if stats
                else "-"
            )
            row += cell.rjust(12)
        print(row)

    best_strict = max(
        (r["strict"]["pass"] if r["strict"] else -1)
        for r in results
    )
    best_strict_rows = [
        r for r in results
        if r["strict"] and r["strict"]["pass"] == best_strict
    ]

    best_val = min(r["val_loss"] for r in results)

    print()
    print("Best observed")
    print("-------------")
    print(
        "Strict score :",
        ", ".join(
            f"block-lr={r['block_lr']:g} "
            f"({format_score(r['strict'])})"
            for r in best_strict_rows
        ),
    )
    print(
        "Lowest val   :",
        ", ".join(
            f"block-lr={r['block_lr']:g} ({r['val_loss']:.6f})"
            for r in results
            if abs(r["val_loss"] - best_val) < 1e-12
        ),
    )
    print("CSV          :", csv_path)
    print()
    print(
        "Baseline reference: v0.8 pairwise-best strict = "
        "22/30 (73.3%)."
    )


if __name__ == "__main__":
    main()
