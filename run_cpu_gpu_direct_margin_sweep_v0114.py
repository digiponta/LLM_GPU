# run_cpu_gpu_direct_margin_sweep_v0114.py
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results/cpu_gpu_direct_margin_sweep_v0114")

CONFIGS = [
    {"name": "d16_lr2e3_m15", "max_delta": 16.0, "lr": 2e-3, "margin": 1.5},
    {"name": "d20_lr2e3_m15", "max_delta": 20.0, "lr": 2e-3, "margin": 1.5},
    {"name": "d20_lr3e3_m20", "max_delta": 20.0, "lr": 3e-3, "margin": 2.0},
    {"name": "d24_lr3e3_m20", "max_delta": 24.0, "lr": 3e-3, "margin": 2.0},
]

def child_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env

def run(cmd):
    p = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=child_env(),
    )
    out = p.stdout or ""
    print(out, end="" if out.endswith("\n") else "\n")
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    return out

def extract_case(text, case_no):
    m = re.search(
        rf"\[G{case_no:02d}\].*?(?=\n\[G\d{{2}}\]|\nSummary)",
        text,
        flags=re.S,
    )
    return m.group(0) if m else ""

def parse_gap(block):
    m = re.search(
        r"CPU-GPU logit gap: base=([+-]?\d+\.\d+) "
        r"delta=([+-]?\d+\.\d+) after=([+-]?\d+\.\d+)",
        block,
    )
    if not m:
        return None
    return tuple(float(x) for x in m.groups())

def parse_direct(block):
    m = re.search(r"direct-entity=(PASS|MISS)", block)
    return None if not m else m.group(1)

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    summary = []

    print("====================================================")
    print(" Direct CPU/GPU Margin Strength Sweep v0.11.4")
    print("====================================================")
    print("Selection criterion: G05 PASS, G27 PASS, G28 PASS")
    print()

    for cfg in CONFIGS:
        print()
        print("----------------------------------------------------")
        print("Config:", cfg["name"])
        print("max_delta:", cfg["max_delta"], "lr:", cfg["lr"], "target_margin:", cfg["margin"])
        print("----------------------------------------------------")

        checkpoint = f"model/model-gpu-v0.11.4-{cfg['name']}.pt"

        train_out = run([
            py,
            "train_cpu_gpu_direct_margin_binding_v0113.py",
            "--output", checkpoint,
            "--max-delta", str(cfg["max_delta"]),
            "--lr", str(cfg["lr"]),
            "--target-margin", str(cfg["margin"]),
        ])
        (RESULTS / f"{cfg['name']}_train.log").write_text(train_out, encoding="utf-8")

        eval_out = run([
            py,
            "evaluate_cpu_gpu_direct_margin_binding_v0113.py",
            "--binding", checkpoint,
        ])
        (RESULTS / f"{cfg['name']}_eval.log").write_text(eval_out, encoding="utf-8")

        g05 = extract_case(eval_out, 5)
        g27 = extract_case(eval_out, 27)
        g28 = extract_case(eval_out, 28)

        g05_gap = parse_gap(g05)
        g05_direct = parse_direct(g05)
        g27_direct = parse_direct(g27)
        g28_direct = parse_direct(g28)

        qualifies = (
            g05_direct == "PASS"
            and g27_direct == "PASS"
            and g28_direct == "PASS"
        )

        summary.append({
            "name": cfg["name"],
            "checkpoint": checkpoint,
            "g05_gap": g05_gap,
            "g05": g05_direct,
            "g27": g27_direct,
            "g28": g28_direct,
            "qualifies": qualifies,
        })

    print()
    print("====================================================")
    print(" v0.11.4 Sweep Summary")
    print("====================================================")
    for row in summary:
        gap_text = "n/a"
        if row["g05_gap"] is not None:
            b, d, a = row["g05_gap"]
            gap_text = f"base={b:+.3f} delta={d:+.3f} after={a:+.3f}"
        print(
            f"{row['name']:18s} | G05={row['g05']} G27={row['g27']} G28={row['g28']} "
            f"| {gap_text} | {'QUALIFY' if row['qualifies'] else 'reject'}"
        )

    candidates = [r for r in summary if r["qualifies"]]
    if candidates:
        # Choose the smallest absolute G05 overshoot above +1.0.
        candidates.sort(
            key=lambda r: abs((r["g05_gap"][2] if r["g05_gap"] else 999.0) - 1.0)
        )
        best = candidates[0]
        print()
        print("Recommended checkpoint:", best["checkpoint"])
        print("Reason: G05/G27/G28 all PASS with the smallest G05 margin overshoot.")
    else:
        print()
        print("No configuration satisfied all three direct-entity guards.")
        print("Inspect the per-config logs before increasing correction strength further.")

if __name__ == "__main__":
    main()
