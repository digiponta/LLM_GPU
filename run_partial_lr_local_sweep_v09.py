# run_partial_lr_local_sweep_v09.py
#
# Local refinement around the current best block LR (1e-5) for
# LLM_GPU v0.9 Partial Fine-Tuning.
#
# This wrapper reuses run_partial_lr_sweep_v09.py and keeps results separate
# from the coarse sweep.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


LOCAL_BLOCK_LRS = [7.5e-6, 1e-5, 1.25e-5, 1.5e-5]
RESULTS_DIR = "results/partial_lr_local_sweep_v09"


def main() -> None:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "run_partial_lr_sweep_v09.py",
        "--block-lrs",
        *(str(value) for value in LOCAL_BLOCK_LRS),
        "--results-dir",
        RESULTS_DIR,
    ]

    print()
    print("============================================")
    print(" LLM_GPU v0.9 Local Block-LR Refinement")
    print("============================================")
    print(
        "Block LRs      :",
        ", ".join(f"{value:g}" for value in LOCAL_BLOCK_LRS),
    )
    print("Projection LR :", "1e-3")
    print("Alpha         :", "0.1")
    print("Inject after  :", "3")
    print("Results dir   :", RESULTS_DIR)
    print()
    print("Running:")
    print(" ".join(command))
    print()

    completed = subprocess.run(command, check=False)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    print()
    print("Local refinement completed.")
    print(
        "Compare the best strict score against the current "
        "23/30 (76.7%) result at block-lr=1e-5."
    )


if __name__ == "__main__":
    main()
