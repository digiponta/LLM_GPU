# train.py
#
# CUDA/PyTorch training loop for LLM_GPU.

from __future__ import annotations

import math
import time
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def format_duration(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds):
        return "--:--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Trainer:
    def __init__(
        self,
        model,
        device: torch.device,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train(
        self,
        loader: DataLoader,
        epochs: int = 1,
    ) -> List[float]:
        history: List[float] = []
        total_steps = epochs * len(loader)
        completed_steps = 0
        global_start = time.perf_counter()

        self.model.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_start = time.perf_counter()

            for step, (input_ids, targets) in enumerate(loader, start=1):
                input_ids = input_ids.to(
                    self.device,
                    non_blocking=True,
                )
                targets = targets.to(
                    self.device,
                    non_blocking=True,
                )

                self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )

                loss.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip,
                    )

                self.optimizer.step()

                value = float(loss.item())
                epoch_loss += value
                completed_steps += 1

                elapsed = time.perf_counter() - global_start
                steps_per_second = (
                    completed_steps / elapsed if elapsed > 0 else 0.0
                )
                remaining = total_steps - completed_steps
                eta = (
                    remaining / steps_per_second
                    if steps_per_second > 0 else float("inf")
                )

                print(
                    f"\rEpoch {epoch}/{epochs} "
                    f"| Step {step:,}/{len(loader):,} "
                    f"| Loss {value:.6f} "
                    f"| Elapsed {format_duration(elapsed)} "
                    f"| ETA {format_duration(eta)}",
                    end="",
                    flush=True,
                )

            print()

            average_loss = epoch_loss / max(1, len(loader))
            history.append(average_loss)

            epoch_elapsed = time.perf_counter() - epoch_start
            print(
                f"Epoch {epoch} completed "
                f"| Average loss {average_loss:.6f} "
                f"| Perplexity {math.exp(min(average_loss, 20.0)):.3f} "
                f"| Time {format_duration(epoch_elapsed)}"
            )

        return history
