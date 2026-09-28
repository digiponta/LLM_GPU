# cpu_gpu_direct_margin_binding_v0113.py
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn


class DirectGapBinding(nn.Module):
    """Predict one signed CPU-vs-GPU logit correction.

    Positive delta favors CPU. Negative delta favors GPU.
    The correction is applied symmetrically:
        CPU += delta / 2
        GPU -= delta / 2
    so the CPU-GPU logit gap changes by exactly delta.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, max_delta: float = 12.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.net(features).squeeze(-1)
        return self.max_delta * torch.tanh(raw / self.max_delta)


def save_checkpoint(
    filename: str,
    adapter: DirectGapBinding,
    intent_labels: Sequence[str],
    role_checkpoint: str,
    **meta,
):
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "intent_labels": list(intent_labels),
            "role_checkpoint": role_checkpoint,
            "input_dim": adapter.input_dim,
            "hidden_dim": adapter.hidden_dim,
            "max_delta": adapter.max_delta,
            **meta,
        },
        filename,
    )


def load_checkpoint(filename: str, intent_labels: Sequence[str], device):
    ck = torch.load(filename, map_location=device)
    if list(ck["intent_labels"]) != list(intent_labels):
        raise ValueError("Intent label order mismatch.")
    adapter = DirectGapBinding(
        int(ck["input_dim"]),
        int(ck.get("hidden_dim", 32)),
        float(ck.get("max_delta", 12.0)),
    ).to(device)
    adapter.load_state_dict(ck["state_dict"])
    adapter.eval()
    return adapter, ck
