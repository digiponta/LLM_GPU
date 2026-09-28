# safe_hardest_competitor_binding_v0117.py
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class SafeHardestCompetitorBoost(nn.Module):
    """Predict a non-negative first-token boost with a near-zero initialization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        max_boost: float = 12.0,
        initial_boost: float = 0.20,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_boost = float(max_boost)
        self.initial_boost = float(initial_boost)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        nn.init.zeros_(self.net[-1].weight)

        ratio = min(max(self.initial_boost / self.max_boost, 1e-6), 1.0 - 1e-6)
        initial_bias = math.log(ratio / (1.0 - ratio))
        nn.init.constant_(self.net[-1].bias, initial_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.net(features).squeeze(-1)
        return self.max_boost * torch.sigmoid(raw)


def save_checkpoint(
    filename: str,
    adapter: SafeHardestCompetitorBoost,
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
            "max_boost": adapter.max_boost,
            "initial_boost": adapter.initial_boost,
            **meta,
        },
        filename,
    )


def load_checkpoint(filename: str, intent_labels: Sequence[str], device):
    ck = torch.load(filename, map_location=device)
    if list(ck["intent_labels"]) != list(intent_labels):
        raise ValueError("Intent label order mismatch.")

    adapter = SafeHardestCompetitorBoost(
        int(ck["input_dim"]),
        int(ck.get("hidden_dim", 32)),
        float(ck.get("max_boost", 12.0)),
        float(ck.get("initial_boost", 0.20)),
    ).to(device)
    adapter.load_state_dict(ck["state_dict"])
    adapter.eval()
    return adapter, ck
