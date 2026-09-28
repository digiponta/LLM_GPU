# hardest_competitor_binding_v0116.py
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn


class HardestCompetitorBoost(nn.Module):
    """Predict a non-negative boost for the selected target first token."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, max_boost: float = 12.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_boost = float(max_boost)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.net(features).squeeze(-1)
        return self.max_boost * torch.sigmoid(raw)


def save_checkpoint(
    filename: str,
    adapter: HardestCompetitorBoost,
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
            **meta,
        },
        filename,
    )


def load_checkpoint(filename: str, intent_labels: Sequence[str], device):
    ck = torch.load(filename, map_location=device)
    if list(ck["intent_labels"]) != list(intent_labels):
        raise ValueError("Intent label order mismatch.")
    adapter = HardestCompetitorBoost(
        int(ck["input_dim"]),
        int(ck.get("hidden_dim", 32)),
        float(ck.get("max_boost", 12.0)),
    ).to(device)
    adapter.load_state_dict(ck["state_dict"])
    adapter.eval()
    return adapter, ck
