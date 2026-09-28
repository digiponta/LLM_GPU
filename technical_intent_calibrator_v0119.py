# technical_intent_calibrator_v0119.py
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

TECH_LABELS = (
    "tech_gpu",
    "tech_cpu",
    "tech_llm",
    "tech_transformer",
    "tech_cuda",
    "tech_python",
)


class TechnicalIntentCalibrator(nn.Module):
    """Residual calibrator over the six frozen technical intent logits."""

    def __init__(self, num_labels: int = len(TECH_LABELS), hidden_dim: int = 24):
        super().__init__()
        self.num_labels = int(num_labels)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.num_labels, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.num_labels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tech_logits: torch.Tensor) -> torch.Tensor:
        return tech_logits + self.net(tech_logits)


def extract_technical_logits(
    intent_logits: torch.Tensor,
    intent_labels: Sequence[str],
) -> torch.Tensor:
    indices = [list(intent_labels).index(label) for label in TECH_LABELS]
    return intent_logits[:, indices]


def save_checkpoint(
    filename: str,
    calibrator: TechnicalIntentCalibrator,
    intent_labels: Sequence[str],
    **meta,
):
    torch.save(
        {
            "state_dict": calibrator.state_dict(),
            "intent_labels": list(intent_labels),
            "technical_labels": list(TECH_LABELS),
            "num_labels": calibrator.num_labels,
            "hidden_dim": calibrator.hidden_dim,
            **meta,
        },
        filename,
    )


def load_checkpoint(filename: str, intent_labels: Sequence[str], device):
    ck = torch.load(filename, map_location=device)
    if list(ck["intent_labels"]) != list(intent_labels):
        raise ValueError("Intent label order mismatch.")
    if list(ck["technical_labels"]) != list(TECH_LABELS):
        raise ValueError("Technical label order mismatch.")
    calibrator = TechnicalIntentCalibrator(
        int(ck.get("num_labels", len(TECH_LABELS))),
        int(ck.get("hidden_dim", 24)),
    ).to(device)
    calibrator.load_state_dict(ck["state_dict"])
    calibrator.eval()
    return calibrator, ck
