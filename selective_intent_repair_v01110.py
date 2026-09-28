# selective_intent_repair_v01110.py
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


class SelectiveIntentRepair(nn.Module):
    """Residual six-way technical repair plus a technical-scope gate."""

    def __init__(self, num_labels: int = len(TECH_LABELS), hidden_dim: int = 24):
        super().__init__()
        self.num_labels = int(num_labels)
        self.hidden_dim = int(hidden_dim)

        self.repair = nn.Sequential(
            nn.Linear(self.num_labels, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.num_labels),
        )
        self.scope = nn.Sequential(
            nn.Linear(self.num_labels, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        nn.init.zeros_(self.repair[-1].weight)
        nn.init.zeros_(self.repair[-1].bias)

    def forward(self, tech_logits: torch.Tensor):
        repaired = tech_logits + self.repair(tech_logits)
        scope_logit = self.scope(tech_logits).squeeze(-1)
        return repaired, scope_logit


def extract_technical_logits(
    intent_logits: torch.Tensor,
    intent_labels: Sequence[str],
) -> torch.Tensor:
    indices = [list(intent_labels).index(label) for label in TECH_LABELS]
    return intent_logits[:, indices]


def save_checkpoint(
    filename: str,
    repair: SelectiveIntentRepair,
    intent_labels: Sequence[str],
    **meta,
):
    torch.save(
        {
            "state_dict": repair.state_dict(),
            "intent_labels": list(intent_labels),
            "technical_labels": list(TECH_LABELS),
            "num_labels": repair.num_labels,
            "hidden_dim": repair.hidden_dim,
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

    repair = SelectiveIntentRepair(
        int(ck.get("num_labels", len(TECH_LABELS))),
        int(ck.get("hidden_dim", 24)),
    ).to(device)
    repair.load_state_dict(ck["state_dict"])
    repair.eval()
    return repair, ck
