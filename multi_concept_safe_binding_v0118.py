# multi_concept_safe_binding_v0118.py
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

CONCEPTS = (
    "tech_gpu",
    "tech_cpu",
    "tech_llm",
    "tech_transformer",
    "tech_cuda",
    "tech_python",
)

CANONICAL = {
    "tech_gpu": "GPU",
    "tech_cpu": "CPU",
    "tech_llm": "LLM",
    "tech_transformer": "Transformer",
    "tech_cuda": "CUDA",
    "tech_python": "Python",
}


class MultiConceptSafeBoost(nn.Module):
    """Predict a non-negative boost conditioned on semantic features and target concept."""

    def __init__(
        self,
        feature_dim: int,
        num_concepts: int = len(CONCEPTS),
        hidden_dim: int = 48,
        max_boost: float = 12.0,
        initial_boost: float = 0.20,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_concepts = int(num_concepts)
        self.hidden_dim = int(hidden_dim)
        self.max_boost = float(max_boost)
        self.initial_boost = float(initial_boost)

        self.net = nn.Sequential(
            nn.Linear(self.feature_dim + self.num_concepts, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        nn.init.zeros_(self.net[-1].weight)
        ratio = min(max(self.initial_boost / self.max_boost, 1e-6), 1.0 - 1e-6)
        nn.init.constant_(self.net[-1].bias, math.log(ratio / (1.0 - ratio)))

    def forward(self, features: torch.Tensor, concept_index: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(
            concept_index.long(), num_classes=self.num_concepts
        ).to(dtype=features.dtype)
        x = torch.cat([features, one_hot], dim=-1)
        raw = self.net(x).squeeze(-1)
        return self.max_boost * torch.sigmoid(raw)


def save_checkpoint(
    filename: str,
    adapter: MultiConceptSafeBoost,
    intent_labels: Sequence[str],
    role_checkpoint: str,
    token_ids: dict,
    **meta,
):
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "intent_labels": list(intent_labels),
            "role_checkpoint": role_checkpoint,
            "concepts": list(CONCEPTS),
            "canonical": dict(CANONICAL),
            "token_ids": dict(token_ids),
            "feature_dim": adapter.feature_dim,
            "num_concepts": adapter.num_concepts,
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
    adapter = MultiConceptSafeBoost(
        int(ck["feature_dim"]),
        int(ck.get("num_concepts", len(CONCEPTS))),
        int(ck.get("hidden_dim", 48)),
        float(ck.get("max_boost", 12.0)),
        float(ck.get("initial_boost", 0.20)),
    ).to(device)
    adapter.load_state_dict(ck["state_dict"])
    adapter.eval()
    return adapter, ck
