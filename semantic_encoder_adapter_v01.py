# semantic_encoder_adapter_v01.py
#
# LLM_GPU v0.9 Semantic Encoder Adapter v0.1
#
# Frozen base semantic encoder -> prompt hidden (256)
#   -> residual adapter: 256 -> 64 -> 256
#   -> adapted semantic hidden
#
# Supervision heads:
#   concept head   : 256 -> 6 technical concepts
#   attribute head : 256 -> 4 semantic attributes
#
# The base encoder is never updated.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


CONCEPT_LABELS = [
    "tech_gpu",
    "tech_cpu",
    "tech_llm",
    "tech_transformer",
    "tech_cuda",
    "tech_python",
]

ATTRIBUTE_LABELS = [
    "property_parallel",
    "property_general",
    "property_language",
    "property_attention_structure",
]


class SemanticEncoderAdapter(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 64,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)

        self.down = nn.Linear(self.d_model, self.hidden_dim)
        self.up = nn.Linear(self.hidden_dim, self.d_model)

        # Start close to identity.
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        delta = self.up(F.gelu(self.down(hidden)))
        return hidden + self.residual_scale * delta


class SemanticSupervisionHeads(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_concepts: int = len(CONCEPT_LABELS),
        num_attributes: int = len(ATTRIBUTE_LABELS),
    ):
        super().__init__()
        self.concept = nn.Linear(d_model, num_concepts)
        self.attribute = nn.Linear(d_model, num_attributes)

    def forward(self, hidden: torch.Tensor):
        return self.concept(hidden), self.attribute(hidden)


def save_semantic_adapter_checkpoint(
    filename: str,
    adapter: SemanticEncoderAdapter,
    heads: SemanticSupervisionHeads,
    *,
    epoch: int,
    loss: float,
    base_model: str,
    learning_rate: float,
    concept_weight: float,
    attribute_weight: float,
    preservation_weight: float,
) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": "llm-gpu-v0.9-semantic-encoder-adapter-v0.1",
            "adapter_state_dict": adapter.state_dict(),
            "heads_state_dict": heads.state_dict(),
            "d_model": adapter.d_model,
            "adapter_hidden_dim": adapter.hidden_dim,
            "residual_scale": adapter.residual_scale,
            "concept_labels": list(CONCEPT_LABELS),
            "attribute_labels": list(ATTRIBUTE_LABELS),
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "learning_rate": float(learning_rate),
            "concept_weight": float(concept_weight),
            "attribute_weight": float(attribute_weight),
            "preservation_weight": float(preservation_weight),
        },
        path,
    )


def load_semantic_adapter_checkpoint(
    filename: str,
    device: torch.device,
):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9-semantic-encoder-adapter-v0.1":
        raise ValueError("Not a Semantic Encoder Adapter v0.1 checkpoint.")

    adapter = SemanticEncoderAdapter(
        d_model=int(checkpoint["d_model"]),
        hidden_dim=int(checkpoint.get("adapter_hidden_dim", 64)),
        residual_scale=float(checkpoint.get("residual_scale", 0.25)),
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()

    heads = SemanticSupervisionHeads(
        d_model=int(checkpoint["d_model"]),
        num_concepts=len(checkpoint["concept_labels"]),
        num_attributes=len(checkpoint["attribute_labels"]),
    ).to(device)
    heads.load_state_dict(checkpoint["heads_state_dict"])
    heads.eval()

    return adapter, heads, checkpoint
