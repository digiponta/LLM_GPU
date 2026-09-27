# semantic_token_balanced_v091.py
#
# LLM_GPU v0.9.1
# Semantic-to-Generation Interface v0.8: Balanced Semantic Tokens
#
# v0.7 problem:
#   SEM_IDENTITY norm  ~ 40
#   SEM_CONCEPT norm   ~ 0.2-0.3
#   SEM_HIERARCHY norm ~ 0.3-0.4
#
# v0.8:
#   project each semantic source
#   -> L2 normalize
#   -> scale to sqrt(d_model)
#   -> apply learnable per-token log scale
#
# This preserves direction while equalizing starting magnitudes.

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LanguageModel
from semantic_encoder_adapter_v05 import (
    ATTRIBUTE_LABELS,
    CONCEPT_LABELS,
    HIERARCHY_LABELS,
)
from semantic_token_conditioning_v091 import (
    SEMANTIC_TOKEN_COUNT,
    SEMANTIC_TOKEN_LABELS,
    configure_token_generation_model,
    forward_semantic_tokens,
)


class BalancedSemanticTokenProjector(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        concept_dim: int = len(CONCEPT_LABELS),
        attribute_dim: int = len(ATTRIBUTE_LABELS),
        hierarchy_dim: int = len(HIERARCHY_LABELS),
        base_norm: float | None = None,
        inject_after: int = 3,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.concept_dim = int(concept_dim)
        self.attribute_dim = int(attribute_dim)
        self.hierarchy_dim = int(hierarchy_dim)
        self.base_norm = float(
            math.sqrt(self.d_model) if base_norm is None else base_norm
        )
        self.inject_after = int(inject_after)

        self.identity_projection = nn.Linear(
            self.d_model,
            self.d_model,
            bias=False,
        )
        self.concept_projection = nn.Linear(
            self.concept_dim + self.attribute_dim,
            self.d_model,
            bias=False,
        )
        self.hierarchy_projection = nn.Linear(
            self.hierarchy_dim,
            self.d_model,
            bias=False,
        )

        # Same small initialization as v0.7.
        nn.init.normal_(self.identity_projection.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.concept_projection.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.hierarchy_projection.weight, mean=0.0, std=1e-3)

        # Equal initial relative strength. exp(log_scale) keeps scales positive.
        self.log_token_scales = nn.Parameter(
            torch.zeros(SEMANTIC_TOKEN_COUNT)
        )

    def _balance(self, token: torch.Tensor, index: int) -> torch.Tensor:
        direction = F.normalize(token, p=2, dim=-1, eps=1e-8)
        scale = torch.exp(self.log_token_scales[index])
        return direction * self.base_norm * scale

    def current_scales(self) -> torch.Tensor:
        return torch.exp(self.log_token_scales)

    def forward(
        self,
        adapted_hidden: torch.Tensor,
        concept_prob: torch.Tensor,
        attribute_prob: torch.Tensor,
        hierarchy_prob: torch.Tensor,
    ) -> torch.Tensor:
        identity_raw = self.identity_projection(adapted_hidden)
        concept_raw = self.concept_projection(
            torch.cat([concept_prob, attribute_prob], dim=-1)
        )
        hierarchy_raw = self.hierarchy_projection(hierarchy_prob)

        identity = self._balance(identity_raw, 0)
        concept = self._balance(concept_raw, 1)
        hierarchy = self._balance(hierarchy_raw, 2)

        return torch.stack(
            [identity, concept, hierarchy],
            dim=1,
        )


def save_balanced_semantic_token_checkpoint(
    filename,
    generation_model,
    projector,
    *,
    epoch,
    loss,
    base_model,
    semantic_adapter_checkpoint,
    block_learning_rate,
    projector_learning_rate,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": "llm-gpu-v0.9.1-balanced-semantic-token-v0.8",
            "config": generation_model.config(),
            "model_state_dict": generation_model.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "semantic_token_count": SEMANTIC_TOKEN_COUNT,
            "semantic_token_labels": list(SEMANTIC_TOKEN_LABELS),
            "base_norm": projector.base_norm,
            "inject_after": projector.inject_after,
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "semantic_adapter_checkpoint": semantic_adapter_checkpoint,
            "block_learning_rate": float(block_learning_rate),
            "projector_learning_rate": float(projector_learning_rate),
        },
        path,
    )


def load_balanced_semantic_token_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9.1-balanced-semantic-token-v0.8":
        raise ValueError("Not a balanced semantic-token v0.8 checkpoint.")

    if int(checkpoint.get("semantic_token_count", 0)) != SEMANTIC_TOKEN_COUNT:
        raise ValueError("Semantic token count mismatch.")

    config = checkpoint["config"]
    model = LanguageModel(
        vocab_size=int(config["vocab_size"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        hidden_dim=int(config["hidden_dim"]),
        num_heads=int(config.get("num_heads", 1)),
        causal=bool(config.get("causal", True)),
        context_length=int(config.get("context_length", 64)),
        use_position_embedding=bool(config.get("use_position_embedding", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    projector = BalancedSemanticTokenProjector(
        d_model=model.d_model,
        base_norm=float(checkpoint.get("base_norm", math.sqrt(model.d_model))),
        inject_after=int(checkpoint.get("inject_after", 3)),
    ).to(device)
    projector.load_state_dict(checkpoint["projector_state_dict"])
    projector.eval()

    return model, projector, checkpoint
