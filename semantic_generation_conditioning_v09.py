# semantic_generation_conditioning_v09.py
#
# LLM_GPU v0.9
# Semantic Adapter v0.5 -> Generation Conditioning Integration
#
# Frozen semantic path:
#   frozen v0.8 prompt encoder
#     -> frozen Semantic Encoder Adapter v0.5
#     -> adapted semantic hidden (256)
#     -> concept probabilities (6)
#     -> attribute probabilities (4)
#     -> hierarchy probabilities (5)
#     -> concatenate = 271
#
# Generation conditioning:
#   271 -> 256 projection
#   -> alpha scaling
#   -> inject after Transformer Block 3
#
# Generation model:
#   Blocks 1-3 frozen
#   Blocks 4-6 trainable
#   FinalNorm trainable
#   LM Head frozen

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from mid_intent_conditioning_v09 import validate_injection_point
from model import LanguageModel
from partial_intent_conditioning_v09 import (
    configure_partial_finetune,
    forward_partial_conditioned,
)
from semantic_encoder_adapter_v05 import (
    ATTRIBUTE_LABELS,
    CONCEPT_LABELS,
    HIERARCHY_LABELS,
)


SEMANTIC_FEATURE_DIM = (
    256
    + len(CONCEPT_LABELS)
    + len(ATTRIBUTE_LABELS)
    + len(HIERARCHY_LABELS)
)


class SemanticGenerationProjection(nn.Module):
    def __init__(
        self,
        input_dim: int = SEMANTIC_FEATURE_DIM,
        d_model: int = 256,
        alpha: float = 0.1,
        inject_after: int = 3,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.alpha = float(alpha)
        self.inject_after = int(inject_after)
        self.linear = nn.Linear(self.input_dim, self.d_model)

        # Exact no-op at initialization.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.linear(features)


@torch.no_grad()
def infer_semantic_features(
    semantic_model: LanguageModel,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    token_ids: torch.Tensor,
    prompt_index: torch.Tensor | None = None,
):
    hidden = semantic_model.forward_hidden(token_ids)

    if prompt_index is None:
        raw_prompt = hidden[:, -1, :]
    else:
        batch = torch.arange(hidden.size(0), device=hidden.device)
        raw_prompt = hidden[batch, prompt_index]

    adapted = semantic_adapter(raw_prompt)
    concept_logits, attribute_logits = semantic_heads(adapted)
    hierarchy_logits = hierarchy_head(adapted)

    concept_prob = torch.softmax(concept_logits, dim=-1)
    attribute_prob = torch.sigmoid(attribute_logits)
    hierarchy_prob = torch.sigmoid(hierarchy_logits)

    features = torch.cat(
        [
            adapted,
            concept_prob,
            attribute_prob,
            hierarchy_prob,
        ],
        dim=-1,
    )

    return features, {
        "adapted_hidden": adapted,
        "concept_prob": concept_prob,
        "attribute_prob": attribute_prob,
        "hierarchy_prob": hierarchy_prob,
    }


def save_semantic_generation_checkpoint(
    filename: str,
    generation_model: LanguageModel,
    projection: SemanticGenerationProjection,
    *,
    epoch: int,
    loss: float,
    base_model: str,
    semantic_adapter: str,
    block_learning_rate: float,
    projection_learning_rate: float,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": "llm-gpu-v0.9-semantic-v05-generation-integration",
            "config": generation_model.config(),
            "model_state_dict": generation_model.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "semantic_feature_dim": projection.input_dim,
            "d_model": projection.d_model,
            "alpha": projection.alpha,
            "inject_after": projection.inject_after,
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "semantic_adapter": semantic_adapter,
            "block_learning_rate": float(block_learning_rate),
            "projection_learning_rate": float(projection_learning_rate),
            "concept_labels": list(CONCEPT_LABELS),
            "attribute_labels": list(ATTRIBUTE_LABELS),
            "hierarchy_labels": list(HIERARCHY_LABELS),
        },
        path,
    )


def load_semantic_generation_checkpoint(
    filename: str,
    device: torch.device,
):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9-semantic-v05-generation-integration":
        raise ValueError("Not a semantic-v0.5 generation integration checkpoint.")

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

    projection = SemanticGenerationProjection(
        input_dim=int(checkpoint["semantic_feature_dim"]),
        d_model=int(checkpoint["d_model"]),
        alpha=float(checkpoint.get("alpha", 0.1)),
        inject_after=int(checkpoint.get("inject_after", 3)),
    ).to(device)
    projection.load_state_dict(checkpoint["projection_state_dict"])
    projection.eval()

    return model, projection, checkpoint
