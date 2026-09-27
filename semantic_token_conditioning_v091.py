# semantic_token_conditioning_v091.py
#
# LLM_GPU v0.9.1
# Semantic-to-Generation Interface v0.7: Semantic Token Conditioning
#
# Flow:
#   text tokens -> embedding -> Blocks 1-3
#
#   frozen Semantic Adapter v0.5 produces:
#     adapted semantic hidden (256)
#     concept probs (6)
#     attribute probs (4)
#     hierarchy probs (5)
#
#   these become 3 virtual tokens:
#     [SEM_IDENTITY]  : adapted hidden -> 256
#     [SEM_CONCEPT]   : concept + attribute -> 256
#     [SEM_HIERARCHY] : hierarchy -> 256
#
#   virtual tokens are prepended after Block 3:
#     [SEM_IDENTITY][SEM_CONCEPT][SEM_HIERARCHY][text hidden...]
#
#   Blocks 4-6 can attend to semantic tokens.
#   Semantic-token outputs are removed before LM head scoring.

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model import LanguageModel
from semantic_encoder_adapter_v05 import (
    ATTRIBUTE_LABELS,
    CONCEPT_LABELS,
    HIERARCHY_LABELS,
)


SEMANTIC_TOKEN_COUNT = 3
SEMANTIC_TOKEN_LABELS = [
    "SEM_IDENTITY",
    "SEM_CONCEPT",
    "SEM_HIERARCHY",
]


class SemanticTokenProjector(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        concept_dim: int = len(CONCEPT_LABELS),
        attribute_dim: int = len(ATTRIBUTE_LABELS),
        hierarchy_dim: int = len(HIERARCHY_LABELS),
        scale: float = 1.0,
        inject_after: int = 3,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.concept_dim = int(concept_dim)
        self.attribute_dim = int(attribute_dim)
        self.hierarchy_dim = int(hierarchy_dim)
        self.scale = float(scale)
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

        # Start near no-op but not exactly zero: tiny semantic tokens let
        # late blocks learn an attention path from the first step.
        nn.init.normal_(self.identity_projection.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.concept_projection.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.hierarchy_projection.weight, mean=0.0, std=1e-3)

    def forward(
        self,
        adapted_hidden: torch.Tensor,
        concept_prob: torch.Tensor,
        attribute_prob: torch.Tensor,
        hierarchy_prob: torch.Tensor,
    ) -> torch.Tensor:
        identity = self.identity_projection(adapted_hidden)
        concept = self.concept_projection(
            torch.cat([concept_prob, attribute_prob], dim=-1)
        )
        hierarchy = self.hierarchy_projection(hierarchy_prob)

        tokens = torch.stack(
            [identity, concept, hierarchy],
            dim=1,
        )
        return self.scale * tokens


def configure_token_generation_model(
    model: LanguageModel,
    inject_after: int,
):
    for p in model.parameters():
        p.requires_grad_(False)

    for index, block in enumerate(model.blocks, start=1):
        if index > inject_after:
            for p in block.parameters():
                p.requires_grad_(True)

    for p in model.final_norm.parameters():
        p.requires_grad_(True)

    for p in model.lm_head.parameters():
        p.requires_grad_(False)


def _initial_hidden(model: LanguageModel, token_ids: torch.Tensor):
    if token_ids.dim() != 2:
        raise ValueError("token_ids must have shape [batch, time].")

    batch, time = token_ids.shape
    if time > model.context_length:
        raise ValueError(
            f"Sequence length {time} exceeds context length "
            f"{model.context_length}."
        )

    x = model.embedding(token_ids)
    if model.position_embedding is not None:
        positions = torch.arange(time, device=token_ids.device)
        x = x + model.position_embedding(positions).unsqueeze(0)
    return x


def forward_semantic_tokens(
    model: LanguageModel,
    token_ids: torch.Tensor,
    semantic_tokens: torch.Tensor,
    inject_after: int,
) -> torch.Tensor:
    if semantic_tokens.dim() != 3:
        raise ValueError(
            "semantic_tokens must have shape [batch, semantic_tokens, d_model]."
        )

    x = _initial_hidden(model, token_ids)

    for index, block in enumerate(model.blocks, start=1):
        if index <= inject_after:
            x = block(x)

    x = torch.cat([semantic_tokens, x], dim=1)

    for index, block in enumerate(model.blocks, start=1):
        if index > inject_after:
            x = block(x)

    x = model.final_norm(x)

    # Remove virtual token positions so targets remain aligned with text.
    return x[:, semantic_tokens.size(1):, :]


def save_semantic_token_checkpoint(
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
            "format": "llm-gpu-v0.9.1-semantic-token-conditioning-v0.7",
            "config": generation_model.config(),
            "model_state_dict": generation_model.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "semantic_token_count": SEMANTIC_TOKEN_COUNT,
            "semantic_token_labels": list(SEMANTIC_TOKEN_LABELS),
            "scale": projector.scale,
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


def load_semantic_token_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9.1-semantic-token-conditioning-v0.7":
        raise ValueError("Not a semantic-token conditioning v0.7 checkpoint.")

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

    projector = SemanticTokenProjector(
        d_model=model.d_model,
        scale=float(checkpoint.get("scale", 1.0)),
        inject_after=int(checkpoint.get("inject_after", 3)),
    ).to(device)
    projector.load_state_dict(checkpoint["projector_state_dict"])
    projector.eval()

    return model, projector, checkpoint
