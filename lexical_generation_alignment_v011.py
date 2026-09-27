# lexical_generation_alignment_v011.py
#
# LLM_GPU v0.9.1 v0.11
# Lexical Identity -> Generation Alignment
#
# 281-d constrained semantic representation
# + 64-d frozen CPU name-binding representation
# = 345-d generation condition.

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from model import LanguageModel
from semantic_generation_integration_v09 import forward_semantic_conditioned


FORMAT = "llm-gpu-v0.9.1-lexical-generation-alignment-v0.11"


class LexicalGenerationProjection(nn.Module):
    def __init__(
        self,
        semantic_dim=256,
        concept_dim=6,
        attribute_dim=4,
        hierarchy_dim=15,
        lexical_dim=64,
        d_model=256,
        alpha=0.1,
        inject_after=3,
    ):
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.concept_dim = int(concept_dim)
        self.attribute_dim = int(attribute_dim)
        self.hierarchy_dim = int(hierarchy_dim)
        self.lexical_dim = int(lexical_dim)
        self.input_dim = (
            self.semantic_dim
            + self.concept_dim
            + self.attribute_dim
            + self.hierarchy_dim
            + self.lexical_dim
        )
        self.d_model = int(d_model)
        self.alpha = float(alpha)
        self.inject_after = int(inject_after)
        self.projection = nn.Linear(self.input_dim, self.d_model, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(
        self,
        adapted_hidden,
        concept_prob,
        attribute_prob,
        hierarchy_prob,
        lexical_identity,
    ):
        fused = torch.cat(
            [
                adapted_hidden,
                concept_prob,
                attribute_prob,
                hierarchy_prob,
                lexical_identity,
            ],
            dim=-1,
        )
        return self.alpha * self.projection(fused)


def save_checkpoint(
    filename,
    generation_model,
    projection,
    *,
    epoch,
    loss,
    base_model,
    semantic_adapter_checkpoint,
    name_binding_checkpoint,
    block_learning_rate,
    projection_learning_rate,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "config": generation_model.config(),
            "model_state_dict": generation_model.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "semantic_dim": projection.semantic_dim,
            "concept_dim": projection.concept_dim,
            "attribute_dim": projection.attribute_dim,
            "hierarchy_dim": projection.hierarchy_dim,
            "lexical_dim": projection.lexical_dim,
            "alpha": projection.alpha,
            "inject_after": projection.inject_after,
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "semantic_adapter_checkpoint": semantic_adapter_checkpoint,
            "name_binding_checkpoint": name_binding_checkpoint,
            "block_learning_rate": float(block_learning_rate),
            "projection_learning_rate": float(projection_learning_rate),
        },
        path,
    )


def load_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a Lexical Generation Alignment v0.11 checkpoint.")

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
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    projection = LexicalGenerationProjection(
        semantic_dim=int(checkpoint["semantic_dim"]),
        concept_dim=int(checkpoint["concept_dim"]),
        attribute_dim=int(checkpoint["attribute_dim"]),
        hierarchy_dim=int(checkpoint["hierarchy_dim"]),
        lexical_dim=int(checkpoint["lexical_dim"]),
        d_model=model.d_model,
        alpha=float(checkpoint.get("alpha", 0.1)),
        inject_after=int(checkpoint.get("inject_after", 3)),
    ).to(device)
    projection.load_state_dict(checkpoint["projection_state_dict"])
    projection.eval()
    return model, projection, checkpoint
