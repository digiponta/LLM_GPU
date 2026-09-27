# semantic_generation_integration_v08.py
#
# Semantic Encoder Adapter v0.8 -> original v0.9 additive generation integration.
#
# 256 hidden + 6 concept + 4 attribute + 15 hierarchy = 281 dims.

from __future__ import annotations

from pathlib import Path
import torch

from model import LanguageModel
from semantic_encoder_adapter_v08 import (
    load_semantic_adapter_v08_checkpoint,
)
from semantic_generation_integration_v09 import (
    SemanticGenerationProjection,
)

FORMAT = "llm-gpu-v0.9.1-semantic-generation-integration-v08"


def load_frozen_semantic_path_v08(base_model_path, adapter_path, device):
    semantic_model, base_checkpoint = LanguageModel.load_checkpoint(
        base_model_path, device=device
    )
    semantic_adapter, semantic_heads, hierarchy_head, adapter_checkpoint = (
        load_semantic_adapter_v08_checkpoint(adapter_path, device)
    )

    for module in (
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
    ):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)

    return (
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        base_checkpoint,
        adapter_checkpoint,
    )


def save_integration_checkpoint_v08(
    filename,
    generation_model,
    projection,
    *,
    epoch,
    loss,
    base_model,
    semantic_adapter_checkpoint,
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
            "alpha": projection.alpha,
            "inject_after": projection.inject_after,
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "semantic_adapter_checkpoint": semantic_adapter_checkpoint,
            "block_learning_rate": float(block_learning_rate),
            "projection_learning_rate": float(projection_learning_rate),
        },
        path,
    )


def load_integration_checkpoint_v08(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a v0.8 constrained semantic-generation checkpoint.")

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
        semantic_dim=int(checkpoint["semantic_dim"]),
        concept_dim=int(checkpoint["concept_dim"]),
        attribute_dim=int(checkpoint["attribute_dim"]),
        hierarchy_dim=int(checkpoint["hierarchy_dim"]),
        d_model=model.d_model,
        alpha=float(checkpoint.get("alpha", 0.1)),
        inject_after=int(checkpoint.get("inject_after", 3)),
    ).to(device)
    projection.load_state_dict(checkpoint["projection_state_dict"])
    projection.eval()

    return model, projection, checkpoint
