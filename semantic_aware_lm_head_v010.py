# semantic_aware_lm_head_v010.py
#
# LLM_GPU v0.9.1 v0.10
# Semantic-Aware LM Head Adaptation
#
# Same semantic consistency architecture as v0.9, but the LM head is trainable
# at a very low learning rate.

from __future__ import annotations

from pathlib import Path
import torch

from model import LanguageModel
from semantic_consistency_v09 import (
    SemanticConsistencyHead,
    SEMANTIC_TARGET_DIM,
)
from semantic_generation_integration_v09 import SemanticGenerationProjection


FORMAT = "llm-gpu-v0.9.1-semantic-aware-lm-head-v0.10"


def save_semantic_aware_lm_head_checkpoint(
    filename,
    generation_model,
    projection,
    consistency_head,
    *,
    epoch,
    loss,
    lm_loss,
    consistency_loss,
    semantic_adapter_checkpoint,
    base_model,
    consistency_weight,
    block_learning_rate,
    projection_learning_rate,
    consistency_learning_rate,
    lm_head_learning_rate,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": FORMAT,
            "config": generation_model.config(),
            "model_state_dict": generation_model.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "consistency_head_state_dict": consistency_head.state_dict(),
            "semantic_dim": projection.semantic_dim,
            "concept_dim": projection.concept_dim,
            "attribute_dim": projection.attribute_dim,
            "hierarchy_dim": projection.hierarchy_dim,
            "semantic_target_dim": SEMANTIC_TARGET_DIM,
            "alpha": projection.alpha,
            "inject_after": projection.inject_after,
            "epoch": int(epoch),
            "loss": float(loss),
            "lm_loss": float(lm_loss),
            "consistency_loss": float(consistency_loss),
            "semantic_adapter_checkpoint": semantic_adapter_checkpoint,
            "base_model": base_model,
            "consistency_weight": float(consistency_weight),
            "block_learning_rate": float(block_learning_rate),
            "projection_learning_rate": float(projection_learning_rate),
            "consistency_learning_rate": float(consistency_learning_rate),
            "lm_head_learning_rate": float(lm_head_learning_rate),
        },
        path,
    )


def load_semantic_aware_lm_head_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a Semantic-Aware LM Head v0.10 checkpoint.")

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

    consistency_head = SemanticConsistencyHead(
        d_model=model.d_model,
        output_dim=int(checkpoint["semantic_target_dim"]),
    ).to(device)
    consistency_head.load_state_dict(checkpoint["consistency_head_state_dict"])
    consistency_head.eval()

    return model, projection, consistency_head, checkpoint
