# semantic_consistency_v09.py
#
# LLM_GPU v0.9.1 Semantic Consistency Training v0.9
#
# A consistency head predicts frozen semantic targets from generation hidden:
#   6 concept + 4 attribute + 15 constrained hierarchy = 25 dims.

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from model import LanguageModel
from semantic_encoder_adapter_v01 import CONCEPT_LABELS, ATTRIBUTE_LABELS
from semantic_encoder_adapter_v08 import SEMANTIC_HIERARCHY_LABELS
from semantic_generation_integration_v09 import SemanticGenerationProjection


SEMANTIC_TARGET_DIM = (
    len(CONCEPT_LABELS)
    + len(ATTRIBUTE_LABELS)
    + len(SEMANTIC_HIERARCHY_LABELS)
)
FORMAT = "llm-gpu-v0.9.1-semantic-consistency-v0.9"


class SemanticConsistencyHead(nn.Module):
    def __init__(self, d_model=256, output_dim=SEMANTIC_TARGET_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, hidden):
        return self.linear(self.norm(hidden))


def semantic_target(concept_prob, attribute_prob, hierarchy_prob):
    return torch.cat(
        [concept_prob, attribute_prob, hierarchy_prob],
        dim=-1,
    )


def gather_prompt_hidden(hidden, prompt_index):
    # hidden: [B, T, D], prompt_index: [B]
    if prompt_index is None:
        return hidden[:, -1, :]
    idx = prompt_index.long().clamp(0, hidden.size(1) - 1)
    batch = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch, idx, :]


def split_semantic_vector(vector):
    c = len(CONCEPT_LABELS)
    a = len(ATTRIBUTE_LABELS)
    concept = vector[..., :c]
    attribute = vector[..., c:c+a]
    hierarchy = vector[..., c+a:]
    return concept, attribute, hierarchy


def save_semantic_consistency_checkpoint(
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
        },
        path,
    )


def load_semantic_consistency_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a Semantic Consistency v0.9 checkpoint.")

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
