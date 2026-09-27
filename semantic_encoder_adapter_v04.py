# semantic_encoder_adapter_v04.py
#
# LLM_GPU v0.9 Semantic Encoder Adapter v0.4
# Acronym Contrastive Alignment

from __future__ import annotations

from pathlib import Path
import torch

from semantic_encoder_adapter_v01 import (
    ATTRIBUTE_LABELS,
    CONCEPT_LABELS,
    SemanticEncoderAdapter,
    SemanticSupervisionHeads,
)


def save_semantic_adapter_v04_checkpoint(
    filename,
    adapter,
    heads,
    *,
    epoch,
    loss,
    base_model,
    learning_rate,
    concept_weight,
    attribute_weight,
    centroid_margin_weight,
    centroid_margin,
    pairwise_weight,
    pairwise_margin,
    alignment_weight,
    alignment_temperature,
    preservation_weight,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": "llm-gpu-v0.9-semantic-encoder-adapter-v0.4",
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
            "centroid_margin_weight": float(centroid_margin_weight),
            "centroid_margin": float(centroid_margin),
            "pairwise_weight": float(pairwise_weight),
            "pairwise_margin": float(pairwise_margin),
            "alignment_weight": float(alignment_weight),
            "alignment_temperature": float(alignment_temperature),
            "preservation_weight": float(preservation_weight),
        },
        path,
    )


def load_semantic_adapter_v04_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9-semantic-encoder-adapter-v0.4":
        raise ValueError("Not a Semantic Encoder Adapter v0.4 checkpoint.")

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
