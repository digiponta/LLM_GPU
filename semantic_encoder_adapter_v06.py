# semantic_encoder_adapter_v06.py
#
# LLM_GPU v0.9.1 Semantic Encoder Adapter v0.6
# Instruction Semantics
#
# Extends v0.5 hierarchy with:
#   heterogeneous_instruction  (CPU-like)
#   homogeneous_computation    (GPU-like)

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from semantic_encoder_adapter_v01 import (
    ATTRIBUTE_LABELS,
    CONCEPT_LABELS,
    SemanticEncoderAdapter,
    SemanticSupervisionHeads,
)
from semantic_encoder_adapter_v05 import (
    HIERARCHY_LABELS as V05_HIERARCHY_LABELS,
    load_semantic_adapter_v05_checkpoint,
)


HIERARCHY_LABELS = [
    "processor",
    "general_purpose",
    "control_oriented",
    "throughput_oriented",
    "data_parallel",
    "heterogeneous_instruction",
    "homogeneous_computation",
]


class InstructionSemanticHead(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_labels: int = len(HIERARCHY_LABELS),
    ):
        super().__init__()
        self.linear = nn.Linear(d_model, num_labels)

    def forward(self, hidden: torch.Tensor):
        return self.linear(hidden)


def initialize_from_v05(filename, device):
    adapter, heads, old_hierarchy, checkpoint = (
        load_semantic_adapter_v05_checkpoint(filename, device)
    )

    hierarchy = InstructionSemanticHead(
        d_model=adapter.d_model,
        num_labels=len(HIERARCHY_LABELS),
    ).to(device)

    with torch.no_grad():
        old_count = len(V05_HIERARCHY_LABELS)
        hierarchy.linear.weight[:old_count].copy_(
            old_hierarchy.linear.weight
        )
        hierarchy.linear.bias[:old_count].copy_(
            old_hierarchy.linear.bias
        )

    return adapter, heads, hierarchy, checkpoint


def save_semantic_adapter_v06_checkpoint(
    filename,
    adapter,
    heads,
    hierarchy_head,
    *,
    epoch,
    loss,
    base_model,
    learning_rate,
    hierarchy_weight,
    instruction_contrast_weight,
    preservation_weight,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": "llm-gpu-v0.9.1-semantic-encoder-adapter-v0.6",
            "adapter_state_dict": adapter.state_dict(),
            "heads_state_dict": heads.state_dict(),
            "hierarchy_head_state_dict": hierarchy_head.state_dict(),
            "d_model": adapter.d_model,
            "adapter_hidden_dim": adapter.hidden_dim,
            "residual_scale": adapter.residual_scale,
            "concept_labels": list(CONCEPT_LABELS),
            "attribute_labels": list(ATTRIBUTE_LABELS),
            "hierarchy_labels": list(HIERARCHY_LABELS),
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "learning_rate": float(learning_rate),
            "hierarchy_weight": float(hierarchy_weight),
            "instruction_contrast_weight": float(instruction_contrast_weight),
            "preservation_weight": float(preservation_weight),
        },
        path,
    )


def load_semantic_adapter_v06_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9.1-semantic-encoder-adapter-v0.6":
        raise ValueError("Not a Semantic Encoder Adapter v0.6 checkpoint.")

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

    hierarchy = InstructionSemanticHead(
        d_model=int(checkpoint["d_model"]),
        num_labels=len(checkpoint["hierarchy_labels"]),
    ).to(device)
    hierarchy.load_state_dict(checkpoint["hierarchy_head_state_dict"])
    hierarchy.eval()

    return adapter, heads, hierarchy, checkpoint
