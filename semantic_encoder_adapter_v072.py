# semantic_encoder_adapter_v072.py
#
# LLM_GPU v0.9.1 Semantic Encoder Adapter v0.7.2
# Program Composition Hierarchy

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
from semantic_encoder_adapter_v071 import (
    load_semantic_adapter_v071_checkpoint,
)
from semantic_encoder_adapter_v07 import (
    SEMANTIC_HIERARCHY_LABELS as V071_LABELS,
)


SEMANTIC_HIERARCHY_LABELS = [
    "processor",
    "program_execution",
    "instruction_sequence",
    "instruction_execution",
    "computation",
    "arithmetic_logic",
    "control_flow",
    "memory_operation",
    "data_movement",
    "heterogeneous_instruction_stream",
    "repeated_computation",
    "general_purpose",
    "control_oriented",
    "throughput_oriented",
    "data_parallel",
]


class ProgramCompositionHead(nn.Module):
    def __init__(self, d_model=256, num_labels=len(SEMANTIC_HIERARCHY_LABELS)):
        super().__init__()
        self.linear = nn.Linear(d_model, num_labels)

    def forward(self, hidden):
        return self.linear(hidden)


def initialize_from_v071(filename, device):
    adapter, heads, old_head, checkpoint = load_semantic_adapter_v071_checkpoint(
        filename, device
    )
    new_head = ProgramCompositionHead(adapter.d_model).to(device)

    old_labels = list(checkpoint["hierarchy_labels"])
    with torch.no_grad():
        for old_name in old_labels:
            if old_name in SEMANTIC_HIERARCHY_LABELS:
                oi = old_labels.index(old_name)
                ni = SEMANTIC_HIERARCHY_LABELS.index(old_name)
                new_head.linear.weight[ni].copy_(old_head.linear.weight[oi])
                new_head.linear.bias[ni].copy_(old_head.linear.bias[oi])

    return adapter, heads, new_head, checkpoint


def save_semantic_adapter_v072_checkpoint(
    filename,
    adapter,
    heads,
    hierarchy_head,
    *,
    epoch,
    loss,
    base_model,
    init_adapter,
    learning_rate,
    hierarchy_weight,
    relation_weight,
    preservation_weight,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "llm-gpu-v0.9.1-semantic-encoder-adapter-v0.7.2",
            "adapter_state_dict": adapter.state_dict(),
            "heads_state_dict": heads.state_dict(),
            "hierarchy_head_state_dict": hierarchy_head.state_dict(),
            "d_model": adapter.d_model,
            "adapter_hidden_dim": adapter.hidden_dim,
            "residual_scale": adapter.residual_scale,
            "concept_labels": list(CONCEPT_LABELS),
            "attribute_labels": list(ATTRIBUTE_LABELS),
            "hierarchy_labels": list(SEMANTIC_HIERARCHY_LABELS),
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "init_adapter": init_adapter,
            "learning_rate": float(learning_rate),
            "hierarchy_weight": float(hierarchy_weight),
            "relation_weight": float(relation_weight),
            "preservation_weight": float(preservation_weight),
        },
        path,
    )


def load_semantic_adapter_v072_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9.1-semantic-encoder-adapter-v0.7.2":
        raise ValueError("Not a Semantic Encoder Adapter v0.7.2 checkpoint.")

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

    hierarchy_head = ProgramCompositionHead(
        d_model=int(checkpoint["d_model"]),
        num_labels=len(checkpoint["hierarchy_labels"]),
    ).to(device)
    hierarchy_head.load_state_dict(checkpoint["hierarchy_head_state_dict"])
    hierarchy_head.eval()

    return adapter, heads, hierarchy_head, checkpoint
