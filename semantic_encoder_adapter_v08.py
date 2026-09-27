# semantic_encoder_adapter_v08.py
#
# LLM_GPU v0.9.1 Semantic Encoder Adapter v0.8
# Hierarchy-Constrained Semantic Head
#
# Marginal probabilities are constructed from conditional probabilities:
#   P(child) = P(parent) * P(child | parent)
#
# The head returns logits corresponding to these constrained marginals, so
# existing callers may continue to use sigmoid(head(hidden)).

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
from semantic_encoder_adapter_v072 import (
    SEMANTIC_HIERARCHY_LABELS,
    load_semantic_adapter_v072_checkpoint,
)


FORMAT = "llm-gpu-v0.9.1-semantic-encoder-adapter-v0.8"

PARENT = {
    "processor": None,
    "program_execution": "processor",
    "instruction_sequence": "program_execution",
    "instruction_execution": "instruction_sequence",
    "computation": "instruction_execution",
    "arithmetic_logic": "computation",
    "control_flow": "instruction_execution",
    "memory_operation": "instruction_execution",
    "data_movement": "instruction_execution",
    "heterogeneous_instruction_stream": "instruction_sequence",
    "repeated_computation": "computation",
    "general_purpose": "processor",
    "control_oriented": "processor",
    "throughput_oriented": "processor",
    "data_parallel": "processor",
}


class HierarchyConstrainedSemanticHead(nn.Module):
    def __init__(
        self,
        d_model=256,
        labels=SEMANTIC_HIERARCHY_LABELS,
    ):
        super().__init__()
        self.labels = list(labels)
        self.local = nn.Linear(d_model, len(self.labels))

    def marginal_probabilities(self, hidden):
        local_prob = torch.sigmoid(self.local(hidden))
        values = {}

        for i, label in enumerate(self.labels):
            parent = PARENT[label]
            if parent is None:
                values[label] = local_prob[..., i]
            else:
                values[label] = values[parent] * local_prob[..., i]

        return torch.stack(
            [values[label] for label in self.labels],
            dim=-1,
        )

    def forward(self, hidden):
        probs = self.marginal_probabilities(hidden)
        eps = torch.finfo(probs.dtype).eps
        return torch.logit(probs.clamp(eps, 1.0 - eps))


def initialize_from_v072(filename, device):
    adapter, heads, old_head, checkpoint = load_semantic_adapter_v072_checkpoint(
        filename,
        device,
    )
    head = HierarchyConstrainedSemanticHead(
        d_model=adapter.d_model,
        labels=SEMANTIC_HIERARCHY_LABELS,
    ).to(device)

    old_labels = list(checkpoint["hierarchy_labels"])
    with torch.no_grad():
        for label in SEMANTIC_HIERARCHY_LABELS:
            if label not in old_labels:
                continue
            oi = old_labels.index(label)
            ni = SEMANTIC_HIERARCHY_LABELS.index(label)
            head.local.weight[ni].copy_(old_head.linear.weight[oi])
            head.local.bias[ni].copy_(old_head.linear.bias[oi])

        # Children represent conditional probabilities in v0.8. A small
        # positive bias keeps initial child marginals from collapsing solely
        # because of multiplication through several levels.
        for label in SEMANTIC_HIERARCHY_LABELS:
            if PARENT[label] is not None:
                ni = SEMANTIC_HIERARCHY_LABELS.index(label)
                head.local.bias[ni].add_(1.5)

    return adapter, heads, head, checkpoint


def save_semantic_adapter_v08_checkpoint(
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
    preservation_weight,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": FORMAT,
            "adapter_state_dict": adapter.state_dict(),
            "heads_state_dict": heads.state_dict(),
            "hierarchy_head_state_dict": hierarchy_head.state_dict(),
            "d_model": adapter.d_model,
            "adapter_hidden_dim": adapter.hidden_dim,
            "residual_scale": adapter.residual_scale,
            "concept_labels": list(CONCEPT_LABELS),
            "attribute_labels": list(ATTRIBUTE_LABELS),
            "hierarchy_labels": list(SEMANTIC_HIERARCHY_LABELS),
            "parent_map": dict(PARENT),
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "init_adapter": init_adapter,
            "learning_rate": float(learning_rate),
            "hierarchy_weight": float(hierarchy_weight),
            "preservation_weight": float(preservation_weight),
        },
        path,
    )


def load_semantic_adapter_v08_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a Semantic Encoder Adapter v0.8 checkpoint.")

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

    hierarchy_head = HierarchyConstrainedSemanticHead(
        d_model=int(checkpoint["d_model"]),
        labels=checkpoint["hierarchy_labels"],
    ).to(device)
    hierarchy_head.load_state_dict(checkpoint["hierarchy_head_state_dict"])
    hierarchy_head.eval()

    return adapter, heads, hierarchy_head, checkpoint
