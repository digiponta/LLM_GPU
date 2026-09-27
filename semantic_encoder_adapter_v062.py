# semantic_encoder_adapter_v062.py
#
# LLM_GPU v0.9.1 Semantic Encoder Adapter v0.6.2
# G05-Neighborhood Binding

from __future__ import annotations

from pathlib import Path
import torch

from semantic_encoder_adapter_v01 import (
    ATTRIBUTE_LABELS,
    CONCEPT_LABELS,
    SemanticEncoderAdapter,
    SemanticSupervisionHeads,
)
from semantic_encoder_adapter_v06 import (
    HIERARCHY_LABELS,
    InstructionSemanticHead,
)


def save_semantic_adapter_v062_checkpoint(
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
    neighborhood_weight,
    neighborhood_margin,
    preservation_weight,
):
    path=Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format":"llm-gpu-v0.9.1-semantic-encoder-adapter-v0.6.2",
        "adapter_state_dict":adapter.state_dict(),
        "heads_state_dict":heads.state_dict(),
        "hierarchy_head_state_dict":hierarchy_head.state_dict(),
        "d_model":adapter.d_model,
        "adapter_hidden_dim":adapter.hidden_dim,
        "residual_scale":adapter.residual_scale,
        "concept_labels":list(CONCEPT_LABELS),
        "attribute_labels":list(ATTRIBUTE_LABELS),
        "hierarchy_labels":list(HIERARCHY_LABELS),
        "epoch":int(epoch),
        "loss":float(loss),
        "base_model":base_model,
        "init_adapter":init_adapter,
        "learning_rate":float(learning_rate),
        "neighborhood_weight":float(neighborhood_weight),
        "neighborhood_margin":float(neighborhood_margin),
        "preservation_weight":float(preservation_weight),
    },path)


def load_semantic_adapter_v062_checkpoint(filename,device):
    checkpoint=torch.load(filename,map_location=device)
    if checkpoint.get("format")!="llm-gpu-v0.9.1-semantic-encoder-adapter-v0.6.2":
        raise ValueError("Not a Semantic Encoder Adapter v0.6.2 checkpoint.")

    adapter=SemanticEncoderAdapter(
        d_model=int(checkpoint["d_model"]),
        hidden_dim=int(checkpoint.get("adapter_hidden_dim",64)),
        residual_scale=float(checkpoint.get("residual_scale",0.25)),
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()

    heads=SemanticSupervisionHeads(
        d_model=int(checkpoint["d_model"]),
        num_concepts=len(checkpoint["concept_labels"]),
        num_attributes=len(checkpoint["attribute_labels"]),
    ).to(device)
    heads.load_state_dict(checkpoint["heads_state_dict"])
    heads.eval()

    hierarchy_head=InstructionSemanticHead(
        d_model=int(checkpoint["d_model"]),
        num_labels=len(checkpoint["hierarchy_labels"]),
    ).to(device)
    hierarchy_head.load_state_dict(checkpoint["hierarchy_head_state_dict"])
    hierarchy_head.eval()
    return adapter,heads,hierarchy_head,checkpoint
