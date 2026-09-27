# entity_contrastive_logit_alignment_v0122.py
#
# LLM_GPU v0.9.1 v0.12.2
# Entity Contrastive Logit Alignment
#
# Explicitly trains relative ordering among technical entity logits and adds
# an entity gate to suppress intervention on non-entity prompts.

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from semantic_lexical_logit_alignment_v012 import SemanticLexicalLogitAdapter
from entity_target_logit_alignment_v0121 import ENTITY_TARGETS


FORMAT = "llm-gpu-v0.9.1-entity-contrastive-logit-alignment-v0.12.2"


class EntityGate(nn.Module):
    def __init__(self, input_dim=345, hidden_dim=64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, condition):
        return torch.sigmoid(self.net(condition)).squeeze(-1)


def save_checkpoint(
    filename,
    adapter,
    gate,
    *,
    epoch,
    loss,
    v011_checkpoint,
    semantic_adapter_checkpoint,
    name_binding_checkpoint,
    learning_rate,
    margin,
    gate_weight,
    l2_weight,
    target_token_ids,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "adapter_state_dict": adapter.state_dict(),
            "gate_state_dict": gate.state_dict(),
            "input_dim": adapter.input_dim,
            "vocab_size": adapter.vocab_size,
            "rank": adapter.rank,
            "beta": adapter.beta,
            "gate_hidden_dim": gate.hidden_dim,
            "epoch": int(epoch),
            "loss": float(loss),
            "v011_checkpoint": v011_checkpoint,
            "semantic_adapter_checkpoint": semantic_adapter_checkpoint,
            "name_binding_checkpoint": name_binding_checkpoint,
            "learning_rate": float(learning_rate),
            "margin": float(margin),
            "gate_weight": float(gate_weight),
            "l2_weight": float(l2_weight),
            "entity_targets": dict(ENTITY_TARGETS),
            "target_token_ids": dict(target_token_ids),
        },
        path,
    )


def load_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not an Entity Contrastive Logit Alignment v0.12.2 checkpoint.")

    adapter = SemanticLexicalLogitAdapter(
        input_dim=int(checkpoint["input_dim"]),
        vocab_size=int(checkpoint["vocab_size"]),
        rank=int(checkpoint.get("rank", 64)),
        beta=float(checkpoint.get("beta", 0.30)),
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()

    gate = EntityGate(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint.get("gate_hidden_dim", 64)),
    ).to(device)
    gate.load_state_dict(checkpoint["gate_state_dict"])
    gate.eval()
    return adapter, gate, checkpoint
