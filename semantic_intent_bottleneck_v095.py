# semantic_intent_bottleneck_v095.py
#
# LLM_GPU v0.9.5
# Semantic Intent Bottleneck -> Generation Coupling
#
# Prompt hidden (256)
#   -> LayerNorm -> Linear(256, 64) -> GELU
#   -> semantic bottleneck (64)
#       |-> semantic tag head (24), supervised during training
#       |
#       + frozen v0.9.2 intent probabilities (24)
#       + learned semantic tag probabilities (24)
#       -> fused 112-d representation
#       -> MLP 112 -> 128 -> 512
#       -> FiLM gamma/beta after Block 3
#
# The FiLM output layer is zero initialized, so step zero is an exact no-op.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mid_intent_conditioning_v09 import embedding_input, validate_injection_point


FORMAT = "llm-gpu-v0.9.5-semantic-intent-bottleneck"


class SemanticIntentBottleneck(nn.Module):
    def __init__(
        self,
        num_labels: int,
        d_model: int,
        semantic_dim: int = 64,
        fusion_dim: int = 128,
        inject_after: int = 3,
        scale_limit: float = 0.50,
        shift_limit: float = 0.50,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.d_model = int(d_model)
        self.semantic_dim = int(semantic_dim)
        self.fusion_dim = int(fusion_dim)
        self.inject_after = int(inject_after)
        self.scale_limit = float(scale_limit)
        self.shift_limit = float(shift_limit)

        self.semantic_norm = nn.LayerNorm(self.d_model)
        self.semantic_in = nn.Linear(self.d_model, self.semantic_dim)
        self.semantic_head = nn.Linear(self.semantic_dim, self.num_labels)

        fused_dim = self.semantic_dim + 2 * self.num_labels
        self.fusion1 = nn.Linear(fused_dim, self.fusion_dim)
        self.fusion2 = nn.Linear(self.fusion_dim, 2 * self.d_model)

        nn.init.xavier_uniform_(self.semantic_in.weight)
        nn.init.zeros_(self.semantic_in.bias)
        nn.init.xavier_uniform_(self.semantic_head.weight)
        nn.init.zeros_(self.semantic_head.bias)
        nn.init.xavier_uniform_(self.fusion1.weight)
        nn.init.zeros_(self.fusion1.bias)
        nn.init.zeros_(self.fusion2.weight)
        nn.init.zeros_(self.fusion2.bias)

    def semantic(self, prompt_hidden: torch.Tensor):
        z = F.gelu(self.semantic_in(self.semantic_norm(prompt_hidden)))
        logits = self.semantic_head(z)
        probs = torch.sigmoid(logits)
        return z, logits, probs

    def forward(
        self,
        prompt_hidden: torch.Tensor,
        frozen_intent_prob: torch.Tensor,
    ):
        z, semantic_logits, semantic_prob = self.semantic(prompt_hidden)
        fused = torch.cat(
            [z, semantic_prob, frozen_intent_prob],
            dim=-1,
        )
        h = F.gelu(self.fusion1(fused))
        gamma, beta = self.fusion2(h).chunk(2, dim=-1)
        scale = self.scale_limit * torch.tanh(gamma)
        shift = self.shift_limit * torch.tanh(beta)
        return scale, shift, semantic_logits, semantic_prob, z


def forward_semantic_intent(
    model,
    token_ids: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    inject_after: int,
):
    validate_injection_point(model, inject_after)
    x = embedding_input(model, token_ids)

    for index, block in enumerate(model.blocks, start=1):
        x = block(x)
        if index == inject_after:
            x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return model.final_norm(x)


def save_checkpoint(
    filename,
    coupling,
    labels: Sequence[str],
    *,
    epoch,
    loss,
    base_model,
    intent_head,
    learning_rate,
    semantic_weight,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "labels": list(labels),
            "d_model": coupling.d_model,
            "semantic_dim": coupling.semantic_dim,
            "fusion_dim": coupling.fusion_dim,
            "inject_after": coupling.inject_after,
            "scale_limit": coupling.scale_limit,
            "shift_limit": coupling.shift_limit,
            "state_dict": coupling.state_dict(),
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "intent_head": intent_head,
            "learning_rate": float(learning_rate),
            "semantic_weight": float(semantic_weight),
        },
        path,
    )


def load_checkpoint(filename, model, labels: Sequence[str], device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a v0.9.5 Semantic Intent Bottleneck checkpoint.")
    if list(checkpoint["labels"]) != list(labels):
        raise ValueError("Intent label order mismatch.")

    coupling = SemanticIntentBottleneck(
        num_labels=len(labels),
        d_model=model.d_model,
        semantic_dim=int(checkpoint.get("semantic_dim", 64)),
        fusion_dim=int(checkpoint.get("fusion_dim", 128)),
        inject_after=int(checkpoint.get("inject_after", 3)),
        scale_limit=float(checkpoint.get("scale_limit", 0.50)),
        shift_limit=float(checkpoint.get("shift_limit", 0.50)),
    ).to(device)
    coupling.load_state_dict(checkpoint["state_dict"])
    coupling.eval()
    return coupling, checkpoint
