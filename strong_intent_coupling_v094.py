# strong_intent_coupling_v094.py
#
# LLM_GPU v0.9.4
# Strong Intent -> Generation Coupling
#
# Replace the weak additive 24 -> 256 bias with a zero-initialized FiLM-style
# modulation injected after a selected Transformer block:
#
#   intent_prob (24)
#       -> Linear(24, hidden) -> GELU
#       -> Linear(hidden, 2*d_model)
#       -> gamma, beta
#
#   x' = x * (1 + scale_limit * tanh(gamma))
#        + shift_limit * tanh(beta)
#
# The final layer is zero-initialized, so step zero is an exact no-op.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mid_intent_conditioning_v09 import embedding_input, validate_injection_point


FORMAT = "llm-gpu-v0.9.4-strong-intent-film"


class StrongIntentFiLM(nn.Module):
    def __init__(
        self,
        num_labels: int,
        d_model: int,
        hidden_dim: int = 64,
        inject_after: int = 3,
        scale_limit: float = 0.50,
        shift_limit: float = 0.50,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.inject_after = int(inject_after)
        self.scale_limit = float(scale_limit)
        self.shift_limit = float(shift_limit)

        self.fc1 = nn.Linear(self.num_labels, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 2 * self.d_model)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, intent_prob: torch.Tensor):
        h = F.gelu(self.fc1(intent_prob))
        gamma, beta = self.fc2(h).chunk(2, dim=-1)
        scale = self.scale_limit * torch.tanh(gamma)
        shift = self.shift_limit * torch.tanh(beta)
        return scale, shift


def forward_strong_intent(
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
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "labels": list(labels),
            "d_model": coupling.d_model,
            "hidden_dim": coupling.hidden_dim,
            "inject_after": coupling.inject_after,
            "scale_limit": coupling.scale_limit,
            "shift_limit": coupling.shift_limit,
            "state_dict": coupling.state_dict(),
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "intent_head": intent_head,
            "learning_rate": float(learning_rate),
        },
        path,
    )


def load_checkpoint(filename, model, labels: Sequence[str], device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a v0.9.4 Strong Intent FiLM checkpoint.")
    if list(checkpoint["labels"]) != list(labels):
        raise ValueError("Intent label order mismatch.")

    coupling = StrongIntentFiLM(
        num_labels=len(labels),
        d_model=model.d_model,
        hidden_dim=int(checkpoint.get("hidden_dim", 64)),
        inject_after=int(checkpoint.get("inject_after", 3)),
        scale_limit=float(checkpoint.get("scale_limit", 0.50)),
        shift_limit=float(checkpoint.get("shift_limit", 0.50)),
    ).to(device)
    coupling.load_state_dict(checkpoint["state_dict"])
    coupling.eval()
    return coupling, checkpoint
