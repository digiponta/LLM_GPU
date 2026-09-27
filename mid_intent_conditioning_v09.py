# mid_intent_conditioning_v09.py
#
# Mid-layer soft intent conditioning for LLM_GPU v0.9.
#
# Intent probabilities are inferred from the unconditioned prompt. A learned
# projection maps the 24-dimensional probability vector to d_model and injects
# it after a selected Transformer block. The remaining Transformer blocks then
# process the intent-conditioned representation.

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MidLayerIntentProjection(nn.Module):
    def __init__(
        self,
        num_labels: int,
        d_model: int,
        alpha: float = 0.1,
        inject_after: int = 3,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.d_model = int(d_model)
        self.alpha = float(alpha)
        self.inject_after = int(inject_after)
        self.projection = nn.Linear(self.num_labels, self.d_model, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, intent_prob: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.projection(intent_prob)


def validate_injection_point(model, inject_after: int) -> None:
    if inject_after < 1 or inject_after >= model.num_layers:
        raise ValueError(
            f"inject_after must be in [1, {model.num_layers - 1}], "
            f"got {inject_after}"
        )


def embedding_input(model, token_ids: torch.Tensor) -> torch.Tensor:
    if token_ids.dim() != 2:
        raise ValueError("token_ids must have shape [batch, time].")
    _batch, time = token_ids.shape
    if time > model.context_length:
        raise ValueError(
            f"Sequence length {time} exceeds context length "
            f"{model.context_length}."
        )

    x = model.embedding(token_ids)
    if model.position_embedding is not None:
        positions = torch.arange(time, device=token_ids.device)
        x = x + model.position_embedding(positions).unsqueeze(0)
    return x


def forward_mid_conditioned(
    model,
    token_ids: torch.Tensor,
    intent_bias: torch.Tensor,
    inject_after: int,
) -> torch.Tensor:
    """Return final hidden states with intent injected after N blocks."""
    validate_injection_point(model, inject_after)

    x = embedding_input(model, token_ids)

    # inject_after is 1-based: inject_after=3 means after Block 3.
    for index, block in enumerate(model.blocks, start=1):
        x = block(x)
        if index == inject_after:
            x = x + intent_bias.unsqueeze(1)

    return model.final_norm(x)


@torch.no_grad()
def infer_prompt_intent(
    model,
    intent_head: nn.Module,
    prompt_ids: torch.Tensor,
) -> torch.Tensor:
    """Use the original unconditioned v0.8 representation for intent inference."""
    hidden = model.forward_hidden(prompt_ids)
    prompt_repr = hidden[:, -1, :]
    return torch.sigmoid(intent_head(prompt_repr))


def load_mid_projection(
    filename: str,
    model,
    labels: Sequence[str],
    device: torch.device,
):
    checkpoint = torch.load(filename, map_location=device)
    stored_labels = list(checkpoint["labels"])
    if stored_labels != list(labels):
        raise ValueError("Intent label order mismatch in mid-layer checkpoint.")

    projection = MidLayerIntentProjection(
        num_labels=len(labels),
        d_model=model.d_model,
        alpha=float(checkpoint.get("alpha", 0.1)),
        inject_after=int(checkpoint.get("inject_after", 3)),
    ).to(device)
    projection.load_state_dict(checkpoint["state_dict"])
    projection.eval()
    return projection, checkpoint
