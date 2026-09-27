# intent_conditioning_v09.py
#
# Soft intent conditioning for LLM_GPU v0.9.
# The v0.8 language model and intent head can remain frozen while a small
# projection learns how to inject the 24-dimensional soft intent probability
# vector into the 256-dimensional LM hidden representation.

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


def build_intent_head(d_model: int, num_labels: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, num_labels),
    )


class SoftIntentProjection(nn.Module):
    def __init__(self, num_labels: int, d_model: int, alpha: float = 0.1):
        super().__init__()
        self.num_labels = int(num_labels)
        self.d_model = int(d_model)
        self.alpha = float(alpha)
        self.projection = nn.Linear(self.num_labels, self.d_model, bias=False)

        # Start as an exact no-op. At step zero v0.9 therefore reproduces the
        # unconditioned v0.8 hidden state.
        nn.init.zeros_(self.projection.weight)

    def forward(self, intent_prob: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.projection(intent_prob)


@torch.no_grad()
def infer_soft_intent(
    model,
    intent_head: nn.Module,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden = model.forward_hidden(input_ids)
    prompt_repr = hidden[:, -1, :]
    intent_logits = intent_head(prompt_repr)
    intent_prob = torch.sigmoid(intent_logits)
    return intent_prob, intent_logits


def load_intent_head(
    filename: str,
    model,
    device: torch.device,
):
    checkpoint = torch.load(filename, map_location=device)
    labels: Sequence[str] = list(checkpoint["labels"])
    d_model = int(checkpoint.get("d_model", model.d_model))
    if d_model != model.d_model:
        raise ValueError(
            f"Intent-head/model d_model mismatch: {d_model} != {model.d_model}"
        )
    head = build_intent_head(model.d_model, len(labels)).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    return head, checkpoint, list(labels)


def load_projection(
    filename: str,
    model,
    labels: Sequence[str],
    device: torch.device,
):
    checkpoint = torch.load(filename, map_location=device)
    stored_labels = list(checkpoint["labels"])
    if stored_labels != list(labels):
        raise ValueError("Intent label order mismatch in projection checkpoint.")
    projection = SoftIntentProjection(
        num_labels=len(labels),
        d_model=model.d_model,
        alpha=float(checkpoint.get("alpha", 0.1)),
    ).to(device)
    projection.load_state_dict(checkpoint["state_dict"])
    projection.eval()
    return projection, checkpoint
