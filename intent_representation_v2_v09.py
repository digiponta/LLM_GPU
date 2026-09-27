# intent_representation_v2_v09.py
#
# LLM_GPU v0.9 Intent Representation v2
#
# Conditioning representation:
#   [24-d intent probabilities ; 256-d frozen prompt semantic hidden]
#       -> zero-initialized Linear(280 -> 256)
#       -> alpha scaling
#       -> inject after Transformer Block 3
#
# The semantic hidden state is taken from the same frozen v0.8 intent model
# used by the intent head, so the representation remains stationary while the
# generation-side late blocks are fine-tuned.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from mid_intent_conditioning_v09 import forward_mid_conditioned
from model import LanguageModel


class IntentRepresentationV2Projection(nn.Module):
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
        self.input_dim = self.num_labels + self.d_model
        self.projection = nn.Linear(self.input_dim, self.d_model, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(
        self,
        intent_prob: torch.Tensor,
        semantic_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if intent_prob.dim() != 2:
            raise ValueError("intent_prob must have shape [batch, labels].")
        if semantic_hidden.dim() != 2:
            raise ValueError("semantic_hidden must have shape [batch, d_model].")
        fused = torch.cat([intent_prob, semantic_hidden], dim=-1)
        return self.alpha * self.projection(fused)


@torch.no_grad()
def infer_intent_representation_v2(
    intent_model: LanguageModel,
    intent_head,
    token_ids: torch.Tensor,
    prompt_index: torch.Tensor | None = None,
):
    hidden = intent_model.forward_hidden(token_ids)

    if prompt_index is None:
        semantic_hidden = hidden[:, -1, :]
    else:
        batch = torch.arange(hidden.size(0), device=hidden.device)
        semantic_hidden = hidden[batch, prompt_index]

    intent_prob = torch.sigmoid(intent_head(semantic_hidden))
    return intent_prob, semantic_hidden


def forward_representation_v2_conditioned(
    generation_model: LanguageModel,
    token_ids: torch.Tensor,
    conditioning_bias: torch.Tensor,
    inject_after: int,
) -> torch.Tensor:
    return forward_mid_conditioned(
        generation_model,
        token_ids,
        conditioning_bias,
        inject_after,
    )


def save_representation_v2_checkpoint(
    filename: str,
    generation_model: LanguageModel,
    projection: IntentRepresentationV2Projection,
    labels: Sequence[str],
    *,
    epoch: int,
    loss: float,
    base_model: str,
    intent_head: str,
    block_learning_rate: float,
    projection_learning_rate: float,
) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": "llm-gpu-v0.9-intent-representation-v2",
            "config": generation_model.config(),
            "model_state_dict": generation_model.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "labels": list(labels),
            "num_labels": projection.num_labels,
            "d_model": projection.d_model,
            "representation_dim": projection.input_dim,
            "alpha": projection.alpha,
            "inject_after": projection.inject_after,
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "intent_head": intent_head,
            "block_learning_rate": float(block_learning_rate),
            "projection_learning_rate": float(projection_learning_rate),
            "lm_head_trainable": False,
            "semantic_hidden_source": "frozen-intent-model-prompt-state",
        },
        path,
    )


def load_representation_v2_checkpoint(
    filename: str,
    device: torch.device,
    labels: Sequence[str],
):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9-intent-representation-v2":
        raise ValueError("Not an Intent Representation v2 checkpoint.")
    if list(checkpoint["labels"]) != list(labels):
        raise ValueError("Intent label order mismatch.")

    config = checkpoint["config"]
    model = LanguageModel(
        vocab_size=int(config["vocab_size"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        hidden_dim=int(config["hidden_dim"]),
        num_heads=int(config.get("num_heads", 1)),
        causal=bool(config.get("causal", True)),
        context_length=int(config.get("context_length", 64)),
        use_position_embedding=bool(config.get("use_position_embedding", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    projection = IntentRepresentationV2Projection(
        num_labels=len(labels),
        d_model=model.d_model,
        alpha=float(checkpoint.get("alpha", 0.1)),
        inject_after=int(checkpoint.get("inject_after", 3)),
    ).to(device)
    projection.load_state_dict(checkpoint["projection_state_dict"])
    projection.eval()

    return model, projection, checkpoint
