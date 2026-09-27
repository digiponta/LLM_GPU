# partial_intent_conditioning_v09.py
#
# Partial fine-tuning utilities for LLM_GPU v0.9.
#
# Design:
#   intent model : frozen v0.8 pairwise-best model
#   intent head  : frozen
#   generation model:
#       Blocks 1-3 frozen
#       Blocks 4-6 trainable
#       FinalNorm trainable
#       LM head frozen
#   intent projection: trainable
#
# The intent model is kept separate from the generation model so the frozen
# intent head always sees the same representation distribution it was trained
# on.

from __future__ import annotations

from typing import Sequence

import torch

from mid_intent_conditioning_v09 import (
    MidLayerIntentProjection,
    forward_mid_conditioned,
)
from model import LanguageModel


def configure_partial_finetune(
    model: LanguageModel,
    inject_after: int,
) -> None:
    """Freeze everything except blocks after injection and final norm."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for index, block in enumerate(model.blocks, start=1):
        if index > inject_after:
            for parameter in block.parameters():
                parameter.requires_grad_(True)

    for parameter in model.final_norm.parameters():
        parameter.requires_grad_(True)

    # LM head intentionally remains frozen in the first partial-FT experiment.
    for parameter in model.lm_head.parameters():
        parameter.requires_grad_(False)


def trainable_model_parameters(model: LanguageModel):
    return [p for p in model.parameters() if p.requires_grad]


def trainable_parameter_count(model: LanguageModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def infer_intent_prob(
    intent_model: LanguageModel,
    intent_head,
    token_ids: torch.Tensor,
    prompt_index: torch.Tensor | None = None,
) -> torch.Tensor:
    hidden = intent_model.forward_hidden(token_ids)

    if prompt_index is None:
        prompt_repr = hidden[:, -1, :]
    else:
        batch = torch.arange(hidden.size(0), device=hidden.device)
        prompt_repr = hidden[batch, prompt_index]

    return torch.sigmoid(intent_head(prompt_repr))


def forward_partial_conditioned(
    generation_model: LanguageModel,
    token_ids: torch.Tensor,
    intent_bias: torch.Tensor,
    inject_after: int,
) -> torch.Tensor:
    return forward_mid_conditioned(
        generation_model,
        token_ids,
        intent_bias,
        inject_after,
    )


def save_partial_checkpoint(
    filename: str,
    generation_model: LanguageModel,
    projection: MidLayerIntentProjection,
    labels: Sequence[str],
    *,
    epoch: int,
    loss: float,
    base_model: str,
    intent_head: str,
    block_learning_rate: float,
    projection_learning_rate: float,
) -> None:
    from pathlib import Path

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": "llm-gpu-v0.9-partial-intent",
            "config": generation_model.config(),
            "model_state_dict": generation_model.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "labels": list(labels),
            "d_model": generation_model.d_model,
            "alpha": projection.alpha,
            "inject_after": projection.inject_after,
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "intent_head": intent_head,
            "block_learning_rate": float(block_learning_rate),
            "projection_learning_rate": float(projection_learning_rate),
            "lm_head_trainable": False,
        },
        path,
    )


def load_partial_checkpoint(
    filename: str,
    device: torch.device,
    labels: Sequence[str],
):
    checkpoint = torch.load(filename, map_location=device)
    if list(checkpoint["labels"]) != list(labels):
        raise ValueError("Intent label order mismatch in partial checkpoint.")

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

    projection = MidLayerIntentProjection(
        num_labels=len(labels),
        d_model=model.d_model,
        alpha=float(checkpoint.get("alpha", 0.1)),
        inject_after=int(checkpoint.get("inject_after", 3)),
    ).to(device)
    projection.load_state_dict(checkpoint["projection_state_dict"])
    projection.eval()

    return model, projection, checkpoint
