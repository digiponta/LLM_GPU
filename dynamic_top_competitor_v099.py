# dynamic_top_competitor_v099.py
#
# LLM_GPU v0.9.9
# Dynamic Top-Competitor Alignment.
#
# For entity-answer rows, the correct entity first token must beat the strongest
# non-target token over the full vocabulary by a margin. The competitor is
# recomputed from the current logits on every forward pass.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from entity_logit_alignment_v098 import ENTITY_BY_TAG
from semantic_encoder_adaptation_v096 import SemanticTagHead


FORMAT = "llm-gpu-v0.9.9-dynamic-top-competitor"


def single_entity_tag(tags: Iterable[str]):
    tagset = set(tags)
    found = [tag for tag in ENTITY_BY_TAG if tag in tagset]
    return found[0] if len(found) == 1 else None


def entity_token_ids(tokenizer):
    out = {}
    for tag, name in ENTITY_BY_TAG.items():
        ids = tokenizer.encode(name)
        if not ids:
            raise RuntimeError(f"Tokenizer produced no token for {name}.")
        out[tag] = int(ids[0])
    return out


def dynamic_top_competitor_loss(
    prompt_logits: torch.Tensor,
    entity_target: torch.Tensor,
    eligible: torch.Tensor,
    margin: float,
    ignored_token_ids: Sequence[int] = (),
):
    losses = []
    target_values = []
    competitor_values = []
    competitor_ids = []

    ignored = set(int(x) for x in ignored_token_ids)

    for row in range(prompt_logits.size(0)):
        target = int(entity_target[row].item())
        if target < 0 or not bool(eligible[row].item()):
            continue

        scores = prompt_logits[row]
        masked = scores.clone()
        masked[target] = float("-inf")
        for token_id in ignored:
            if 0 <= token_id < masked.numel() and token_id != target:
                masked[token_id] = float("-inf")

        competitor_value, competitor_id = torch.max(masked, dim=0)
        target_value = scores[target]
        losses.append(F.relu(margin + competitor_value - target_value))
        target_values.append(target_value.detach())
        competitor_values.append(competitor_value.detach())
        competitor_ids.append(competitor_id.detach())

    zero = prompt_logits.sum() * 0.0
    if not losses:
        return zero, 0, [], [], []

    return (
        torch.stack(losses).mean(),
        len(losses),
        target_values,
        competitor_values,
        competitor_ids,
    )


def save_checkpoint(
    filename,
    model,
    semantic_head: SemanticTagHead,
    labels,
    *,
    epoch,
    loss,
    source_checkpoint,
    top_margin,
    top_weight,
    lm_head_learning_rate,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "config": model.config(),
            "model_state_dict": model.state_dict(),
            "semantic_head_state_dict": semantic_head.state_dict(),
            "labels": list(labels),
            "epoch": int(epoch),
            "loss": float(loss),
            "source_checkpoint": source_checkpoint,
            "top_margin": float(top_margin),
            "top_weight": float(top_weight),
            "lm_head_learning_rate": float(lm_head_learning_rate),
        },
        path,
    )


def load_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a v0.9.9 Dynamic Top-Competitor checkpoint.")

    from model import LanguageModel

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
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = list(checkpoint["labels"])
    semantic_head = SemanticTagHead(model.d_model, len(labels)).to(device)
    semantic_head.load_state_dict(checkpoint["semantic_head_state_dict"])
    semantic_head.eval()
    return model, semantic_head, labels, checkpoint
