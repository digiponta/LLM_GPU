# semantic_encoder_adaptation_v096.py
#
# LLM_GPU v0.9.6
# Semantic Encoder Adaptation for the generation model itself.
#
# Blocks 1-3: frozen
# Blocks 4-6: trainable
# FinalNorm : trainable
# LM head   : frozen
# Semantic supervision head: trainable
#
# The semantic head reads the adapted prompt representation. Unlike v0.9.5,
# no frozen semantic bottleneck is placed after the base hidden state: the
# late Transformer representation itself is allowed to move.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from model import LanguageModel


FORMAT = "llm-gpu-v0.9.6-semantic-encoder-adaptation"


class SemanticTagHead(nn.Module):
    def __init__(self, d_model: int, num_labels: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, num_labels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(hidden))


def configure_semantic_adaptation(
    model: LanguageModel,
    freeze_through: int = 3,
) -> None:
    for p in model.parameters():
        p.requires_grad_(False)

    for index, block in enumerate(model.blocks, start=1):
        if index > freeze_through:
            for p in block.parameters():
                p.requires_grad_(True)

    for p in model.final_norm.parameters():
        p.requires_grad_(True)

    # Keep token identity/output geometry fixed in this experiment.
    for p in model.lm_head.parameters():
        p.requires_grad_(False)


def save_checkpoint(
    filename: str,
    model: LanguageModel,
    semantic_head: SemanticTagHead,
    labels: Sequence[str],
    *,
    epoch: int,
    loss: float,
    base_model: str,
    freeze_through: int,
    block_learning_rate: float,
    semantic_learning_rate: float,
    semantic_weight: float,
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
            "base_model": base_model,
            "freeze_through": int(freeze_through),
            "block_learning_rate": float(block_learning_rate),
            "semantic_learning_rate": float(semantic_learning_rate),
            "semantic_weight": float(semantic_weight),
        },
        path,
    )


def load_checkpoint(
    filename: str,
    device: torch.device,
):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a v0.9.6 Semantic Encoder Adaptation checkpoint.")

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
