# output_alignment_v097.py
#
# LLM_GPU v0.9.7
# Output alignment after v0.9.6 semantic encoder adaptation.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from model import LanguageModel
from semantic_encoder_adaptation_v096 import SemanticTagHead


FORMAT = "llm-gpu-v0.9.7-output-alignment"


def configure_output_alignment(
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

    # v0.9.7 key change: allow the output geometry to follow the adapted hidden
    # representation, but train it with a much smaller learning rate.
    for p in model.lm_head.parameters():
        p.requires_grad_(True)


def save_checkpoint(
    filename: str,
    model: LanguageModel,
    semantic_head: SemanticTagHead,
    labels: Sequence[str],
    *,
    epoch: int,
    loss: float,
    source_checkpoint: str,
    freeze_through: int,
    block_learning_rate: float,
    semantic_learning_rate: float,
    lm_head_learning_rate: float,
    semantic_weight: float,
    lm_head_anchor_weight: float,
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
            "freeze_through": int(freeze_through),
            "block_learning_rate": float(block_learning_rate),
            "semantic_learning_rate": float(semantic_learning_rate),
            "lm_head_learning_rate": float(lm_head_learning_rate),
            "semantic_weight": float(semantic_weight),
            "lm_head_anchor_weight": float(lm_head_anchor_weight),
        },
        path,
    )


def load_checkpoint(filename: str, device: torch.device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a v0.9.7 Output Alignment checkpoint.")

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
