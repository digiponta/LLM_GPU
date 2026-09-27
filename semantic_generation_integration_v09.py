# semantic_generation_integration_v09.py
#
# v0.5 Semantic Adapter -> Generation Conditioning Integration.
#
# Frozen semantic path:
#   v0.8 encoder -> semantic adapter v0.5 -> adapted hidden (256)
#                                      -> hierarchy head -> sigmoid (5)
#   concat = 261 dims
#
# Generation path:
#   Blocks 1-3 frozen
#   261 -> 256 projection, alpha scaling, inject after Block 3
#   Blocks 4-6 trainable
#   FinalNorm trainable
#   LM head frozen

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from mid_intent_conditioning_v09 import forward_mid_conditioned
from model import LanguageModel
from semantic_encoder_adapter_v05 import load_semantic_adapter_v05_checkpoint


class SemanticGenerationProjection(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        hierarchy_dim: int,
        d_model: int,
        alpha: float = 0.1,
        inject_after: int = 3,
    ):
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.hierarchy_dim = int(hierarchy_dim)
        self.input_dim = self.semantic_dim + self.hierarchy_dim
        self.d_model = int(d_model)
        self.alpha = float(alpha)
        self.inject_after = int(inject_after)
        self.projection = nn.Linear(self.input_dim, self.d_model, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(
        self,
        adapted_hidden: torch.Tensor,
        hierarchy_prob: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([adapted_hidden, hierarchy_prob], dim=-1)
        return self.alpha * self.projection(fused)


@torch.no_grad()
def infer_semantic_condition(
    semantic_model,
    semantic_adapter,
    hierarchy_head,
    token_ids: torch.Tensor,
    prompt_index: torch.Tensor | None = None,
):
    hidden = semantic_model.forward_hidden(token_ids)
    if prompt_index is None:
        prompt_hidden = hidden[:, -1, :]
    else:
        batch = torch.arange(hidden.size(0), device=hidden.device)
        prompt_hidden = hidden[batch, prompt_index]

    adapted = semantic_adapter(prompt_hidden)
    hierarchy_prob = torch.sigmoid(hierarchy_head(adapted))
    return adapted, hierarchy_prob


def configure_generation_model(model: LanguageModel, inject_after: int):
    for p in model.parameters():
        p.requires_grad_(False)

    for index, block in enumerate(model.blocks, start=1):
        if index > inject_after:
            for p in block.parameters():
                p.requires_grad_(True)

    for p in model.final_norm.parameters():
        p.requires_grad_(True)

    for p in model.lm_head.parameters():
        p.requires_grad_(False)


def forward_semantic_conditioned(
    model,
    token_ids,
    semantic_bias,
    inject_after,
):
    return forward_mid_conditioned(
        model,
        token_ids,
        semantic_bias,
        inject_after,
    )


def save_integration_checkpoint(
    filename,
    generation_model,
    projection,
    *,
    epoch,
    loss,
    base_model,
    semantic_adapter_checkpoint,
    block_learning_rate,
    projection_learning_rate,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "llm-gpu-v0.9-semantic-generation-integration",
            "config": generation_model.config(),
            "model_state_dict": generation_model.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "semantic_dim": projection.semantic_dim,
            "hierarchy_dim": projection.hierarchy_dim,
            "alpha": projection.alpha,
            "inject_after": projection.inject_after,
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "semantic_adapter_checkpoint": semantic_adapter_checkpoint,
            "block_learning_rate": float(block_learning_rate),
            "projection_learning_rate": float(projection_learning_rate),
        },
        path,
    )


def load_integration_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9-semantic-generation-integration":
        raise ValueError("Not a semantic generation integration checkpoint.")

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

    projection = SemanticGenerationProjection(
        semantic_dim=int(checkpoint["semantic_dim"]),
        hierarchy_dim=int(checkpoint["hierarchy_dim"]),
        d_model=model.d_model,
        alpha=float(checkpoint.get("alpha", 0.1)),
        inject_after=int(checkpoint.get("inject_after", 3)),
    ).to(device)
    projection.load_state_dict(checkpoint["projection_state_dict"])
    projection.eval()

    return model, projection, checkpoint


def load_frozen_semantic_path(base_model_path, adapter_path, device):
    semantic_model, base_checkpoint = LanguageModel.load_checkpoint(
        base_model_path,
        device=device,
    )
    semantic_adapter, semantic_heads, hierarchy_head, adapter_checkpoint = (
        load_semantic_adapter_v05_checkpoint(adapter_path, device)
    )

    semantic_model.eval()
    semantic_adapter.eval()
    semantic_heads.eval()
    hierarchy_head.eval()

    for module in (
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
    ):
        for p in module.parameters():
            p.requires_grad_(False)

    return (
        semantic_model,
        semantic_adapter,
        hierarchy_head,
        base_checkpoint,
        adapter_checkpoint,
    )
