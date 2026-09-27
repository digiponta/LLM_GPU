# semantic_cross_attention_v091.py
#
# LLM_GPU v0.9.1
# Semantic-to-Generation Interface v0.9: Semantic Cross-Attention
#
# Q   = text hidden after Block 3
# K,V = 3 balanced semantic tokens from v0.8
#
# Flow:
#   text -> Blocks 1-3
#        -> Semantic Cross-Attention(text <- semantic)
#        -> residual
#        -> Blocks 4-6
#        -> FinalNorm
#        -> LM Head
#
# The balanced semantic-token projector is loaded from v0.8 and frozen.

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LanguageModel
from semantic_token_balanced_v091 import BalancedSemanticTokenProjector


class SemanticCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        initial_residual_scale: float = 0.1,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if not (0.0 < initial_residual_scale < 1.0):
            raise ValueError("initial_residual_scale must be in (0, 1).")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads

        self.query_norm = nn.LayerNorm(self.d_model)
        self.semantic_norm = nn.LayerNorm(self.d_model)

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        initial_logit = math.log(
            initial_residual_scale / (1.0 - initial_residual_scale)
        )
        self.residual_logit = nn.Parameter(
            torch.tensor(initial_logit, dtype=torch.float32)
        )

    def _split_heads(self, x):
        batch, time, _ = x.shape
        x = x.view(batch, time, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def residual_scale(self):
        return torch.sigmoid(self.residual_logit)

    def forward(
        self,
        text_hidden: torch.Tensor,
        semantic_tokens: torch.Tensor,
        return_weights: bool = False,
    ):
        q_input = self.query_norm(text_hidden)
        kv_input = self.semantic_norm(semantic_tokens)

        q = self._split_heads(self.q_proj(q_input))
        k = self._split_heads(self.k_proj(kv_input))
        v = self._split_heads(self.v_proj(kv_input))

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(float(self.head_dim))
        weights = F.softmax(scores, dim=-1)

        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(
            text_hidden.size(0),
            text_hidden.size(1),
            self.d_model,
        )

        residual = self.out_proj(context)
        output = text_hidden + self.residual_scale() * residual

        if return_weights:
            return output, weights
        return output


def configure_cross_attention_generation_model(
    model: LanguageModel,
    inject_after: int,
):
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


def _initial_hidden(model: LanguageModel, token_ids: torch.Tensor):
    if token_ids.dim() != 2:
        raise ValueError("token_ids must have shape [batch, time].")

    _, time = token_ids.shape
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


def forward_semantic_cross_attention(
    model: LanguageModel,
    cross_attention: SemanticCrossAttention,
    token_ids: torch.Tensor,
    semantic_tokens: torch.Tensor,
    inject_after: int,
    return_cross_weights: bool = False,
):
    x = _initial_hidden(model, token_ids)

    for index, block in enumerate(model.blocks, start=1):
        if index <= inject_after:
            x = block(x)

    if return_cross_weights:
        x, weights = cross_attention(
            x,
            semantic_tokens,
            return_weights=True,
        )
    else:
        x = cross_attention(x, semantic_tokens)
        weights = None

    for index, block in enumerate(model.blocks, start=1):
        if index > inject_after:
            x = block(x)

    x = model.final_norm(x)

    if return_cross_weights:
        return x, weights
    return x


def load_frozen_balanced_projector(
    checkpoint_path: str,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9.1-balanced-semantic-token-v0.8":
        raise ValueError("Expected Balanced Semantic Tokens v0.8 checkpoint.")

    config = checkpoint["config"]
    d_model = int(config["d_model"])

    projector = BalancedSemanticTokenProjector(
        d_model=d_model,
        base_norm=float(checkpoint.get("base_norm", math.sqrt(d_model))),
        inject_after=int(checkpoint.get("inject_after", 3)),
    ).to(device)
    projector.load_state_dict(checkpoint["projector_state_dict"])
    projector.eval()
    for p in projector.parameters():
        p.requires_grad_(False)

    return projector, checkpoint


def save_cross_attention_checkpoint(
    filename,
    generation_model,
    cross_attention,
    *,
    epoch,
    loss,
    base_model,
    semantic_adapter_checkpoint,
    balanced_checkpoint,
    inject_after,
    cross_learning_rate,
    block_learning_rate,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": "llm-gpu-v0.9.1-semantic-cross-attention-v0.9",
            "config": generation_model.config(),
            "model_state_dict": generation_model.state_dict(),
            "cross_attention_state_dict": cross_attention.state_dict(),
            "cross_num_heads": cross_attention.num_heads,
            "inject_after": int(inject_after),
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "semantic_adapter_checkpoint": semantic_adapter_checkpoint,
            "balanced_checkpoint": balanced_checkpoint,
            "cross_learning_rate": float(cross_learning_rate),
            "block_learning_rate": float(block_learning_rate),
        },
        path,
    )


def load_cross_attention_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9.1-semantic-cross-attention-v0.9":
        raise ValueError("Not a Semantic Cross-Attention v0.9 checkpoint.")

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

    cross_attention = SemanticCrossAttention(
        d_model=model.d_model,
        num_heads=int(checkpoint.get("cross_num_heads", model.num_heads)),
        initial_residual_scale=0.1,
    ).to(device)
    cross_attention.load_state_dict(
        checkpoint["cross_attention_state_dict"]
    )
    cross_attention.eval()

    return model, cross_attention, checkpoint
