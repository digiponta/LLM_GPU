# model.py
#
# CUDA/PyTorch implementation of the homemade v0.3 Transformer LM.

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Single-head causal self-attention, matching the original v0.3 design."""

    def __init__(self, d_model: int, causal: bool = True):
        super().__init__()
        self.d_model = d_model
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(float(self.d_model))

        if self.causal:
            time = x.size(1)
            mask = torch.triu(
                torch.ones(
                    (time, time),
                    dtype=torch.bool,
                    device=x.device,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        causal: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, causal=causal)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        hidden_dim: int = 256,
        causal: bool = True,
        context_length: int = 64,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if d_model <= 0 or num_layers <= 0 or hidden_dim <= 0:
            raise ValueError("Model dimensions must be > 0.")
        if context_length <= 0:
            raise ValueError("context_length must be > 0.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.context_length = context_length

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                hidden_dim=hidden_dim,
                causal=causal,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def config(self) -> Dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "causal": self.causal,
            "context_length": self.context_length,
        }

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch, time]
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape [batch, time].")

        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def save_checkpoint(
        self,
        filename: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
    ) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "format": "homemade-llm-gpu-v0.3",
            "config": self.config(),
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        filename: str,
        device: torch.device,
    ):
        checkpoint = torch.load(filename, map_location=device)
        config = checkpoint["config"]

        model = cls(
            vocab_size=int(config["vocab_size"]),
            d_model=int(config["d_model"]),
            num_layers=int(config["num_layers"]),
            hidden_dim=int(config["hidden_dim"]),
            causal=bool(config.get("causal", True)),
            context_length=int(config.get("context_length", 64)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, checkpoint

    @torch.no_grad()
    def generate(
        self,
        token_ids: List[int],
        max_new_tokens: int = 100,
        eos_id: Optional[int] = None,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        repetition_penalty: float = 1.15,
    ) -> List[int]:
        if not token_ids:
            raise ValueError("token_ids must not be empty.")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0.")

        self.eval()
        device = next(self.parameters()).device
        generated = list(token_ids)

        for _ in range(max_new_tokens):
            context = generated[-self.context_length:]
            x = torch.tensor(
                [context],
                dtype=torch.long,
                device=device,
            )

            logits = self(x)[0, -1, :].clone()

            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if 0 <= token_id < logits.numel():
                        if logits[token_id] >= 0:
                            logits[token_id] /= repetition_penalty
                        else:
                            logits[token_id] *= repetition_penalty

            if temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                logits = logits / temperature

                if top_k is not None and 0 < top_k < logits.numel():
                    top_values, top_indices = torch.topk(logits, top_k)
                    probabilities = F.softmax(top_values, dim=-1)
                    selected = torch.multinomial(probabilities, num_samples=1)
                    next_id = int(top_indices[selected].item())
                else:
                    probabilities = F.softmax(logits, dim=-1)
                    next_id = int(
                        torch.multinomial(probabilities, num_samples=1).item()
                    )

            generated.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

        return generated
