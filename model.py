# model.py
#
# CUDA/PyTorch Transformer language model.
#
# v0.6 additions:
#   - configurable multi-head causal self-attention
#   - optional learned positional embeddings
#   - backward-compatible loading of older single-head checkpoints

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        causal: bool = True,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        x = x.view(batch, time, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(float(self.head_dim))

        if self.causal:
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
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, time, self.d_model)
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
        num_heads: int = 1,
        causal: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            causal=causal,
        )
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
        num_heads: int = 1,
        causal: bool = True,
        context_length: int = 64,
        use_position_embedding: bool = False,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if d_model <= 0 or num_layers <= 0 or hidden_dim <= 0:
            raise ValueError("Model dimensions must be > 0.")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if context_length <= 0:
            raise ValueError("context_length must be > 0.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.causal = causal
        self.context_length = context_length
        self.use_position_embedding = use_position_embedding

        self.embedding = nn.Embedding(vocab_size, d_model)
        if use_position_embedding:
            self.position_embedding = nn.Embedding(context_length, d_model)
        else:
            self.position_embedding = None

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
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
            "num_heads": self.num_heads,
            "causal": self.causal,
            "context_length": self.context_length,
            "use_position_embedding": self.use_position_embedding,
        }

    def forward_hidden(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return final normalized hidden states before the LM head."""
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape [batch, time].")

        batch, time = token_ids.shape
        if time > self.context_length:
            raise ValueError(
                f"Sequence length {time} exceeds context length "
                f"{self.context_length}."
            )

        x = self.embedding(token_ids)

        if self.position_embedding is not None:
            positions = torch.arange(time, device=token_ids.device)
            x = x + self.position_embedding(positions).unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_hidden(token_ids)
        return self.lm_head(hidden)

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
            "format": "homemade-llm-gpu-v0.6",
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
            num_heads=int(config.get("num_heads", 1)),
            causal=bool(config.get("causal", True)),
            context_length=int(config.get("context_length", 64)),
            use_position_embedding=bool(
                config.get("use_position_embedding", False)
            ),
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
