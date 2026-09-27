# semantic_lexical_logit_alignment_v012.py
#
# LLM_GPU v0.9.1 v0.12
# Semantic / Lexical -> Logit Alignment
#
# Directly maps the frozen 345-d semantic+lexical condition to a vocabulary
# logit bias. The bias is applied only to the first assistant token.

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from lexical_generation_alignment_v011 import load_checkpoint as load_v011_checkpoint


FORMAT = "llm-gpu-v0.9.1-semantic-lexical-logit-alignment-v0.12"


class SemanticLexicalLogitAdapter(nn.Module):
    def __init__(self, input_dim, vocab_size, rank=64, beta=0.10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.beta = float(beta)

        self.norm = nn.LayerNorm(self.input_dim)
        self.down = nn.Linear(self.input_dim, self.rank)
        self.act = nn.GELU()
        self.up = nn.Linear(self.rank, self.vocab_size, bias=False)

        # Start from exactly the frozen v0.11 behavior.
        nn.init.zeros_(self.up.weight)

    def forward(self, condition):
        return self.beta * self.up(self.act(self.down(self.norm(condition))))


def save_checkpoint(
    filename,
    logit_adapter,
    *,
    epoch,
    loss,
    v011_checkpoint,
    semantic_adapter_checkpoint,
    name_binding_checkpoint,
    learning_rate,
    l2_weight,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "logit_adapter_state_dict": logit_adapter.state_dict(),
            "input_dim": logit_adapter.input_dim,
            "vocab_size": logit_adapter.vocab_size,
            "rank": logit_adapter.rank,
            "beta": logit_adapter.beta,
            "epoch": int(epoch),
            "loss": float(loss),
            "v011_checkpoint": v011_checkpoint,
            "semantic_adapter_checkpoint": semantic_adapter_checkpoint,
            "name_binding_checkpoint": name_binding_checkpoint,
            "learning_rate": float(learning_rate),
            "l2_weight": float(l2_weight),
        },
        path,
    )


def load_logit_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a Semantic/Lexical Logit Alignment v0.12 checkpoint.")

    adapter = SemanticLexicalLogitAdapter(
        input_dim=int(checkpoint["input_dim"]),
        vocab_size=int(checkpoint["vocab_size"]),
        rank=int(checkpoint.get("rank", 64)),
        beta=float(checkpoint.get("beta", 0.10)),
    ).to(device)
    adapter.load_state_dict(checkpoint["logit_adapter_state_dict"])
    adapter.eval()
    return adapter, checkpoint


def load_frozen_v011(filename, device):
    model, projection, checkpoint = load_v011_checkpoint(filename, device)
    for module in (model, projection):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    return model, projection, checkpoint
