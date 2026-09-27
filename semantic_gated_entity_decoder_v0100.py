# semantic_gated_entity_decoder_v0100.py
#
# LLM_GPU v0.10.0
# Semantic-Gated Entity Decoder.
#
# The v0.9.9 language model and semantic head are frozen.
# A small learned decoder reads the adapted prompt hidden state:
#
#   prompt hidden (256)
#       -> LayerNorm -> Linear(256, 64) -> GELU
#          |-> gate: entity-identification vs normal LM
#          +-> entity classifier: CPU/GPU/LLM/Transformer/CUDA/Python
#
# At inference, only when gate >= threshold:
#   - force the FIRST token from the selected entity candidate
#   - then return immediately to ordinary LM decoding.
#
# This avoids making the entity token compete against all 8k vocabulary items
# at the first entity-identification step.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from entity_logit_alignment_v098 import ENTITY_BY_TAG
from dynamic_top_competitor_v099 import load_checkpoint as load_v099


FORMAT = "llm-gpu-v0.10.0-semantic-gated-entity-decoder"
ENTITY_TAGS = tuple(ENTITY_BY_TAG.keys())


class SemanticGatedEntityDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        entity_tags: Sequence[str] = ENTITY_TAGS,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.entity_tags = list(entity_tags)

        self.norm = nn.LayerNorm(self.d_model)
        self.shared = nn.Linear(self.d_model, self.hidden_dim)
        self.gate = nn.Linear(self.hidden_dim, 1)
        self.entity = nn.Linear(self.hidden_dim, len(self.entity_tags))

    def forward(self, prompt_hidden: torch.Tensor):
        h = F.gelu(self.shared(self.norm(prompt_hidden)))
        gate_logit = self.gate(h).squeeze(-1)
        entity_logits = self.entity(h)
        return gate_logit, entity_logits


def entity_token_ids(tokenizer):
    out = []
    for tag in ENTITY_TAGS:
        name = ENTITY_BY_TAG[tag]
        ids = tokenizer.encode(name)
        if not ids:
            raise RuntimeError(f"Tokenizer produced no token for {name}.")
        out.append(int(ids[0]))
    return out


def save_checkpoint(
    filename,
    decoder,
    *,
    epoch,
    loss,
    source_checkpoint,
    gate_threshold,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "d_model": decoder.d_model,
            "hidden_dim": decoder.hidden_dim,
            "entity_tags": list(decoder.entity_tags),
            "decoder_state_dict": decoder.state_dict(),
            "epoch": int(epoch),
            "loss": float(loss),
            "source_checkpoint": source_checkpoint,
            "gate_threshold": float(gate_threshold),
        },
        path,
    )


def load_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a v0.10.0 semantic-gated entity decoder.")

    source = checkpoint["source_checkpoint"]
    model, semantic_head, labels, source_ckpt = load_v099(source, device)

    decoder = SemanticGatedEntityDecoder(
        d_model=int(checkpoint["d_model"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 64)),
        entity_tags=checkpoint["entity_tags"],
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder.eval()

    return model, semantic_head, labels, decoder, checkpoint, source_ckpt
