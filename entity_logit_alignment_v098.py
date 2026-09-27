# entity_logit_alignment_v098.py
#
# LLM_GPU v0.9.8
# Semantic state -> entity token/logit causal alignment.
#
# Continues from v0.9.7.  The adapted semantic representation is preserved,
# while explicit margin losses connect single technical entity states to the
# first token of CPU/GPU/LLM/Transformer/CUDA/Python.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F

from output_alignment_v097 import load_checkpoint as load_v097
from semantic_encoder_adaptation_v096 import SemanticTagHead


FORMAT = "llm-gpu-v0.9.8-entity-logit-alignment"

ENTITY_BY_TAG = {
    "tech_cpu": "CPU",
    "tech_gpu": "GPU",
    "tech_llm": "LLM",
    "tech_transformer": "Transformer",
    "tech_cuda": "CUDA",
    "tech_python": "Python",
}


def entity_token_ids(tokenizer) -> Dict[str, int]:
    out = {}
    for tag, name in ENTITY_BY_TAG.items():
        ids = tokenizer.encode(name)
        if not ids:
            raise RuntimeError(f"Tokenizer produced no token for {name}.")
        out[tag] = int(ids[0])
    return out


def single_entity_tag(tags: Iterable[str]):
    found = [tag for tag in ENTITY_BY_TAG if tag in set(tags)]
    return found[0] if len(found) == 1 else None


def entity_margin_loss(
    prompt_logits: torch.Tensor,
    entity_target: torch.Tensor,
    entity_ids: Sequence[int],
    margin: float,
):
    """Target entity logit must exceed every competing entity logit."""
    losses = []
    device = prompt_logits.device
    entity_ids_t = torch.tensor(entity_ids, dtype=torch.long, device=device)

    for row in range(prompt_logits.size(0)):
        target = int(entity_target[row].item())
        if target < 0:
            continue
        target_logit = prompt_logits[row, target]
        competitors = entity_ids_t[entity_ids_t != target]
        if competitors.numel() == 0:
            continue
        competitor_max = prompt_logits[row, competitors].max()
        losses.append(F.relu(margin - (target_logit - competitor_max)))

    if not losses:
        return prompt_logits.sum() * 0.0
    return torch.stack(losses).mean()


def blocker_margin_loss(
    prompt_logits: torch.Tensor,
    entity_target: torch.Tensor,
    answer_starts_entity: torch.Tensor,
    margin: float,
):
    """For rows whose gold answer starts with the entity, make it global top-1."""
    losses = []
    for row in range(prompt_logits.size(0)):
        target = int(entity_target[row].item())
        if target < 0 or not bool(answer_starts_entity[row].item()):
            continue

        target_logit = prompt_logits[row, target]
        masked = prompt_logits[row].clone()
        masked[target] = float("-inf")
        blocker = masked.max()
        losses.append(F.relu(margin - (target_logit - blocker)))

    if not losses:
        return prompt_logits.sum() * 0.0
    return torch.stack(losses).mean()


def save_checkpoint(
    filename,
    model,
    semantic_head: SemanticTagHead,
    labels,
    *,
    epoch,
    loss,
    source_checkpoint,
    entity_margin,
    blocker_margin,
    entity_weight,
    blocker_weight,
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
            "entity_margin": float(entity_margin),
            "blocker_margin": float(blocker_margin),
            "entity_weight": float(entity_weight),
            "blocker_weight": float(blocker_weight),
        },
        path,
    )


def load_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a v0.9.8 entity-logit checkpoint.")

    # Reuse v0.9.7 loader machinery by presenting an in-memory-equivalent
    # structure directly here.
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
