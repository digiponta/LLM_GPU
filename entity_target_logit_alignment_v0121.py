# entity_target_logit_alignment_v0121.py
#
# LLM_GPU v0.9.1 v0.12.1
# Entity-Target Logit Alignment
#
# Explicit semantic concept -> entity-name supervision:
# tech_cpu -> CPU, tech_gpu -> GPU, tech_llm -> LLM,
# tech_transformer -> Transformer, tech_cuda -> CUDA, tech_python -> Python.

from __future__ import annotations

from pathlib import Path
import torch

from semantic_lexical_logit_alignment_v012 import SemanticLexicalLogitAdapter


FORMAT = "llm-gpu-v0.9.1-entity-target-logit-alignment-v0.12.1"

ENTITY_TARGETS = {
    "tech_gpu": "GPU",
    "tech_cpu": "CPU",
    "tech_llm": "LLM",
    "tech_transformer": "Transformer",
    "tech_cuda": "CUDA",
    "tech_python": "Python",
}


def save_checkpoint(
    filename,
    adapter,
    *,
    epoch,
    loss,
    v011_checkpoint,
    semantic_adapter_checkpoint,
    name_binding_checkpoint,
    learning_rate,
    l2_weight,
    target_token_ids,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "adapter_state_dict": adapter.state_dict(),
            "input_dim": adapter.input_dim,
            "vocab_size": adapter.vocab_size,
            "rank": adapter.rank,
            "beta": adapter.beta,
            "epoch": int(epoch),
            "loss": float(loss),
            "v011_checkpoint": v011_checkpoint,
            "semantic_adapter_checkpoint": semantic_adapter_checkpoint,
            "name_binding_checkpoint": name_binding_checkpoint,
            "learning_rate": float(learning_rate),
            "l2_weight": float(l2_weight),
            "entity_targets": dict(ENTITY_TARGETS),
            "target_token_ids": dict(target_token_ids),
        },
        path,
    )


def load_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not an Entity-Target Logit Alignment v0.12.1 checkpoint.")

    adapter = SemanticLexicalLogitAdapter(
        input_dim=int(checkpoint["input_dim"]),
        vocab_size=int(checkpoint["vocab_size"]),
        rank=int(checkpoint.get("rank", 64)),
        beta=float(checkpoint.get("beta", 0.10)),
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    return adapter, checkpoint
