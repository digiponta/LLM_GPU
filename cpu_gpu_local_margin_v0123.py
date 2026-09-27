# cpu_gpu_local_margin_v0123.py
#
# LLM_GPU v0.9.1 v0.12.3
# CPU-GPU Local Margin Refinement
#
# Starts from v0.12.2 and locally refines only the direct logit adapter.
# The entity gate is frozen.

from __future__ import annotations

from pathlib import Path

import torch

from entity_contrastive_logit_alignment_v0122 import (
    FORMAT as V0122_FORMAT,
    load_checkpoint as load_v0122_checkpoint,
)


FORMAT = "llm-gpu-v0.9.1-cpu-gpu-local-margin-v0.12.3"


def save_checkpoint(
    filename,
    adapter,
    gate,
    *,
    epoch,
    loss,
    source_checkpoint,
    learning_rate,
    margin,
    preservation_weight,
    classification_weight,
    cpu_token_id,
    gpu_token_id,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "adapter_state_dict": adapter.state_dict(),
            "gate_state_dict": gate.state_dict(),
            "input_dim": adapter.input_dim,
            "vocab_size": adapter.vocab_size,
            "rank": adapter.rank,
            "beta": adapter.beta,
            "gate_hidden_dim": gate.hidden_dim,
            "epoch": int(epoch),
            "loss": float(loss),
            "source_checkpoint": source_checkpoint,
            "learning_rate": float(learning_rate),
            "margin": float(margin),
            "preservation_weight": float(preservation_weight),
            "classification_weight": float(classification_weight),
            "cpu_token_id": int(cpu_token_id),
            "gpu_token_id": int(gpu_token_id),
        },
        path,
    )


def load_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a CPU-GPU Local Margin v0.12.3 checkpoint.")

    source = checkpoint.get("source_checkpoint")
    if not source:
        raise ValueError("v0.12.3 checkpoint has no source_checkpoint.")

    adapter, gate, source_checkpoint = load_v0122_checkpoint(source, device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    gate.load_state_dict(checkpoint["gate_state_dict"])
    adapter.eval()
    gate.eval()
    return adapter, gate, checkpoint, source_checkpoint
