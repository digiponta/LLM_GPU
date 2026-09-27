# cpu_name_binding_v0102.py
#
# LLM_GPU v0.9.1 v0.10.2
# CPU Name-Meaning Binding
#
# Cross-lingual lexical/semantic binding for:
#   CPU
#   Central Processing Unit
#   中央処理装置
#   中央演算処理装置

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


FORMAT = "llm-gpu-v0.9.1-cpu-name-binding-v0.10.2"


class CPUNameBindingProjection(nn.Module):
    def __init__(self, d_model=256, binding_dim=64):
        super().__init__()
        self.d_model = int(d_model)
        self.binding_dim = int(binding_dim)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, binding_dim)

    def forward(self, hidden):
        z = self.proj(self.norm(hidden))
        return F.normalize(z, dim=-1)


def save_cpu_name_binding_checkpoint(
    filename,
    projection,
    *,
    epoch,
    loss,
    base_model,
    semantic_adapter,
    learning_rate,
    margin,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "projection_state_dict": projection.state_dict(),
            "d_model": projection.d_model,
            "binding_dim": projection.binding_dim,
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "semantic_adapter": semantic_adapter,
            "learning_rate": float(learning_rate),
            "margin": float(margin),
        },
        path,
    )


def load_cpu_name_binding_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a CPU Name Binding v0.10.2 checkpoint.")

    projection = CPUNameBindingProjection(
        d_model=int(checkpoint["d_model"]),
        binding_dim=int(checkpoint["binding_dim"]),
    ).to(device)
    projection.load_state_dict(checkpoint["projection_state_dict"])
    projection.eval()
    return projection, checkpoint
