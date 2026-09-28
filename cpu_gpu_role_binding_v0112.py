# cpu_gpu_role_binding_v0112.py
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn

ROLE_LABELS = ("controller", "executor")

class RoleHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(ROLE_LABELS)),
        )
    def forward(self, x):
        return self.net(x)

class RoleEntityBinding(nn.Module):
    def __init__(
        self,
        intent_dim: int,
        role_dim: int,
        vocab_size: int,
        rank: int = 16,
        beta: float = 1.0,
    ):
        super().__init__()
        self.intent_dim = int(intent_dim)
        self.role_dim = int(role_dim)
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.beta = float(beta)

        self.down = nn.Linear(self.intent_dim + self.role_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.vocab_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, intent_prob: torch.Tensor, role_prob: torch.Tensor):
        x = torch.cat([intent_prob, role_prob], dim=-1)
        return self.beta * self.up(torch.tanh(self.down(x)))


def save_checkpoint(
    filename: str,
    role_head: RoleHead,
    binding: RoleEntityBinding,
    intent_labels: Sequence[str],
    target_token_ids: dict,
    **meta,
):
    torch.save(
        {
            "role_labels": list(ROLE_LABELS),
            "role_head_state": role_head.state_dict(),
            "binding_state": binding.state_dict(),
            "intent_labels": list(intent_labels),
            "target_token_ids": dict(target_token_ids),
            "d_model": role_head.net[0].in_features,
            "rank": binding.rank,
            "beta": binding.beta,
            "vocab_size": binding.vocab_size,
            **meta,
        },
        filename,
    )


def load_checkpoint(filename: str, intent_labels: Sequence[str], device):
    ck = torch.load(filename, map_location=device)
    if list(ck["intent_labels"]) != list(intent_labels):
        raise ValueError("Intent label order mismatch.")

    role_head = RoleHead(int(ck["d_model"])).to(device)
    role_head.load_state_dict(ck["role_head_state"])
    role_head.eval()

    up_w = ck["binding_state"]["up.weight"]
    down_w = ck["binding_state"]["down.weight"]
    binding = RoleEntityBinding(
        len(intent_labels),
        len(ROLE_LABELS),
        int(ck.get("vocab_size", up_w.shape[0])),
        int(ck.get("rank", down_w.shape[0])),
        float(ck.get("beta", 1.0)),
    ).to(device)
    binding.load_state_dict(ck["binding_state"])
    binding.eval()
    return role_head, binding, ck
