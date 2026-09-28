# intent_entity_logit_binding_v0110.py
#
# v0.11.0 Clean Intent -> Entity Logit Binding
#
# A small low-rank adapter maps the frozen 24-dimensional clean intent
# probabilities directly to a vocabulary-logit bias.  The bias is applied
# only at the first generated token and only when a technical intent is
# sufficiently confident.

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


TECHNICAL_LABELS = (
    "tech_gpu",
    "tech_cpu",
    "tech_llm",
    "tech_transformer",
    "tech_cuda",
    "tech_python",
)

ENTITY_TARGETS = {
    "tech_gpu": "GPU",
    "tech_cpu": "CPU",
    "tech_llm": "LLM",
    "tech_transformer": "Transformer",
    "tech_cuda": "CUDA",
    "tech_python": "Python",
}


class IntentEntityLogitBinding(nn.Module):
    def __init__(
        self,
        num_labels: int,
        vocab_size: int,
        rank: int = 32,
        beta: float = 1.0,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.beta = float(beta)

        self.down = nn.Linear(self.num_labels, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.vocab_size, bias=False)

        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, intent_prob: torch.Tensor) -> torch.Tensor:
        return self.beta * self.up(torch.tanh(self.down(intent_prob)))


def technical_confidence(
    intent_prob: torch.Tensor,
    labels: Sequence[str],
) -> torch.Tensor:
    indices = [
        labels.index(label)
        for label in TECHNICAL_LABELS
        if label in labels
    ]
    if not indices:
        return torch.zeros(
            intent_prob.size(0),
            device=intent_prob.device,
            dtype=intent_prob.dtype,
        )
    return intent_prob[:, indices].max(dim=-1).values


def save_binding_checkpoint(
    filename: str,
    adapter: IntentEntityLogitBinding,
    labels: Sequence[str],
    target_token_ids: dict,
    *,
    epoch: int,
    loss: float,
    base_model: str,
    intent_head: str,
    learning_rate: float,
    gate_threshold: float,
):
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "labels": list(labels),
            "target_token_ids": dict(target_token_ids),
            "num_labels": adapter.num_labels,
            "vocab_size": adapter.vocab_size,
            "rank": adapter.rank,
            "beta": adapter.beta,
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "intent_head": intent_head,
            "learning_rate": float(learning_rate),
            "gate_threshold": float(gate_threshold),
        },
        filename,
    )


def load_binding_checkpoint(
    filename: str,
    labels: Sequence[str],
    device: torch.device,
):
    checkpoint = torch.load(filename, map_location=device)
    stored = list(checkpoint["labels"])
    if stored != list(labels):
        raise ValueError("Intent label order mismatch in binding checkpoint.")

    adapter = IntentEntityLogitBinding(
        num_labels=int(checkpoint["num_labels"]),
        vocab_size=int(checkpoint["vocab_size"]),
        rank=int(checkpoint.get("rank", 32)),
        beta=float(checkpoint.get("beta", 1.0)),
    ).to(device)
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()
    return adapter, checkpoint
