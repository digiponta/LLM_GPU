# semantic_generation_gate_v091.py
#
# LLM_GPU v0.9.1
# Semantic-to-Generation Interface v0.6: Semantic-Gated Generation
#
# Frozen semantic path (v0.5):
#   adapted hidden (256)
#   concept probs (6)
#   attribute probs (4)
#   hierarchy probs (5)
#
# Explicit semantic gates:
#   CPU, GPU, Transformer, CUDA, Python, LLM
#
# Injection:
#   semantic residual = Linear(256 -> 256)
#   gate residual     = Linear(6 -> 256)
#   bias = alpha * (semantic residual + gate residual)
#
# Both projections are zero-initialized so integration starts as a no-op.

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model import LanguageModel
from semantic_encoder_adapter_v05 import (
    ATTRIBUTE_LABELS,
    CONCEPT_LABELS,
    HIERARCHY_LABELS,
)
from semantic_generation_integration_v09 import (
    configure_generation_model,
    forward_semantic_conditioned,
    load_frozen_semantic_path,
)


GATE_LABELS = [
    "cpu_general",
    "gpu_parallel",
    "transformer_structure",
    "cuda_platform",
    "python_language",
    "llm_language",
]


def _index(labels, name):
    return list(labels).index(name)


def semantic_gate_values(
    concept_prob: torch.Tensor,
    attribute_prob: torch.Tensor,
    hierarchy_prob: torch.Tensor,
) -> torch.Tensor:
    cpu_prob = concept_prob[:, _index(CONCEPT_LABELS, "tech_cpu")]
    gpu_prob = concept_prob[:, _index(CONCEPT_LABELS, "tech_gpu")]
    transformer_prob = concept_prob[:, _index(CONCEPT_LABELS, "tech_transformer")]
    cuda_prob = concept_prob[:, _index(CONCEPT_LABELS, "tech_cuda")]
    python_prob = concept_prob[:, _index(CONCEPT_LABELS, "tech_python")]
    llm_prob = concept_prob[:, _index(CONCEPT_LABELS, "tech_llm")]

    general = hierarchy_prob[:, _index(HIERARCHY_LABELS, "general_purpose")]
    control = hierarchy_prob[:, _index(HIERARCHY_LABELS, "control_oriented")]
    throughput = hierarchy_prob[:, _index(HIERARCHY_LABELS, "throughput_oriented")]
    parallel_h = hierarchy_prob[:, _index(HIERARCHY_LABELS, "data_parallel")]

    attr_parallel = attribute_prob[:, _index(ATTRIBUTE_LABELS, "property_parallel")]
    attr_language = attribute_prob[:, _index(ATTRIBUTE_LABELS, "property_language")]
    attr_structure = attribute_prob[
        :,
        _index(ATTRIBUTE_LABELS, "property_attention_structure"),
    ]

    cpu_style = 0.5 * (general + control)
    gpu_style = (
        throughput + parallel_h + attr_parallel
    ) / 3.0

    cpu_gate = cpu_prob * cpu_style
    gpu_gate = gpu_prob * gpu_style
    transformer_gate = transformer_prob * attr_structure
    cuda_gate = cuda_prob
    python_gate = python_prob * torch.maximum(
        attr_language,
        python_prob,
    )
    llm_gate = llm_prob * torch.maximum(
        attr_language,
        llm_prob,
    )

    return torch.stack(
        [
            cpu_gate,
            gpu_gate,
            transformer_gate,
            cuda_gate,
            python_gate,
            llm_gate,
        ],
        dim=-1,
    )


class SemanticGatedProjection(nn.Module):
    def __init__(
        self,
        semantic_dim: int = 256,
        gate_dim: int = len(GATE_LABELS),
        d_model: int = 256,
        alpha: float = 0.1,
        gate_alpha: float = 1.0,
        inject_after: int = 3,
    ):
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.gate_dim = int(gate_dim)
        self.d_model = int(d_model)
        self.alpha = float(alpha)
        self.gate_alpha = float(gate_alpha)
        self.inject_after = int(inject_after)

        self.semantic_projection = nn.Linear(
            self.semantic_dim,
            self.d_model,
            bias=False,
        )
        self.gate_projection = nn.Linear(
            self.gate_dim,
            self.d_model,
            bias=False,
        )

        nn.init.zeros_(self.semantic_projection.weight)
        nn.init.zeros_(self.gate_projection.weight)

    def forward(
        self,
        adapted_hidden: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        semantic_residual = self.semantic_projection(adapted_hidden)
        gate_residual = self.gate_projection(gates)
        return self.alpha * (
            semantic_residual
            + self.gate_alpha * gate_residual
        )


def save_gated_checkpoint(
    filename,
    generation_model,
    projection,
    *,
    epoch,
    loss,
    base_model,
    semantic_adapter_checkpoint,
    block_learning_rate,
    projection_learning_rate,
):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": "llm-gpu-v0.9.1-semantic-gated-generation-v0.6",
            "config": generation_model.config(),
            "model_state_dict": generation_model.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "semantic_dim": projection.semantic_dim,
            "gate_dim": projection.gate_dim,
            "gate_labels": list(GATE_LABELS),
            "alpha": projection.alpha,
            "gate_alpha": projection.gate_alpha,
            "inject_after": projection.inject_after,
            "epoch": int(epoch),
            "loss": float(loss),
            "base_model": base_model,
            "semantic_adapter_checkpoint": semantic_adapter_checkpoint,
            "block_learning_rate": float(block_learning_rate),
            "projection_learning_rate": float(projection_learning_rate),
        },
        path,
    )


def load_gated_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    if checkpoint.get("format") != "llm-gpu-v0.9.1-semantic-gated-generation-v0.6":
        raise ValueError("Not a semantic-gated generation v0.6 checkpoint.")

    if list(checkpoint.get("gate_labels", [])) != list(GATE_LABELS):
        raise ValueError("Semantic gate label order mismatch.")

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
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    projection = SemanticGatedProjection(
        semantic_dim=int(checkpoint["semantic_dim"]),
        gate_dim=int(checkpoint["gate_dim"]),
        d_model=model.d_model,
        alpha=float(checkpoint.get("alpha", 0.1)),
        gate_alpha=float(checkpoint.get("gate_alpha", 1.0)),
        inject_after=int(checkpoint.get("inject_after", 3)),
    ).to(device)
    projection.load_state_dict(checkpoint["projection_state_dict"])
    projection.eval()

    return model, projection, checkpoint
