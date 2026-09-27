# diagnose_semantic_neighborhood_v0122.py
#
# LLM_GPU v0.9.1
# v0.12.2 Semantic Neighborhood Diagnostic
#
# No training. Measures where G05 sits in the frozen 345-d semantic/lexical
# condition space relative to the CPU/GPU local rows used by failed v0.12.3.

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from entity_contrastive_logit_alignment_v0122 import load_checkpoint as load_v0122
from evaluate_entity_contrastive_logit_alignment_v0122 import (
    entity_diagnostics,
    prompt_state,
)
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
from semantic_lexical_logit_alignment_v012 import load_frozen_v011
from tokenizer_bpe import Tokenizer
from train_cpu_gpu_local_margin_v0123 import LOCAL_ROWS


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_NAME_BINDING = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
DEFAULT_V011 = "model/model-gpu-v0.9.1-lexical-generation-v011.pt"
DEFAULT_V0122 = "model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt"

ORIGINAL_G05 = "コンピュータの中心で多様な命令を処理する装置は何ですか。"
NAME_REQUEST_G05 = "コンピュータの中心で多様な命令を処理する装置の名前は何ですか。"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--name-binding", default=DEFAULT_NAME_BINDING)
    p.add_argument("--v011-checkpoint", default=DEFAULT_V011)
    p.add_argument("--v0122-checkpoint", default=DEFAULT_V0122)
    return p.parse_args()


@torch.no_grad()
def condition_for(
    text,
    tokenizer,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    generation_model,
    v011_projection,
):
    _prompt, condition, _semantic_bias, _base_logits = prompt_state(
        text,
        tokenizer,
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        name_binding,
        generation_model,
        v011_projection,
    )
    return condition[0]


def cosine(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def euclidean(a, b):
    return float(torch.linalg.vector_norm(a - b).item())


def summarize_group(name, rows):
    sims = [r["cosine"] for r in rows]
    dists = [r["euclidean"] for r in rows]
    margins = [r["margin"] for r in rows]
    print(
        f"{name:8s} n={len(rows):2d} "
        f"cos_mean={sum(sims)/len(sims):+.6f} "
        f"cos_max={max(sims):+.6f} "
        f"dist_mean={sum(dists)/len(dists):.6f} "
        f"dist_min={min(dists):.6f} "
        f"margin_mean={sum(margins)/len(margins):+.6f}"
    )


def main():
    args = parse_args()
    for filename in (
        args.tokenizer,
        args.base_model,
        args.semantic_adapter,
        args.name_binding,
        args.v011_checkpoint,
        args.v0122_checkpoint,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)

    (
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        base_checkpoint,
        semantic_checkpoint,
    ) = load_frozen_semantic_path_v08(
        args.base_model,
        args.semantic_adapter,
        device,
    )
    name_binding, name_checkpoint = load_cpu_name_binding_checkpoint(
        args.name_binding,
        device,
    )
    generation_model, v011_projection, v011_checkpoint = load_frozen_v011(
        args.v011_checkpoint,
        device,
    )
    adapter, gate, v0122_checkpoint = load_v0122(
        args.v0122_checkpoint,
        device,
    )

    local = []
    for i, (text, label) in enumerate(LOCAL_ROWS, start=1):
        condition = condition_for(
            text,
            tokenizer,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            generation_model,
            v011_projection,
        )
        gate_value, diag = entity_diagnostics(
            text,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
        )
        by_label = {r["label"]: r for r in diag}
        margin = (
            by_label["tech_cpu"]["final"] - by_label["tech_gpu"]["final"]
        )
        local.append({
            "index": i,
            "text": text,
            "label": label,
            "condition": condition,
            "gate": gate_value,
            "margin": margin,
        })

    print()
    print("====================================================")
    print(" v0.12.2 Semantic Neighborhood Diagnostic")
    print("====================================================")
    print("Device                 :", device)
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Semantic adapter loss  :", semantic_checkpoint.get("loss"))
    print("Name binding loss      :", name_checkpoint.get("loss"))
    print("v0.11 integration loss :", v011_checkpoint.get("loss"))
    print("v0.12.2 loss           :", v0122_checkpoint.get("loss"))
    print("Condition dim          :", v011_projection.input_dim)
    print("Local rows             :", len(local))
    print("Training               : none")
    print()

    for g05_name, g05_text in (
        ("Original G05", ORIGINAL_G05),
        ("Name-request G05", NAME_REQUEST_G05),
    ):
        g05 = condition_for(
            g05_text,
            tokenizer,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            generation_model,
            v011_projection,
        )

        rows = []
        for item in local:
            rows.append({
                **item,
                "cosine": cosine(g05, item["condition"]),
                "euclidean": euclidean(g05, item["condition"]),
            })

        rows_by_cos = sorted(rows, key=lambda x: x["cosine"], reverse=True)
        cpu_rows = [r for r in rows if r["label"] == "cpu"]
        gpu_rows = [r for r in rows if r["label"] == "gpu"]

        g05_gate, g05_diag = entity_diagnostics(
            g05_text,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
        )
        g05_by = {r["label"]: r for r in g05_diag}
        g05_margin = (
            g05_by["tech_cpu"]["final"] - g05_by["tech_gpu"]["final"]
        )

        cpu_centroid = torch.stack(
            [r["condition"] for r in cpu_rows], dim=0
        ).mean(dim=0)
        gpu_centroid = torch.stack(
            [r["condition"] for r in gpu_rows], dim=0
        ).mean(dim=0)

        print(g05_name)
        print("-" * len(g05_name))
        print("Prompt:", g05_text)
        print(f"Gate                   : {g05_gate:.6f}")
        print(f"CPU-GPU final margin   : {g05_margin:+.6f}")
        print(
            f"CPU centroid cosine    : {cosine(g05, cpu_centroid):+.6f}"
        )
        print(
            f"GPU centroid cosine    : {cosine(g05, gpu_centroid):+.6f}"
        )
        print(
            f"CPU centroid distance  : {euclidean(g05, cpu_centroid):.6f}"
        )
        print(
            f"GPU centroid distance  : {euclidean(g05, gpu_centroid):.6f}"
        )
        print()

        print("Group summary")
        summarize_group("CPU", cpu_rows)
        summarize_group("GPU", gpu_rows)
        print()

        print("Nearest local rows by cosine")
        print("----------------------------")
        for rank, row in enumerate(rows_by_cos, start=1):
            print(
                f"{rank:02d}. {row['label'].upper():3s} "
                f"cos={row['cosine']:+.6f} "
                f"dist={row['euclidean']:.6f} "
                f"gate={row['gate']:.4f} "
                f"CPU-GPU-margin={row['margin']:+.6f}"
            )
            print("    " + row["text"])
        print()

        nearest_cpu = max(cpu_rows, key=lambda x: x["cosine"])
        nearest_gpu = max(gpu_rows, key=lambda x: x["cosine"])
        print("Nearest CPU vs GPU")
        print("------------------")
        print(
            f"CPU nearest: cos={nearest_cpu['cosine']:+.6f} "
            f"dist={nearest_cpu['euclidean']:.6f} "
            f"margin={nearest_cpu['margin']:+.6f}"
        )
        print("  " + nearest_cpu["text"])
        print(
            f"GPU nearest: cos={nearest_gpu['cosine']:+.6f} "
            f"dist={nearest_gpu['euclidean']:.6f} "
            f"margin={nearest_gpu['margin']:+.6f}"
        )
        print("  " + nearest_gpu["text"])
        print()


if __name__ == "__main__":
    main()
