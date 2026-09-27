# diagnose_local_logit_surface_v0122.py
#
# LLM_GPU v0.9.1
# v0.12.2 Local Logit Surface / Jacobian Diagnostic
#
# No training.
#
# Let z be the frozen 345-d semantic+lexical condition and define
#
#   D(z) = final_logit(CPU) - final_logit(GPU)
#
# with the prompt-specific frozen base logits held fixed:
#
#   final_logits(z) = base_logits(prompt) + gate(z) * adapter(z)
#
# This diagnostic measures grad_z D(z), its directional derivatives toward
# CPU/GPU semantic neighbors, and finite-step changes along those directions.

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from entity_contrastive_logit_alignment_v0122 import load_checkpoint as load_v0122
from evaluate_entity_contrastive_logit_alignment_v0122 import prompt_state
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

STEP_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)


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
def state_for(
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
    _prompt, condition, _semantic_bias, base_logits = prompt_state(
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
    return condition[0].detach(), base_logits.detach()


def cosine(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def normalized_direction(source, target):
    delta = target - source
    distance = torch.linalg.vector_norm(delta)
    if float(distance.item()) <= 1e-12:
        raise RuntimeError("Zero-length diagnostic direction.")
    return delta / distance, float(distance.item())


def final_margin(z, base_logits, adapter, gate, cpu_id, gpu_id):
    if z.ndim == 1:
        z = z.unsqueeze(0)
    g = gate(z)
    bias = adapter(z)
    effective = g.unsqueeze(-1) * bias
    final = base_logits.unsqueeze(0) + effective
    return final[:, cpu_id] - final[:, gpu_id], g


def local_gradient(z0, base_logits, adapter, gate, cpu_id, gpu_id):
    z = z0.detach().clone().requires_grad_(True)
    margin, g = final_margin(z, base_logits, adapter, gate, cpu_id, gpu_id)
    margin[0].backward()
    grad = z.grad.detach().clone()
    return grad, float(margin[0].detach().item()), float(g[0].detach().item())


def print_direction(
    name,
    z0,
    target,
    grad,
    base_margin,
    base_logits,
    adapter,
    gate,
    cpu_id,
    gpu_id,
):
    direction, distance = normalized_direction(z0, target)
    directional = float(torch.dot(grad, direction).item())
    grad_norm = float(torch.linalg.vector_norm(grad).item())
    cosine_grad = directional / grad_norm if grad_norm > 0 else 0.0

    print(name)
    print(f"  target distance           : {distance:.6f}")
    print(f"  grad dot unit(direction)  : {directional:+.6f}")
    print(f"  cosine(grad, direction)   : {cosine_grad:+.6f}")
    print(
        "  local interpretation     : "
        + ("CPU-margin increasing" if directional > 0 else "CPU-margin decreasing")
    )
    print("  finite steps:")
    with torch.no_grad():
        for fraction in STEP_FRACTIONS:
            z = z0 + fraction * (target - z0)
            margin, gate_prob = final_margin(
                z, base_logits, adapter, gate, cpu_id, gpu_id
            )
            value = float(margin[0].item())
            actual_delta = value - base_margin
            first_order = directional * (fraction * distance)
            print(
                f"    t={fraction:>4.2f} "
                f"D={value:+.6f} "
                f"delta={actual_delta:+.6f} "
                f"linear={first_order:+.6f} "
                f"gate={float(gate_prob[0].item()):.6f}"
            )
    print()


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
    adapter, gate, checkpoint = load_v0122(
        args.v0122_checkpoint,
        device,
    )

    # Parameters stay fixed; gradients are needed only with respect to z.
    adapter.eval()
    gate.eval()
    for module in (adapter, gate):
        for p in module.parameters():
            p.requires_grad_(False)

    cpu_ids = tokenizer.encode("CPU")
    gpu_ids = tokenizer.encode("GPU")
    if not cpu_ids or not gpu_ids:
        raise RuntimeError("CPU/GPU tokenization failed.")
    cpu_id = int(cpu_ids[0])
    gpu_id = int(gpu_ids[0])

    local = []
    for text, label in LOCAL_ROWS:
        z, _base = state_for(
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
        local.append({"text": text, "label": label, "z": z})

    cpu_local = [r for r in local if r["label"] == "cpu"]
    gpu_local = [r for r in local if r["label"] == "gpu"]
    cpu_centroid = torch.stack([r["z"] for r in cpu_local]).mean(dim=0)
    gpu_centroid = torch.stack([r["z"] for r in gpu_local]).mean(dim=0)

    print()
    print("====================================================")
    print(" v0.12.2 Local Logit Surface / Jacobian Diagnostic")
    print("====================================================")
    print("Device                 :", device)
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Semantic adapter loss  :", semantic_checkpoint.get("loss"))
    print("Name binding loss      :", name_checkpoint.get("loss"))
    print("v0.11 integration loss :", v011_checkpoint.get("loss"))
    print("v0.12.2 loss           :", checkpoint.get("loss"))
    print("Condition dim          :", v011_projection.input_dim)
    print("D(z)                   : final_logit(CPU) - final_logit(GPU)")
    print("Base LM logits         : fixed per G05 prompt")
    print("Differentiated path    : gate(z) * direct_adapter(z)")
    print("Training               : none")
    print()

    for title, text in (
        ("Original G05", ORIGINAL_G05),
        ("Name-request G05", NAME_REQUEST_G05),
    ):
        z0, base_logits = state_for(
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

        grad, base_margin, gate_value = local_gradient(
            z0, base_logits, adapter, gate, cpu_id, gpu_id
        )
        grad_norm = float(torch.linalg.vector_norm(grad).item())

        nearest_cpu = max(
            cpu_local,
            key=lambda r: cosine(z0, r["z"]),
        )
        nearest_gpu = max(
            gpu_local,
            key=lambda r: cosine(z0, r["z"]),
        )

        print(title)
        print("-" * len(title))
        print("Prompt:", text)
        print(f"D(z0)                  : {base_margin:+.6f}")
        print(f"Gate(z0)               : {gate_value:.6f}")
        print(f"||grad D(z0)||         : {grad_norm:.6f}")
        print(
            f"Nearest CPU cosine     : {cosine(z0, nearest_cpu['z']):+.6f}"
        )
        print(
            f"Nearest GPU cosine     : {cosine(z0, nearest_gpu['z']):+.6f}"
        )
        print()

        print_direction(
            "Direction: nearest CPU",
            z0,
            nearest_cpu["z"],
            grad,
            base_margin,
            base_logits,
            adapter,
            gate,
            cpu_id,
            gpu_id,
        )
        print("  reference:", nearest_cpu["text"])
        print()

        print_direction(
            "Direction: CPU centroid",
            z0,
            cpu_centroid,
            grad,
            base_margin,
            base_logits,
            adapter,
            gate,
            cpu_id,
            gpu_id,
        )

        print_direction(
            "Direction: nearest GPU",
            z0,
            nearest_gpu["z"],
            grad,
            base_margin,
            base_logits,
            adapter,
            gate,
            cpu_id,
            gpu_id,
        )
        print("  reference:", nearest_gpu["text"])
        print()

        print_direction(
            "Direction: GPU centroid",
            z0,
            gpu_centroid,
            grad,
            base_margin,
            base_logits,
            adapter,
            gate,
            cpu_id,
            gpu_id,
        )

        cpu_dir, _ = normalized_direction(z0, cpu_centroid)
        gpu_dir, _ = normalized_direction(z0, gpu_centroid)
        cpu_dd = float(torch.dot(grad, cpu_dir).item())
        gpu_dd = float(torch.dot(grad, gpu_dir).item())
        print("Directional summary")
        print("-------------------")
        print(f"toward CPU centroid : {cpu_dd:+.6f}")
        print(f"toward GPU centroid : {gpu_dd:+.6f}")
        print(f"CPU minus GPU dir.  : {cpu_dd-gpu_dd:+.6f}")
        if cpu_dd > 0 and gpu_dd < 0:
            verdict = "locally aligned: CPU direction raises D and GPU direction lowers D"
        elif cpu_dd <= 0:
            verdict = "locally inconsistent: moving toward CPU centroid does not raise D"
        else:
            verdict = "mixed local geometry: CPU direction raises D but GPU direction is not opposite"
        print("Interpretation        :", verdict)
        print()


if __name__ == "__main__":
    main()
