# diagnose_g05_topk_logits_v0122.py
#
# LLM_GPU v0.9.1
# v0.12.2 G05 Top-K Logit Diagnostic
#
# No training.
#
# For both G05 variants and selected direct-bias gains, print the actual top
# vocabulary competitors with:
#   token string
#   token id
#   base logit
#   raw direct bias
#   gamma-scaled effective bias
#   final logit
#   final rank
#
# This identifies which token(s) still outrank CPU after CPU > GPU is achieved.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from entity_contrastive_logit_alignment_v0122 import load_checkpoint as load_v0122
from evaluate_direct_bias_gain_sweep_v0122 import gain_state
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
from semantic_lexical_logit_alignment_v012 import load_frozen_v011
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_NAME_BINDING = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
DEFAULT_V011 = "model/model-gpu-v0.9.1-lexical-generation-v011.pt"
DEFAULT_V0122 = "model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt"

GAINS = (1.00, 1.50, 2.00, 2.50, 3.00)
TOP_K = 10

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
    p.add_argument("--top-k", type=int, default=TOP_K)
    return p.parse_args()


def safe_piece(tokenizer, token_id):
    piece = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    if piece == "":
        return "<SPECIAL/EMPTY>"
    return piece.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def rank_of(logits, token_id):
    return int((logits > logits[token_id]).sum().item()) + 1


@torch.no_grad()
def diagnose_one(
    title,
    text,
    gamma,
    top_k,
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
):
    (
        _prompt,
        _semantic_bias,
        base_logits,
        combined,
        gate_prob,
        raw_bias,
        effective,
    ) = gain_state(
        text,
        gamma,
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

    cpu_ids = tokenizer.encode("CPU")
    gpu_ids = tokenizer.encode("GPU")
    if not cpu_ids or not gpu_ids:
        raise RuntimeError("CPU/GPU tokenization failed.")
    cpu_id = int(cpu_ids[0])
    gpu_id = int(gpu_ids[0])

    values, ids = torch.topk(combined, k=min(top_k, combined.numel()))

    print(title)
    print("-" * len(title))
    print("Prompt:", text)
    print(f"gamma                  : {gamma:.2f}")
    print(f"gate                   : {gate_prob:.6f}")
    print(
        f"CPU id={cpu_id} rank={rank_of(combined, cpu_id)} "
        f"base={float(base_logits[cpu_id]):+.6f} "
        f"raw_bias={float(raw_bias[cpu_id]):+.6f} "
        f"scaled_bias={float(effective[cpu_id]):+.6f} "
        f"final={float(combined[cpu_id]):+.6f}"
    )
    print(
        f"GPU first-id={gpu_id} rank={rank_of(combined, gpu_id)} "
        f"base={float(base_logits[gpu_id]):+.6f} "
        f"raw_bias={float(raw_bias[gpu_id]):+.6f} "
        f"scaled_bias={float(effective[gpu_id]):+.6f} "
        f"final={float(combined[gpu_id]):+.6f}"
    )
    print(
        f"CPU-GPU final margin   : "
        f"{float(combined[cpu_id]-combined[gpu_id]):+.6f}"
    )
    print()
    print("Top vocabulary competitors")
    print("--------------------------")
    print("rank token-id token                 base       raw-bias   scaled-bias final")
    for rank, (value, token_id_t) in enumerate(zip(values.tolist(), ids.tolist()), start=1):
        token_id = int(token_id_t)
        token = safe_piece(tokenizer, token_id)
        marker = ""
        if token_id == cpu_id:
            marker = " <CPU>"
        elif token_id == gpu_id:
            marker = " <GPU-first>"
        print(
            f"{rank:>4d} {token_id:>8d} {token!r:20s} "
            f"{float(base_logits[token_id]):+10.6f} "
            f"{float(raw_bias[token_id]):+10.6f} "
            f"{float(effective[token_id]):+11.6f} "
            f"{float(combined[token_id]):+10.6f}{marker}"
        )
    print()

    cpu_final = float(combined[cpu_id].item())
    blockers = []
    for token_id in ids.tolist():
        token_id = int(token_id)
        if token_id == cpu_id:
            continue
        if float(combined[token_id].item()) > cpu_final:
            blockers.append(token_id)

    print("Tokens outranking CPU")
    print("---------------------")
    if not blockers:
        print("None")
    else:
        for token_id in blockers:
            print(
                f"id={token_id:4d} token={safe_piece(tokenizer, token_id)!r:20s} "
                f"gap={float(combined[token_id]-combined[cpu_id]):+.6f} "
                f"base_gap={float(base_logits[token_id]-base_logits[cpu_id]):+.6f} "
                f"bias_gap={float(effective[token_id]-effective[cpu_id]):+.6f}"
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

    print()
    print("====================================================")
    print(" v0.12.2 G05 Top-K Logit Diagnostic")
    print("====================================================")
    print("Device                 :", device)
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Semantic adapter loss  :", semantic_checkpoint.get("loss"))
    print("Name binding loss      :", name_checkpoint.get("loss"))
    print("v0.11 integration loss :", v011_checkpoint.get("loss"))
    print("v0.12.2 loss           :", checkpoint.get("loss"))
    print("Training               : none")
    print("Formula                : base + gamma * gate * direct_bias")
    print("Gains                  :", ", ".join(str(x) for x in GAINS))
    print("Top K                  :", args.top_k)
    print()

    for gamma in GAINS:
        print()
        print("#" * 72)
        print(f" gamma = {gamma:.2f}")
        print("#" * 72)
        print()

        diagnose_one(
            "Original G05",
            ORIGINAL_G05,
            gamma,
            args.top_k,
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
        diagnose_one(
            "Name-request G05",
            NAME_REQUEST_G05,
            gamma,
            args.top_k,
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


if __name__ == "__main__":
    main()
