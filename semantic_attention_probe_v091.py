# semantic_attention_probe_v091.py
#
# LLM_GPU v0.9.1
# Semantic Attention Probe for Balanced Semantic Tokens v0.8.
#
# No training. No checkpoint mutation.
#
# Measures whether Blocks 4-6 actually attend from text positions to:
#   SEM_IDENTITY
#   SEM_CONCEPT
#   SEM_HIERARCHY
#
# Reports per block / per head:
#   - attention from last prompt token to each semantic token
#   - total last-token semantic attention mass
#   - mean text-to-semantic attention mass
#   - ratio to a uniform-attention baseline

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from chat_semantic_token_balanced_v091 import (
    AI_PREFIX,
    USER_PREFIX,
    compute_balanced_semantic_tokens,
)
from evaluate_partial_intent_v09 import CASES
from semantic_generation_integration_v09 import load_frozen_semantic_path
from semantic_token_balanced_v091 import (
    load_balanced_semantic_token_checkpoint,
)
from semantic_token_conditioning_v091 import SEMANTIC_TOKEN_LABELS
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9-semantic-adapter-v05.pt"
DEFAULT_BALANCED = "model/model-gpu-v0.9.1-semantic-token-balanced-v08.pt"

DEFAULT_CASE_IDS = [5, 8, 9]


def parse_args():
    p = argparse.ArgumentParser(
        description="Probe attention to balanced semantic tokens."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--balanced", default=DEFAULT_BALANCED)
    p.add_argument(
        "--cases",
        default="5,8,9",
        help="Comma-separated fixed benchmark case numbers, e.g. 5,8,9",
    )
    return p.parse_args()


def parse_case_ids(value: str):
    case_ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        case_id = int(item)
        if case_id < 1 or case_id > len(CASES):
            raise ValueError(f"Case id out of range: {case_id}")
        case_ids.append(case_id)
    if not case_ids:
        raise ValueError("At least one case id is required.")
    return case_ids


def initial_hidden(model, token_ids):
    if token_ids.dim() != 2:
        raise ValueError("token_ids must have shape [batch, time].")

    _, time = token_ids.shape
    x = model.embedding(token_ids)

    if model.position_embedding is not None:
        positions = torch.arange(time, device=token_ids.device)
        x = x + model.position_embedding(positions).unsqueeze(0)

    return x


@torch.no_grad()
def attention_weights(attention, x):
    normalized_qkv = x
    q = attention._split_heads(attention.q_proj(normalized_qkv))
    k = attention._split_heads(attention.k_proj(normalized_qkv))

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(float(attention.head_dim))

    if attention.causal:
        time = x.size(1)
        mask = torch.triu(
            torch.ones(
                (time, time),
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    return F.softmax(scores, dim=-1)


@torch.no_grad()
def probe_case(
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    projector,
    tokenizer,
    prompt,
):
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    context = prompt_ids[-generation_model.context_length:]
    device = next(generation_model.parameters()).device
    token_ids = torch.tensor([context], dtype=torch.long, device=device)

    semantic_tokens, semantic_details = compute_balanced_semantic_tokens(
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        projector,
        tokenizer,
        prompt,
    )

    x = initial_hidden(generation_model, token_ids)

    for block_index, block in enumerate(generation_model.blocks, start=1):
        if block_index <= projector.inject_after:
            x = block(x)

    semantic_count = semantic_tokens.size(1)
    text_count = x.size(1)
    x = torch.cat([semantic_tokens, x], dim=1)

    results = []

    for block_index, block in enumerate(generation_model.blocks, start=1):
        if block_index <= projector.inject_after:
            continue

        normalized = block.norm1(x)
        weights = attention_weights(block.attention, normalized)[0]
        # shape: [heads, query, key]

        text_query_start = semantic_count
        last_query = semantic_count + text_count - 1

        last_sem = weights[:, last_query, :semantic_count]
        last_mass = last_sem.sum(dim=-1)

        all_text_sem = weights[
            :,
            text_query_start:,
            :semantic_count,
        ]
        mean_text_mass = all_text_sem.sum(dim=-1).mean(dim=-1)

        # For the last prompt token, all preceding keys are visible.
        visible_key_count = semantic_count + text_count
        uniform_semantic_mass = semantic_count / float(visible_key_count)

        results.append(
            {
                "block": block_index,
                "last_semantic_by_head": last_sem.detach().cpu(),
                "last_mass_by_head": last_mass.detach().cpu(),
                "mean_text_mass_by_head": mean_text_mass.detach().cpu(),
                "uniform_semantic_mass": uniform_semantic_mass,
            }
        )

        # Advance with the exact trained block implementation.
        x = block(x)

    return results, semantic_details, text_count


def print_case(case_id, prompt_text, results, semantic_details, text_count):
    print()
    print("=" * 72)
    print(f"G{case_id:02d}: {prompt_text}")
    print("=" * 72)
    print("Prompt token count :", text_count)
    print(
        "Semantic token norms:",
        ", ".join(
            f"{label}={value:.3f}"
            for label, value in zip(
                SEMANTIC_TOKEN_LABELS,
                semantic_details["token_norms"],
            )
        ),
    )
    print(
        "Learned token scales:",
        ", ".join(
            f"{label}={value:.3f}"
            for label, value in zip(
                SEMANTIC_TOKEN_LABELS,
                semantic_details["token_scales"],
            )
        ),
    )

    for item in results:
        block = item["block"]
        last_sem = item["last_semantic_by_head"]
        last_mass = item["last_mass_by_head"]
        mean_mass = item["mean_text_mass_by_head"]
        uniform = item["uniform_semantic_mass"]

        print()
        print(f"Block {block}")
        print("-" * 72)
        print(
            f"Uniform last-token semantic mass baseline: {uniform:.6f}"
        )

        for head in range(last_sem.size(0)):
            values = last_sem[head].tolist()
            mass = float(last_mass[head].item())
            mean_text = float(mean_mass[head].item())
            ratio = mass / uniform if uniform > 0 else float("nan")

            print(
                f"Head {head:02d} | "
                + " ".join(
                    f"{label}={value:.6f}"
                    for label, value in zip(
                        SEMANTIC_TOKEN_LABELS,
                        values,
                    )
                )
            )
            print(
                f"         last_sem_mass={mass:.6f} "
                f"uniform_ratio={ratio:.3f} "
                f"mean_text_sem_mass={mean_text:.6f}"
            )

        avg_last_sem = last_sem.mean(dim=0)
        avg_last_mass = float(last_mass.mean().item())
        avg_mean_text = float(mean_mass.mean().item())
        avg_ratio = avg_last_mass / uniform if uniform > 0 else float("nan")

        print(
            "Average | "
            + " ".join(
                f"{label}={value:.6f}"
                for label, value in zip(
                    SEMANTIC_TOKEN_LABELS,
                    avg_last_sem.tolist(),
                )
            )
        )
        print(
            f"          last_sem_mass={avg_last_mass:.6f} "
            f"uniform_ratio={avg_ratio:.3f} "
            f"mean_text_sem_mass={avg_mean_text:.6f}"
        )


def main():
    args = parse_args()
    case_ids = parse_case_ids(args.cases)

    for filename in (
        args.tokenizer,
        args.base_model,
        args.semantic_adapter,
        args.balanced,
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
    ) = load_frozen_semantic_path(
        args.base_model,
        args.semantic_adapter,
        device,
    )

    generation_model, projector, balanced_checkpoint = (
        load_balanced_semantic_token_checkpoint(
            args.balanced,
            device,
        )
    )

    generation_model.eval()
    projector.eval()

    print()
    print("====================================================")
    print(" Semantic Attention Probe")
    print("====================================================")
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", base_checkpoint.get("loss"))
    print("Semantic adapter loss:", semantic_checkpoint.get("loss"))
    print("Balanced val loss   :", balanced_checkpoint.get("loss"))
    print("Probe cases         :", ", ".join(f"G{x:02d}" for x in case_ids))
    print("Semantic tokens     :", ", ".join(SEMANTIC_TOKEN_LABELS))
    print("Insert after        :", projector.inject_after)
    print("Training            : none")
    print()

    for case_id in case_ids:
        case = CASES[case_id - 1]
        prompt_text = str(case["prompt"])
        prompt = f"{USER_PREFIX}{prompt_text}\n{AI_PREFIX}"

        results, semantic_details, text_count = probe_case(
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            projector,
            tokenizer,
            prompt,
        )

        print_case(
            case_id,
            prompt_text,
            results,
            semantic_details,
            text_count,
        )

    print()
    print("Interpretation")
    print("--------------")
    print("uniform_ratio < 1.0 : semantic tokens are under-attended")
    print("uniform_ratio ~ 1.0 : roughly uniform semantic attention")
    print("uniform_ratio > 1.0 : semantic tokens are preferentially attended")
    print()
    print(
        "If G05 has correct semantic signals but consistently low "
        "text-to-semantic attention, the next architecture candidate is "
        "explicit Semantic Cross-Attention."
    )


if __name__ == "__main__":
    main()
