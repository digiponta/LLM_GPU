# chat_semantic_token_conditioning_v091.py
#
# Chat/inference for Semantic-to-Generation Interface v0.7.

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from semantic_generation_integration_v09 import (
    infer_semantic_condition,
    load_frozen_semantic_path,
)
from semantic_token_conditioning_v091 import (
    SEMANTIC_TOKEN_LABELS,
    forward_semantic_tokens,
    load_semantic_token_checkpoint,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9-semantic-adapter-v05.pt"
DEFAULT_TOKEN_MODEL = "model/model-gpu-v0.9.1-semantic-token-v07.pt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "


def build_prompt(history, user_text, history_turns):
    chunks = []
    if history_turns > 0:
        for old_user, old_ai in history[-history_turns:]:
            chunks.append(
                f"{USER_PREFIX}{old_user}\n{AI_PREFIX}{old_ai}\n"
            )
    chunks.append(f"{USER_PREFIX}{user_text}\n{AI_PREFIX}")
    return "".join(chunks)


@torch.no_grad()
def compute_semantic_tokens(
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    projector,
    tokenizer,
    prompt,
):
    ids = tokenizer.encode(prompt, add_bos=True)
    context = ids[-semantic_model.context_length:]
    device = next(semantic_model.parameters()).device
    x = torch.tensor([context], dtype=torch.long, device=device)

    (
        adapted,
        concept_prob,
        attribute_prob,
        hierarchy_prob,
    ) = infer_semantic_condition(
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        x,
        prompt_index=None,
    )

    semantic_tokens = projector(
        adapted,
        concept_prob,
        attribute_prob,
        hierarchy_prob,
    )

    return semantic_tokens, {
        "concept_prob": concept_prob[0].tolist(),
        "attribute_prob": attribute_prob[0].tolist(),
        "hierarchy_prob": hierarchy_prob[0].tolist(),
        "token_norms": semantic_tokens[0].norm(dim=-1).tolist(),
    }


@torch.no_grad()
def generate_reply(
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    projector,
    tokenizer,
    prompt,
    max_new_tokens=96,
    temperature=0.45,
    top_k=20,
    repetition_penalty=1.05,
):
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    generated = list(prompt_ids)
    response_ids: List[int] = []
    device = next(generation_model.parameters()).device

    generation_model.eval()
    semantic_model.eval()
    semantic_adapter.eval()
    semantic_heads.eval()
    hierarchy_head.eval()
    projector.eval()

    semantic_tokens, details = compute_semantic_tokens(
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        projector,
        tokenizer,
        prompt,
    )

    for _ in range(max_new_tokens):
        context = generated[-generation_model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)

        hidden = forward_semantic_tokens(
            generation_model,
            x,
            semantic_tokens,
            projector.inject_after,
        )
        logits = generation_model.lm_head(hidden[:, -1, :])[0].clone()

        if repetition_penalty != 1.0:
            for token_id in set(response_ids):
                if logits[token_id] >= 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

        if temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_k is not None and 0 < top_k < logits.numel():
                values, indices = torch.topk(logits, top_k)
                probs = F.softmax(values, dim=-1)
                selected = torch.multinomial(probs, 1)
                next_id = int(indices[selected].item())
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())

        if next_id == tokenizer.eos_id:
            break

        generated.append(next_id)
        response_ids.append(next_id)

        decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
        if "\n" in decoded:
            break

    reply = tokenizer.decode(response_ids, skip_special_tokens=True)
    for marker in ("\n人:", "\nAI:", "\n"):
        if marker in reply:
            reply = reply.split(marker, 1)[0]

    return reply.strip(), len(response_ids), details


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--token-model", default=DEFAULT_TOKEN_MODEL)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.45)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--history-turns", type=int, default=3)
    args = p.parse_args()

    for filename in (
        args.tokenizer,
        args.base_model,
        args.semantic_adapter,
        args.token_model,
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

    generation_model, projector, token_checkpoint = (
        load_semantic_token_checkpoint(args.token_model, device)
    )

    print()
    print("==============================================")
    print(" Semantic Token Conditioning v0.7 Chat")
    print("==============================================")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Base loss       :", base_checkpoint.get("loss"))
    print("Semantic loss   :", semantic_checkpoint.get("loss"))
    print("Token loss      :", token_checkpoint.get("loss"))
    print("Semantic tokens :", ", ".join(SEMANTIC_TOKEN_LABELS))
    print("Insert after    :", projector.inject_after)
    print("Token scale     :", projector.scale)
    print()

    history: List[Tuple[str, str]] = []
    while True:
        user_text = input("You> ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("/exit", "exit", "quit"):
            break
        if user_text.lower() == "/reset":
            history.clear()
            print("[conversation history cleared]")
            continue

        prompt = build_prompt(history, user_text, args.history_turns)
        started = time.perf_counter()

        reply, count, details = generate_reply(
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            projector,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.repetition_penalty,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - started
        rate = count / elapsed if elapsed else 0.0

        if not reply:
            reply = "(no response)"

        print(f"AI> {reply}")
        print(f"[{count} tokens, {elapsed:.2f}s, {rate:.1f} tok/s]")
        print(
            "[semantic-token norms "
            + ", ".join(
                f"{label}={value:.3f}"
                for label, value in zip(
                    SEMANTIC_TOKEN_LABELS,
                    details["token_norms"],
                )
            )
            + "]"
        )
        print()

        history.append((user_text, reply))


if __name__ == "__main__":
    main()
