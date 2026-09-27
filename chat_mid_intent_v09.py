# chat_mid_intent_v09.py
#
# LLM_GPU v0.9 Mid-Layer Intent-Conditioned Generation.

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

from intent_conditioning_v09 import load_intent_head
from mid_intent_conditioning_v09 import (
    forward_mid_conditioned,
    infer_prompt_intent,
    load_mid_projection,
)
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INTENT_HEAD = "model/model-gpu-v0.8-intent-head.pt"
DEFAULT_PROJECTION = "model/model-gpu-v0.9-mid-intent.pt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with v0.9 mid-layer intent model.")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT_HEAD)
    p.add_argument("--projection", default=DEFAULT_PROJECTION)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.45)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--history-turns", type=int, default=3)
    return p.parse_args()


def build_prompt(history, user_text, history_turns):
    chunks = []
    if history_turns > 0:
        for old_user, old_ai in history[-history_turns:]:
            chunks.append(f"{USER_PREFIX}{old_user}\n{AI_PREFIX}{old_ai}\n")
    chunks.append(f"{USER_PREFIX}{user_text}\n{AI_PREFIX}")
    return "".join(chunks)


@torch.no_grad()
def compute_intent_bias(model, tokenizer, intent_head, projection, prompt):
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    context = prompt_ids[-model.context_length:]
    device = next(model.parameters()).device
    x = torch.tensor([context], dtype=torch.long, device=device)
    intent_prob = infer_prompt_intent(model, intent_head, x)
    return projection(intent_prob), intent_prob


@torch.no_grad()
def generate_reply(
    model,
    tokenizer,
    intent_head,
    projection,
    prompt,
    max_new_tokens=96,
    temperature=0.45,
    top_k=20,
    repetition_penalty=1.05,
):
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    generated = list(prompt_ids)
    response_ids: List[int] = []
    device = next(model.parameters()).device

    model.eval()
    intent_head.eval()
    projection.eval()

    intent_bias, intent_prob = compute_intent_bias(
        model, tokenizer, intent_head, projection, prompt
    )

    for _ in range(max_new_tokens):
        context = generated[-model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        hidden = forward_mid_conditioned(
            model,
            x,
            intent_bias,
            projection.inject_after,
        )
        logits = model.lm_head(hidden[:, -1, :])[0].clone()

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

    return reply.strip(), len(response_ids), intent_prob[0].tolist()


def main() -> None:
    args = parse_args()
    for filename in (
        args.tokenizer, args.model, args.intent_head, args.projection
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    intent_head, _head_ckpt, labels = load_intent_head(
        args.intent_head, model, device
    )
    projection, projection_ckpt = load_mid_projection(
        args.projection, model, labels, device
    )

    print()
    print("====================================")
    print(" LLM_GPU v0.9 Mid-Intent Chat")
    print("====================================")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Base loss       :", checkpoint.get("loss"))
    print("Projection loss :", projection_ckpt.get("loss"))
    print("Inject after    :", projection.inject_after)
    print("Alpha           :", projection.alpha)
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
        reply, count, _ = generate_reply(
            model, tokenizer, intent_head, projection, prompt,
            args.max_new_tokens, args.temperature, args.top_k,
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
        print()
        history.append((user_text, reply))


if __name__ == "__main__":
    main()
