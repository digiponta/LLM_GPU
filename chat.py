# chat.py
#
# Interactive short-dialogue interface for LLM_GPU v0.8.
# Generation is intentionally conservative for the very small model:
# low temperature, small top-k, response-only repetition penalty, and
# immediate stop on newline/EOS.

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat.pt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the LLM_GPU v0.8 conversational model."
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--history-turns",
        type=int,
        default=3,
        help="Previous turns included before the current user prompt.",
    )
    return parser.parse_args()


def build_prompt(
    history: List[Tuple[str, str]],
    user_text: str,
    history_turns: int,
) -> str:
    chunks = []
    if history_turns > 0:
        for old_user, old_ai in history[-history_turns:]:
            chunks.append(
                f"{USER_PREFIX}{old_user}\n{AI_PREFIX}{old_ai}\n"
            )
    chunks.append(f"{USER_PREFIX}{user_text}\n{AI_PREFIX}")
    return "".join(chunks)


@torch.no_grad()
def generate_reply(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 96,
    temperature: float = 0.45,
    top_k: int = 20,
    repetition_penalty: float = 1.05,
) -> Tuple[str, int]:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    generated = list(prompt_ids)
    response_ids: List[int] = []

    model.eval()
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        context = generated[-model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :].clone()

        # Penalize only tokens already emitted in the assistant reply.
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

        # BPE may represent a newline or role boundary with multiple IDs.
        # Decode the accumulated response and detect boundaries in text space.
        decoded = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )
        if "\n" in decoded:
            break

    reply = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    )

    for marker in ("\n人:", "\nAI:", "\n"):
        if marker in reply:
            reply = reply.split(marker, 1)[0]

    return reply.strip(), len(response_ids)


def main() -> None:
    args = parse_args()

    tokenizer_path = Path(args.tokenizer)
    model_path = Path(args.model)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Chat model not found: {model_path}. "
            "Run python train_sft_v07.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(model_path),
        device=device,
    )

    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    print()
    print("====================================")
    print(" LLM_GPU v0.8 Chat")
    print("====================================")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Vocabulary size :", tokenizer.vocab_size)
    print("Parameters      :", f"{model.parameter_count:,}")
    print("Context length  :", model.context_length)
    print("Checkpoint loss :", checkpoint.get("loss"))
    print()
    print("Commands: /reset, /exit")
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

        start = time.perf_counter()
        reply, new_tokens = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        rate = new_tokens / elapsed if elapsed > 0 else 0.0

        if not reply:
            reply = "(no response)"

        print(f"AI> {reply}")
        print(f"[{new_tokens} tokens, {elapsed:.2f}s, {rate:.1f} tok/s]")
        print()

        history.append((user_text, reply))


if __name__ == "__main__":
    main()
