# chat.py
#
# Interactive short-dialogue interface for LLM_GPU v0.5.

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import List, Tuple

import torch

from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_MODEL = "model/model-gpu-v0.5-chat.pt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "
STOP_MARKERS = ("\n人:", "\nAI:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the LLM_GPU v0.5 conversational model."
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument(
        "--history-turns",
        type=int,
        default=2,
        help="Number of previous user/AI turns kept before context truncation.",
    )
    return parser.parse_args()


def build_prompt(
    history: List[Tuple[str, str]],
    user_text: str,
    history_turns: int,
) -> str:
    chunks = []
    for old_user, old_ai in history[-max(0, history_turns):]:
        chunks.append(f"{USER_PREFIX}{old_user}\n{AI_PREFIX}{old_ai}\n")
    chunks.append(f"{USER_PREFIX}{user_text}\n{AI_PREFIX}")
    return "".join(chunks)


def clean_reply(text: str) -> str:
    reply = text
    for marker in STOP_MARKERS:
        if marker in reply:
            reply = reply.split(marker, 1)[0]
    return reply.strip()


@torch.no_grad()
def generate_reply(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_k: int = 30,
    repetition_penalty: float = 1.12,
) -> Tuple[str, int]:
    input_ids = tokenizer.encode(prompt, add_bos=True)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    new_ids = output_ids[len(input_ids):]
    reply = clean_reply(
        tokenizer.decode(new_ids, skip_special_tokens=True)
    )
    return reply, len(new_ids)


def main() -> None:
    args = parse_args()

    tokenizer_path = Path(args.tokenizer)
    model_path = Path(args.model)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. "
            "Run python train_corpus.py first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Chat model not found: {model_path}. "
            "Run python train_conversation.py first."
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
    print(" LLM_GPU v0.5 Chat")
    print("====================================")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Vocabulary size :", tokenizer.vocab_size)
    print("Parameters      :", f"{model.parameter_count:,}")
    print("Context length  :", model.context_length)
    print("Checkpoint loss :", checkpoint.get("loss"))
    print()
    print("Type Japanese text. Commands: /reset, /exit")
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

        print(f"AI> {reply}")
        print(f"[{new_tokens} tokens, {elapsed:.2f}s, {rate:.1f} tok/s]")
        print()

        history.append((user_text, reply))


if __name__ == "__main__":
    main()
