# chat.py
#
# Interactive chat interface for the current LLM_GPU conversational checkpoint.
# Defaults to the v0.8 cleaned chat model used by the v1.4/v1.5 semantic experiments.
#
# Features:
#   - CUDA / CPU auto-selection
#   - short multi-turn history
#   - conservative sampling for the small model
#   - response-only repetition penalty
#   - newline / EOS stop
#   - /reset, /info, /exit commands

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
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the current LLM_GPU conversational model."
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
        help="Number of previous turns included in the prompt.",
    )
    return parser.parse_args()


def build_prompt(
    history: List[Tuple[str, str]],
    user_text: str,
    history_turns: int,
) -> str:
    chunks: List[str] = []

    if history_turns > 0:
        for old_user, old_ai in history[-history_turns:]:
            chunks.append(
                f"{USER_PREFIX}{old_user}\n"
                f"{AI_PREFIX}{old_ai}\n"
            )

    chunks.append(f"{USER_PREFIX}{user_text}\n{AI_PREFIX}")
    return "".join(chunks)


@torch.no_grad()
def generate_reply(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
) -> Tuple[str, int]:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    generated = list(prompt_ids)
    response_ids: List[int] = []

    model.eval()
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        context = generated[-model.context_length:]
        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device,
        )

        logits = model(x)[0, -1, :].clone()

        # Penalize only tokens already emitted in the assistant response.
        if repetition_penalty != 1.0:
            for token_id in set(response_ids):
                if 0 <= token_id < logits.numel():
                    if logits[token_id] >= 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty

        if temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature

            if 0 < top_k < logits.numel():
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

        decoded = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )

        # Stop at the end of the first assistant line.
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


def print_info(
    model: LanguageModel,
    tokenizer: Tokenizer,
    checkpoint: dict,
    device: torch.device,
    model_path: Path,
    tokenizer_path: Path,
) -> None:
    print()
    print("==============================================")
    print(" LLM_GPU Chat - current v1.5.x experiment base")
    print("==============================================")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Model           :", model_path)
    print("Tokenizer       :", tokenizer_path)
    print("Vocabulary size :", tokenizer.vocab_size)
    print("Parameters      :", f"{model.parameter_count:,}")
    print("Context length  :", model.context_length)
    print("d_model         :", model.d_model)
    print("Layers          :", model.num_layers)
    print("Attention heads :", model.num_heads)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print()


def main() -> None:
    args = parse_args()

    tokenizer_path = Path(args.tokenizer)
    model_path = Path(args.model)

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Chat model not found: {model_path}\n"
            "Expected the cleaned v0.8 conversational checkpoint."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    print_info(
        model=model,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        device=device,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )

    print("Commands:")
    print("  /reset  clear conversation history")
    print("  /info   show model/checkpoint information")
    print("  /exit   quit")
    print()

    history: List[Tuple[str, str]] = []

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue

        command = user_text.lower()

        if command in ("/exit", "exit", "quit"):
            break

        if command == "/reset":
            history.clear()
            print("[conversation history cleared]")
            print()
            continue

        if command == "/info":
            print_info(
                model=model,
                tokenizer=tokenizer,
                checkpoint=checkpoint,
                device=device,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
            )
            continue

        prompt = build_prompt(
            history=history,
            user_text=user_text,
            history_turns=args.history_turns,
        )

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
        print(
            f"[{new_tokens} tokens, "
            f"{elapsed:.2f}s, "
            f"{rate:.1f} tok/s]"
        )
        print()

        history.append((user_text, reply))


if __name__ == "__main__":
    main()
