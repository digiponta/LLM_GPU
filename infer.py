# infer.py
#
# Interactive inference using the physical GPU.

from pathlib import Path
import time

import torch

from model import LanguageModel
from tokenizer import Tokenizer


TOKENIZER_FILE = "model/tokenizer.json"
MODEL_FILE = "model/model-gpu-v0.4.pt"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40
REPETITION_PENALTY = 1.15


def main():
    print()
    print("====================================")
    print(" Homemade LLM GPU Inference v0.4")
    print("====================================")
    print()

    if not Path(TOKENIZER_FILE).exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_FILE}. "
            "Run python train_corpus.py first."
        )

    if not Path(MODEL_FILE).exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_FILE}. "
            "Run python train_corpus.py first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("CUDA available :", torch.cuda.is_available())
    print("Device         :", device)
    if device.type == "cuda":
        print("GPU            :", torch.cuda.get_device_name(0))

    tokenizer = Tokenizer.load(TOKENIZER_FILE)
    model, checkpoint = LanguageModel.load_checkpoint(
        MODEL_FILE,
        device=device,
    )

    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    print("Vocabulary size:", tokenizer.vocab_size)
    print("Parameters     :", f"{model.parameter_count:,}")
    print("Checkpoint loss:", checkpoint.get("loss"))
    print()
    print("Enter Japanese text.")
    print("Type 'exit' to quit.")
    print()

    while True:
        prompt = input("You> ")

        if prompt.strip().lower() in ("exit", "quit"):
            break
        if not prompt:
            continue

        input_ids = tokenizer.encode(
            prompt,
            add_bos=True,
        )

        start = time.perf_counter()

        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_id=tokenizer.eos_id,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        new_tokens = max(0, len(output_ids) - len(input_ids))
        rate = new_tokens / elapsed if elapsed > 0 else 0.0

        generated_text = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        )

        print()
        print("LLM>", generated_text)
        print(
            f"[Generated {new_tokens} tokens "
            f"in {elapsed:.2f}s, {rate:.2f} tokens/s]"
        )
        print()


if __name__ == "__main__":
    main()
