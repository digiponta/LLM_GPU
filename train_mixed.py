# train_mixed.py
#
# LLM_GPU v0.6 mixed pretraining.
#
# Default mixture:
#   70% general Japanese
#   20% conversational text
#   10% instruction / QA
#
# The model is trained from scratch with:
#   d_model=128, 4 layers, 4 heads, FFN=512, context=256,
#   learned positional embeddings.

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from model import LanguageModel
from tokenizer import Tokenizer
from train import Trainer


TOKENIZER_FILE = "model/tokenizer-v0.6.json"
MODEL_FILE = "model/model-gpu-v0.6-pretrain.pt"

CONTEXT_LENGTH = 256
D_MODEL = 128
NUM_LAYERS = 4
NUM_HEADS = 4
HIDDEN_DIM = 512

SEED = 42
REQUIRE_CUDA = True


def find_data_file(filename: str) -> Path:
    candidates = [
        Path("data") / filename,
        Path("..") / "LLM" / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Searched:\n{searched}")


def clean_comment_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#")
    ).strip()


class MixedTokenWindowDataset(Dataset):
    """Memory-light deterministic 70/20/10 source mixer."""

    def __init__(
        self,
        sources: Sequence[Tuple[str, Sequence[int], int]],
        context_length: int,
        sample_count: int,
        seed: int = 42,
    ):
        self.context_length = context_length
        self.sample_count = int(sample_count)
        self.seed = int(seed)
        self.sources = []

        if self.sample_count <= 0:
            raise ValueError("sample_count must be > 0.")

        total_weight = sum(weight for _, _, weight in sources)
        if total_weight <= 0:
            raise ValueError("Source weights must sum to > 0.")

        cumulative = 0
        for name, token_ids, weight in sources:
            ids = list(token_ids)
            positions = len(ids) - context_length
            if positions <= 0:
                raise ValueError(
                    f"Source {name!r} is too short for context "
                    f"{context_length}: {len(ids)} tokens"
                )
            cumulative += weight
            self.sources.append({
                "name": name,
                "ids": ids,
                "positions": positions,
                "weight": weight,
                "cumulative": cumulative,
            })

        self.total_weight = total_weight

    def __len__(self) -> int:
        return self.sample_count

    def _source_for_index(self, index: int):
        bucket = index % self.total_weight
        for source in self.sources:
            if bucket < source["cumulative"]:
                return source
        return self.sources[-1]

    def __getitem__(self, index: int):
        if index < 0 or index >= self.sample_count:
            raise IndexError("dataset index out of range")

        source = self._source_for_index(index)
        positions = source["positions"]

        # Deterministic pseudo-random source-local starting position.
        start = (
            self.seed * 104729
            + index * 2654435761
            + len(source["name"]) * 7919
        ) % positions

        ids = source["ids"]
        end = start + self.context_length

        x = torch.tensor(ids[start:end], dtype=torch.long)
        y = torch.tensor(ids[start + 1:end + 1], dtype=torch.long)
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM_GPU v0.6 mixed pretraining."
    )
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if REQUIRE_CUDA:
        raise RuntimeError(
            "CUDA is not available. Run python check_gpu.py first."
        )
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)

    print()
    print("====================================")
    print(" LLM_GPU v0.6 Mixed Pretraining")
    print("====================================")
    print()

    device = select_device()

    general_path = find_data_file("general-ja.txt")
    nagato_path = find_data_file("data-nagato.txt")
    conversation_path = Path("data/conversation-ja.txt")
    instruction_path = Path("data/instruction-ja.txt")

    for path in (conversation_path, instruction_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    general_text = (
        general_path.read_text(encoding="utf-8")
        + "\n\n"
        + nagato_path.read_text(encoding="utf-8")
    )
    conversation_text = clean_comment_lines(
        conversation_path.read_text(encoding="utf-8")
    )
    instruction_text = clean_comment_lines(
        instruction_path.read_text(encoding="utf-8")
    )

    print("General corpus      :", f"{len(general_text):,}", "characters")
    print("Conversation corpus :", f"{len(conversation_text):,}", "characters")
    print("Instruction corpus  :", f"{len(instruction_text):,}", "characters")

    print()
    print("Building v0.6 tokenizer from all three sources...")
    tokenizer = Tokenizer()
    tokenizer.fit_texts([
        general_text,
        conversation_text,
        instruction_text,
    ])
    Path("model").mkdir(exist_ok=True)
    tokenizer.save(TOKENIZER_FILE)

    general_ids = tokenizer.encode(
        general_text, add_bos=True, add_eos=True
    )
    conversation_ids = tokenizer.encode(
        conversation_text, add_bos=True, add_eos=True
    )
    instruction_ids = tokenizer.encode(
        instruction_text, add_bos=True, add_eos=True
    )

    dataset = MixedTokenWindowDataset(
        sources=[
            ("general", general_ids, 7),
            ("conversation", conversation_ids, 2),
            ("instruction", instruction_ids, 1),
        ],
        context_length=CONTEXT_LENGTH,
        sample_count=args.samples,
        seed=SEED,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        causal=True,
        context_length=CONTEXT_LENGTH,
        use_position_embedding=True,
    ).to(device)

    presented_tokens = (
        args.samples * CONTEXT_LENGTH * args.epochs
    )

    print()
    print("Device           :", device)
    if device.type == "cuda":
        print("GPU              :", torch.cuda.get_device_name(0))
    print("Vocabulary       :", tokenizer.vocab_size)
    print("Parameters       :", f"{model.parameter_count:,}")
    print("Context length   :", CONTEXT_LENGTH)
    print("d_model          :", D_MODEL)
    print("Layers           :", NUM_LAYERS)
    print("Attention heads  :", NUM_HEADS)
    print("FFN dimension    :", HIDDEN_DIM)
    print("Mixture          : 70% general / 20% conversation / 10% instruction")
    print("Samples/epoch    :", f"{args.samples:,}")
    print("Epochs           :", args.epochs)
    print("Batch size       :", args.batch_size)
    print("Learning rate    :", args.learning_rate)
    print("Token exposures  :", f"{presented_tokens:,}")
    print("Tokenizer        :", TOKENIZER_FILE)
    print()

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
    )
    history = trainer.train(loader=loader, epochs=args.epochs)
    final_loss = history[-1] if history else None

    model.save_checkpoint(
        MODEL_FILE,
        optimizer=trainer.optimizer,
        epoch=args.epochs,
        loss=final_loss,
    )

    print()
    print("Mixed pretraining completed.")
    print("Loss history     :", history)
    print("Model saved      :", MODEL_FILE)
    print()
    print("Next:")
    print("  python train_sft_v06.py")


if __name__ == "__main__":
    main()
