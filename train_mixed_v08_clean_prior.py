# train_mixed_v08_clean_prior.py
#
# LLM_GPU v0.8 curriculum pretraining with byte-level BPE.
#
# v0.8 capacity experiment:
# d_model=256, 6 layers, 8 heads, FFN=1024, context=512.
# Tokenizer and training data remain aligned with v0.7.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from train import Trainer


TOKENIZER_FILE = "model/tokenizer-v0.7-bpe.json"
MODEL_FILE = "model/model-gpu-v0.8-pretrain-clean.pt"

CONTEXT_LENGTH = 512
D_MODEL = 256
NUM_LAYERS = 6
NUM_HEADS = 8
HIDDEN_DIM = 1024

SEED = 42
REQUIRE_CUDA = True

WARMUP_WEIGHTS = {
    "general": 90,
    "conversation": 7,
    "instruction": 3,
}
TARGET_WEIGHTS = {
    "general": 70,
    "conversation": 20,
    "instruction": 10,
}


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


class CurriculumMixedDataset(Dataset):
    def __init__(
        self,
        sources: Sequence[Tuple[str, Sequence[int]]],
        context_length: int,
        sample_count: int,
        warmup_ratio: float = 0.80,
        seed: int = 42,
    ):
        if sample_count <= 0:
            raise ValueError("sample_count must be > 0.")
        if not 0.0 <= warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be between 0 and 1.")

        self.context_length = int(context_length)
        self.sample_count = int(sample_count)
        self.warmup_count = int(round(sample_count * warmup_ratio))
        self.seed = int(seed)
        self.sources: Dict[str, Dict[str, object]] = {}

        for name, token_ids in sources:
            ids = list(token_ids)
            positions = len(ids) - self.context_length
            if positions <= 0:
                raise ValueError(
                    f"Source {name!r} is too short for context "
                    f"{self.context_length}: {len(ids)} tokens"
                )
            self.sources[name] = {
                "ids": ids,
                "positions": positions,
            }

    def __len__(self) -> int:
        return self.sample_count

    @staticmethod
    def _pick_name(index: int, weights: Dict[str, int]) -> str:
        total = sum(weights.values())
        bucket = index % total
        cumulative = 0
        for name, weight in weights.items():
            cumulative += weight
            if bucket < cumulative:
                return name
        return next(reversed(weights))

    def __getitem__(self, index: int):
        if index < 0 or index >= self.sample_count:
            raise IndexError("dataset index out of range")

        if index < self.warmup_count:
            weights = WARMUP_WEIGHTS
            phase_offset = 0
        else:
            weights = TARGET_WEIGHTS
            phase_offset = self.warmup_count

        name = self._pick_name(index - phase_offset, weights)
        source = self.sources[name]
        ids = source["ids"]
        positions = int(source["positions"])

        salt = {
            "general": 101,
            "conversation": 211,
            "instruction": 307,
        }[name]
        start = (
            self.seed * 104729
            + index * 2654435761
            + salt * 7919
        ) % positions

        end = start + self.context_length
        return (
            torch.tensor(ids[start:end], dtype=torch.long),
            torch.tensor(ids[start + 1:end + 1], dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM_GPU v0.8 clean-prior BPE curriculum pretraining."
    )
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.80)
    parser.add_argument("--tokenizer", default=TOKENIZER_FILE)
    parser.add_argument("--output", default=MODEL_FILE)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if REQUIRE_CUDA:
        raise RuntimeError("CUDA is not available. Run python check_gpu.py.")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)

    print()
    print("====================================")
    print(" LLM_GPU v0.8 Clean-Prior BPE Pretraining")
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

    # v0.10.8 controlled base-prior ablation:
    # remove the lexical cue from *all* pretraining sources, including
    # general-ja/data-nagato, so the rebuilt v0.8 model cannot relearn it.
    general_before = general_text.count("多数")
    conversation_before = conversation_text.count("多数")
    instruction_before = instruction_text.count("多数")
    general_text = general_text.replace("多数", "多く")
    conversation_text = conversation_text.replace("多数", "多く")
    instruction_text = instruction_text.replace("多数", "多く")

    print("Ablation word       : 多数 -> 多く")
    print("Removed from general:", general_before)
    print("Removed from conv.  :", conversation_before)
    print("Removed from instr. :", instruction_before)
    print("Remaining 多数      :", general_text.count("多数") + conversation_text.count("多数") + instruction_text.count("多数"))
    print("General corpus      :", f"{len(general_text):,}", "characters")
    print("Conversation corpus :", f"{len(conversation_text):,}", "characters")
    print("Instruction corpus  :", f"{len(instruction_text):,}", "characters")

    print()
    print("Loading fixed v0.7 byte-level BPE tokenizer...")
    if not Path(args.tokenizer).exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {args.tokenizer}. "
            "v0.8 intentionally reuses the v0.7 tokenizer."
        )
    tokenizer = Tokenizer.load(args.tokenizer)
    Path("model").mkdir(exist_ok=True)

    general_ids = tokenizer.encode(general_text, add_bos=True, add_eos=True)
    conversation_ids = tokenizer.encode(
        conversation_text, add_bos=True, add_eos=True
    )
    instruction_ids = tokenizer.encode(
        instruction_text, add_bos=True, add_eos=True
    )

    char_count = (
        len(general_text)
        + len(conversation_text)
        + len(instruction_text)
    )
    token_count = (
        len(general_ids)
        + len(conversation_ids)
        + len(instruction_ids)
        - 6
    )

    dataset = CurriculumMixedDataset(
        sources=[
            ("general", general_ids),
            ("conversation", conversation_ids),
            ("instruction", instruction_ids),
        ],
        context_length=CONTEXT_LENGTH,
        sample_count=args.samples,
        warmup_ratio=args.warmup_ratio,
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

    print()
    print("Device           :", device)
    if device.type == "cuda":
        print("GPU              :", torch.cuda.get_device_name(0))
    print("Tokenizer        :", "byte-level BPE")
    print("Vocabulary       :", tokenizer.vocab_size)
    print("Chars/token      :", f"{char_count / max(1, token_count):.3f}")
    print("Parameters       :", f"{model.parameter_count:,}")
    print("Context length   :", CONTEXT_LENGTH)
    print("d_model          :", D_MODEL)
    print("Layers           :", NUM_LAYERS)
    print("Attention heads  :", NUM_HEADS)
    print("FFN dimension    :", HIDDEN_DIM)
    print("Phase A mixture  :", "90% general / 7% conversation / 3% instruction")
    print("Phase B mixture  :", "70% general / 20% conversation / 10% instruction")
    print("Samples/epoch    :", f"{args.samples:,}")
    print("Epochs           :", args.epochs)
    print("Batch size       :", args.batch_size)
    print("Learning rate    :", args.learning_rate)
    print("Token exposures  :", f"{args.samples * CONTEXT_LENGTH * args.epochs:,}")
    print("Tokenizer reused :", args.tokenizer)
    print()

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
    )
    history = trainer.train(loader=loader, epochs=args.epochs)
    final_loss = history[-1] if history else None

    model.save_checkpoint(
        args.output,
        optimizer=trainer.optimizer,
        epoch=args.epochs,
        loss=final_loss,
    )

    print()
    print("v0.8 clean-prior BPE pretraining completed.")
    print("Loss history     :", history)
    print("Model saved      :", args.output)
    print()
    print("Next:")
    print("  python train_sft_v08.py --base-model " + args.output)


if __name__ == "__main__":
    main()
