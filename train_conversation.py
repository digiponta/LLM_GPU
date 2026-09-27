# train_conversation.py
#
# LLM_GPU v0.5 conversational fine-tuning.
# This keeps the v0.4 tokenizer and model architecture unchanged and
# fine-tunes the pretrained checkpoint on short Japanese dialogue.

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TokenWindowDataset
from model import LanguageModel
from tokenizer import Tokenizer
from train import Trainer


DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_OUTPUT_MODEL = "model/model-gpu-v0.5-chat.pt"
DEFAULT_DATA = "data/conversation-ja.txt"

BATCH_SIZE = 64
SEED = 42
REQUIRE_CUDA = True


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if REQUIRE_CUDA:
        raise RuntimeError(
            "CUDA is not available to PyTorch. "
            "Run python check_gpu.py and install a CUDA-enabled PyTorch build."
        )
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM_GPU v0.4 for short Japanese conversation."
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)

    print()
    print("============================================")
    print(" LLM_GPU v0.5 Conversational Fine-Tuning")
    print("============================================")
    print()

    data_path = Path(args.data)
    tokenizer_path = Path(args.tokenizer)
    base_model_path = Path(args.base_model)

    for path in (data_path, tokenizer_path, base_model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    device = select_device()
    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(base_model_path),
        device=device,
    )

    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    text = data_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    unknown_count = sum(token_id == tokenizer.unk_id for token_id in token_ids)
    unknown_rate = unknown_count / max(1, len(token_ids))

    print()
    print("Base checkpoint :", base_model_path)
    print("Base loss       :", checkpoint.get("loss"))
    print("Conversation    :", data_path)
    print("Characters      :", f"{len(text):,}")
    print("Tokens          :", f"{len(token_ids):,}")
    print("Unknown tokens  :", f"{unknown_count:,} ({unknown_rate:.3%})")
    print("Vocabulary size :", tokenizer.vocab_size)
    print("Context length  :", model.context_length)

    if unknown_rate > 0.02:
        print(
            "WARNING: More than 2% of conversation tokens are <UNK>. "
            "Consider extending the pretraining corpus and rebuilding the "
            "base tokenizer/model before conversational fine-tuning."
        )

    dataset = TokenWindowDataset(
        token_ids=token_ids,
        context_length=model.context_length,
        max_samples=args.max_samples,
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

    print()
    print("Fine-tuning configuration")
    print("-------------------------")
    print("Samples         :", f"{len(dataset):,}")
    print("Unique windows  :", f"{dataset.total_positions:,}")
    print("Batch size      :", args.batch_size)
    print("Epochs          :", args.epochs)
    print("Learning rate   :", args.learning_rate)
    print("Parameters      :", f"{model.parameter_count:,}")

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
    )

    print()
    history = trainer.train(loader=loader, epochs=args.epochs)
    final_loss = history[-1] if history else None

    model.save_checkpoint(
        args.output,
        optimizer=trainer.optimizer,
        epoch=args.epochs,
        loss=final_loss,
    )

    print()
    print("Conversational fine-tuning completed.")
    print("Loss history    :", history)
    print("Model saved     :", args.output)
    print()
    print("Next:")
    print("  python chat.py")
    print("  python evaluate_chat.py")


if __name__ == "__main__":
    main()
