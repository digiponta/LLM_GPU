# train_corpus.py
#
# Train the v0.4 homemade LLM on a physical CUDA GPU.

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TokenWindowDataset
from model import LanguageModel
from tokenizer import Tokenizer
from train import Trainer


TOKENIZER_FILE = "model/tokenizer.json"
MODEL_FILE = "model/model-gpu-v0.4.pt"

CONTEXT_LENGTH = 64
D_MODEL = 64
NUM_LAYERS = 2
HIDDEN_DIM = 256

BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 5e-4
MAX_SAMPLES = 120_000_000
SEED = 42

# GPU project: fail early by default if CUDA is unavailable.
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
    raise FileNotFoundError(
        f"Could not find {filename}. Searched:\n{searched}"
    )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if REQUIRE_CUDA:
        raise RuntimeError(
            "CUDA is not available to PyTorch. "
            "Run python check_gpu.py and install a CUDA-enabled PyTorch build."
        )
    return torch.device("cpu")


def main():
    torch.manual_seed(SEED)

    print()
    print("====================================")
    print(" Homemade LLM GPU Training v0.4")
    print("====================================")
    print()

    device = select_device()

    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    print("Device          :", device)

    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(
            "VRAM            :",
            f"{props.total_memory / (1024 ** 3):.2f} GiB",
        )
        print("CUDA runtime    :", torch.version.cuda)

    general_file = find_data_file("general-ja.txt")
    nagato_file = find_data_file("data-nagato.txt")

    print()
    print("Loading corpora...")
    print("general-ja.txt  :", general_file)
    print("data-nagato.txt :", nagato_file)

    general_text = general_file.read_text(encoding="utf-8")
    nagato_text = nagato_file.read_text(encoding="utf-8")
    training_text = general_text + "\n\n" + nagato_text

    print(
        "general-ja.txt  :",
        f"{len(general_text):,}",
        "characters",
    )
    print(
        "data-nagato.txt :",
        f"{len(nagato_text):,}",
        "characters",
    )
    print(
        "Total           :",
        f"{len(training_text):,}",
        "characters",
    )

    print()
    print("Building tokenizer...")

    tokenizer = Tokenizer()
    tokenizer.fit_texts([general_text, nagato_text])

    Path("model").mkdir(exist_ok=True)
    tokenizer.save(TOKENIZER_FILE)

    print("Vocabulary size :", tokenizer.vocab_size)
    print("Tokenizer saved :", TOKENIZER_FILE)

    print()
    print("Encoding corpus...")

    token_ids = tokenizer.encode(
        training_text,
        add_bos=True,
        add_eos=True,
    )

    print("Token count     :", f"{len(token_ids):,}")

    dataset = TokenWindowDataset(
        token_ids=token_ids,
        context_length=CONTEXT_LENGTH,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )

    print()
    print("Training configuration")
    print("----------------------")
    print("Context length  :", CONTEXT_LENGTH)
    print("Training samples:", f"{len(dataset):,}")
    print("Unique windows  :", f"{dataset.total_positions:,}")
    if len(dataset) > dataset.total_positions:
        repeats = len(dataset) / dataset.total_positions
        print("Corpus passes   :", f"{repeats:.2f} per epoch (approx.)")
    print("Target runtime  :", "about 10 hours (estimated from v0.3 benchmark)")
    print("Batch size      :", BATCH_SIZE)
    print("Epochs          :", EPOCHS)
    print("Learning rate   :", LEARNING_RATE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        # v0.4 dataset already traverses corpus windows in a deterministic
        # pseudo-random permutation. Avoid RandomSampler here because a
        # 120M-sample randperm would require enormous host memory.
        shuffle=False,
        num_workers=0,  # Windows-safe default.
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        causal=True,
        context_length=CONTEXT_LENGTH,
    ).to(device)

    print()
    print("Model")
    print("-----")
    print("Parameters      :", f"{model.parameter_count:,}")
    print(
        "Parameter memory:",
        f"{model.parameter_count * 4 / (1024 ** 2):.2f} MiB",
        "(FP32 only)",
    )

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
    )

    print()
    print("Training...")
    print()

    history = trainer.train(
        loader=loader,
        epochs=EPOCHS,
    )

    final_loss = history[-1] if history else None

    print()
    print("Training completed.")
    print("Loss history    :", history)

    model.save_checkpoint(
        MODEL_FILE,
        optimizer=trainer.optimizer,
        epoch=EPOCHS,
        loss=final_loss,
    )

    print("Model saved     :", MODEL_FILE)

    if device.type == "cuda":
        torch.cuda.synchronize()
        print(
            "Peak GPU memory :",
            f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MiB",
        )


if __name__ == "__main__":
    main()
