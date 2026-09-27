# train_conversation.py
#
# LLM_GPU v0.5 conversational supervised fine-tuning (SFT).
#
# Important difference from the first v0.5 experiment:
# - one dialogue pair is one training sample
# - loss is calculated only on the AI answer
# - prompt/padding tokens are masked from the loss
# - a validation split and early stopping reduce memorization

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_OUTPUT_MODEL = "model/model-gpu-v0.5-chat.pt"
DEFAULT_DATA = "data/conversation-ja.txt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "
SEED = 42
REQUIRE_CUDA = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning for short Japanese conversation."
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=6)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if REQUIRE_CUDA:
        raise RuntimeError(
            "CUDA is not available to PyTorch. Run python check_gpu.py first."
        )
    return torch.device("cpu")


def parse_dialogues(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pending_user = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith(USER_PREFIX):
            pending_user = line[len(USER_PREFIX):].strip()
            continue

        if line.startswith(AI_PREFIX) and pending_user is not None:
            answer = line[len(AI_PREFIX):].strip()
            if pending_user and answer:
                pairs.append((pending_user, answer))
            pending_user = None

    if not pairs:
        raise ValueError("No dialogue pairs found in conversation corpus.")
    return pairs


class ConversationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        tokenizer: Tokenizer,
        context_length: int,
    ):
        self.rows = []
        self.context_length = context_length
        self.pad_id = tokenizer.pad_id

        for user_text, answer_text in pairs:
            prompt = f"{USER_PREFIX}{user_text}\n{AI_PREFIX}"
            prompt_ids = tokenizer.encode(prompt, add_bos=True)
            answer_ids = tokenizer.encode(answer_text, add_eos=True)

            # Need context_length + 1 tokens for shifted input/target.
            max_sequence = context_length + 1

            # Preserve the answer. Trim old prompt characters from the left first.
            if len(prompt_ids) + len(answer_ids) > max_sequence:
                keep_prompt = max(1, max_sequence - len(answer_ids))
                prompt_ids = prompt_ids[-keep_prompt:]

            # If the answer itself is too long, keep its beginning plus EOS.
            if len(prompt_ids) + len(answer_ids) > max_sequence:
                answer_room = max_sequence - len(prompt_ids)
                if answer_room <= 1:
                    prompt_ids = prompt_ids[-1:]
                    answer_room = max_sequence - 1
                answer_ids = answer_ids[:answer_room]
                if answer_ids:
                    answer_ids[-1] = tokenizer.eos_id

            sequence = prompt_ids + answer_ids
            answer_start = len(prompt_ids)

            x = sequence[:-1]
            y = sequence[1:]

            # y[i] predicts sequence[i+1]. Score only AI answer/EOS targets.
            loss_mask = [
                1.0 if (i + 1) >= answer_start else 0.0
                for i in range(len(y))
            ]

            pad_count = context_length - len(x)
            if pad_count < 0:
                raise RuntimeError("Conversation sample exceeded context length.")

            x = x + [tokenizer.pad_id] * pad_count
            y = y + [tokenizer.pad_id] * pad_count
            loss_mask = loss_mask + [0.0] * pad_count

            self.rows.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(loss_mask, dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def masked_loss(
    model: LanguageModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)

    weighted = losses * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model: LanguageModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0

    for input_ids, targets, mask in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        loss = masked_loss(model, input_ids, targets, mask)
        total_loss += float(loss.item())
        batches += 1

    return total_loss / max(1, batches)


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    print()
    print("============================================")
    print(" LLM_GPU v0.5 Conversational SFT")
    print("============================================")
    print()

    data_path = Path(args.data)
    tokenizer_path = Path(args.tokenizer)
    base_path = Path(args.base_model)

    for path in (data_path, tokenizer_path, base_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    device = select_device()
    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(base_path),
        device=device,
    )

    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    raw_text = data_path.read_text(encoding="utf-8")
    pairs = parse_dialogues(raw_text)

    # Report unknown-character rate before training.
    corpus_text = "\n".join(
        f"{USER_PREFIX}{u}\n{AI_PREFIX}{a}" for u, a in pairs
    )
    corpus_ids = tokenizer.encode(corpus_text)
    unknown_count = sum(x == tokenizer.unk_id for x in corpus_ids)
    unknown_rate = unknown_count / max(1, len(corpus_ids))

    shuffled = list(pairs)
    random.shuffle(shuffled)

    validation_count = max(
        1,
        int(round(len(shuffled) * args.validation_ratio)),
    )
    validation_count = min(validation_count, len(shuffled) - 1)

    val_pairs = shuffled[:validation_count]
    train_pairs = shuffled[validation_count:]

    train_dataset = ConversationDataset(
        train_pairs, tokenizer, model.context_length
    )
    val_dataset = ConversationDataset(
        val_pairs, tokenizer, model.context_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Base checkpoint :", args.base_model)
    print("Base loss       :", checkpoint.get("loss"))
    print("Dialogue pairs  :", len(pairs))
    print("Train pairs     :", len(train_pairs))
    print("Validation pairs:", len(val_pairs))
    print("Unknown tokens  :", f"{unknown_count} ({unknown_rate:.3%})")
    print("Context length  :", model.context_length)
    print("Parameters      :", f"{model.parameter_count:,}")
    print("Epoch limit     :", args.epochs)
    print("Learning rate   :", args.learning_rate)
    print("Batch size      :", args.batch_size)
    print("Early stopping  :", f"patience={args.patience}")
    print()

    if unknown_rate > 0.02:
        print(
            "WARNING: >2% <UNK> tokens. The base tokenizer may be too limited."
        )
        print()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        started = time.perf_counter()

        for input_ids, targets, mask in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model, input_ids, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            batches += 1

        train_loss = epoch_loss / max(1, batches)
        val_loss = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - started

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train {train_loss:.4f} "
            f"| val {val_loss:.4f} "
            f"| ppl {math.exp(min(val_loss, 20.0)):.2f} "
            f"| {elapsed:.2f}s"
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"Early stopping: validation loss did not improve "
                    f"for {args.patience} epochs."
                )
                break

    if best_state is None:
        raise RuntimeError("No valid training checkpoint was produced.")

    model.load_state_dict(best_state)
    model.to(device)

    model.save_checkpoint(
        args.output,
        optimizer=optimizer,
        epoch=best_epoch,
        loss=best_val,
    )

    print()
    print("Training completed.")
    print("Best epoch      :", best_epoch)
    print("Best val loss   :", f"{best_val:.6f}")
    print("Model saved     :", args.output)
    print()
    print("Run:")
    print("  python chat.py")
    print("  python evaluate_chat.py")


if __name__ == "__main__":
    main()
