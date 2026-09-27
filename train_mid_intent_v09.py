# train_mid_intent_v09.py
#
# Projection-only Mid-Layer Intent Conditioning for LLM_GPU v0.9.
#
# Clean ablation:
#   - frozen v0.8 pairwise-best LM
#   - frozen v0.8 intent head
#   - train only 24 -> 256 projection
#   - inject after Transformer Block 3 by default
#
# Unlike final-layer conditioning, Blocks 4-6 can transform the injected
# semantic signal before the LM head.

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from augment_sft_v07 import augment_pairs, classify_intent
from intent_conditioning_v09 import load_intent_head
from mid_intent_conditioning_v09 import (
    MidLayerIntentProjection,
    forward_mid_conditioned,
    validate_injection_point,
)
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INTENT_HEAD = "model/model-gpu-v0.8-intent-head.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9-mid-intent.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION_DATA = "data/instruction-ja.txt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "
SEED = 42

REPLAY_TAGS = {"debug_error", "control_repeat", "control_topic"}
TECHNICAL_TAGS = {
    "tech_gpu", "tech_cpu", "tech_llm",
    "tech_transformer", "tech_cuda", "tech_python",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train v0.9 projection-only mid-layer intent conditioning."
    )
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--instruction-data", default=DEFAULT_INSTRUCTION_DATA)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT_HEAD)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--technical-repeat", type=int, default=1)
    p.add_argument("--replay-repeat", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument(
        "--inject-after",
        type=int,
        default=3,
        help="1-based Transformer block after which intent is injected.",
    )
    return p.parse_args()


def parse_dialogues(text: str) -> List[Tuple[str, str]]:
    pairs = []
    pending = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(USER_PREFIX):
            pending = line[len(USER_PREFIX):].strip()
        elif line.startswith(AI_PREFIX) and pending is not None:
            answer = line[len(AI_PREFIX):].strip()
            if pending and answer:
                pairs.append((pending, answer))
            pending = None
    return pairs


def parse_instruction_pairs(text: str) -> List[Tuple[str, str]]:
    pairs = []
    pending = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("質問:"):
            pending = line[len("質問:"):].strip()
        elif line.startswith("指示:"):
            pending = line[len("指示:"):].strip()
        elif line.startswith("回答:") and pending is not None:
            answer = line[len("回答:"):].strip()
            if pending and answer:
                pairs.append((pending, answer))
            pending = None
    return pairs


def deduplicate_pairs(pairs):
    seen = set()
    out = []
    for prompt, answer in pairs:
        key = (prompt.strip(), answer.strip())
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def stratified_split(rows, validation_ratio: float, seed: int):
    groups = {}
    for row in rows:
        groups.setdefault(classify_intent(row[0]), []).append(row)

    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for label in sorted(groups):
        items = list(groups[label])
        rng.shuffle(items)
        if len(items) >= 5:
            count = max(1, int(round(len(items) * validation_ratio)))
            count = min(count, len(items) - 3)
        elif len(items) >= 3:
            count = 1
        else:
            count = 0
        val_rows.extend(items[:count])
        train_rows.extend(items[count:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    if not val_rows:
        raise RuntimeError("Validation split is empty.")
    return train_rows, val_rows


def oversample(rows, repeat: int, tags):
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    out = []
    for row in rows:
        out.append(row)
        if any(tag in tags for tag in row[2]):
            out.extend([row] * (repeat - 1))
    return out


class MidIntentDataset(Dataset):
    def __init__(self, rows, tokenizer, context_length):
        self.rows = []
        for user_text, answer_text, _tags in rows:
            prompt = f"{USER_PREFIX}{user_text}\n{AI_PREFIX}"
            prompt_ids = tokenizer.encode(prompt, add_bos=True)
            answer_ids = tokenizer.encode(answer_text, add_eos=True)

            max_sequence = context_length + 1
            if len(prompt_ids) + len(answer_ids) > max_sequence:
                keep_prompt = max(1, max_sequence - len(answer_ids))
                prompt_ids = prompt_ids[-keep_prompt:]
            if len(prompt_ids) + len(answer_ids) > max_sequence:
                room = max_sequence - len(prompt_ids)
                answer_ids = answer_ids[:room]
                if answer_ids:
                    answer_ids[-1] = tokenizer.eos_id

            sequence = prompt_ids + answer_ids
            answer_start = len(prompt_ids)
            x = sequence[:-1]
            y = sequence[1:]
            mask = [
                1.0 if (i + 1) >= answer_start else 0.0
                for i in range(len(y))
            ]
            prompt_index = min(max(0, answer_start - 1), len(x) - 1)

            pad = context_length - len(x)
            if pad < 0:
                raise RuntimeError("SFT row exceeded context length.")
            x += [tokenizer.pad_id] * pad
            y += [tokenizer.pad_id] * pad
            mask += [0.0] * pad

            self.rows.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(mask, dtype=torch.float32),
                torch.tensor(prompt_index, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def conditioned_loss(
    model,
    intent_head,
    projection,
    input_ids,
    targets,
    lm_mask,
    prompt_index,
    label_smoothing,
):
    # Infer intent from the original unconditioned prompt representation.
    with torch.no_grad():
        base_hidden = model.forward_hidden(input_ids)
        batch = torch.arange(base_hidden.size(0), device=base_hidden.device)
        prompt_repr = base_hidden[batch, prompt_index]
        intent_prob = torch.sigmoid(intent_head(prompt_repr))

    intent_bias = projection(intent_prob)

    # Backbone parameters are frozen, but the later blocks stay in the graph so
    # gradients can flow from LM loss back to intent_bias/projection.
    hidden = forward_mid_conditioned(
        model,
        input_ids,
        intent_bias,
        projection.inject_after,
    )
    logits = model.lm_head(hidden)

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(targets)
    return (token_loss * lm_mask).sum() / lm_mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model,
    intent_head,
    projection,
    loader,
    device,
    label_smoothing,
):
    projection.eval()
    total = 0.0
    batches = 0
    for input_ids, targets, mask, prompt_index in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        prompt_index = prompt_index.to(device)
        loss = conditioned_loss(
            model, intent_head, projection,
            input_ids, targets, mask, prompt_index,
            label_smoothing,
        )
        total += float(loss.item())
        batches += 1
    return total / max(1, batches)


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (
        args.data, args.instruction_data, args.tokenizer,
        args.base_model, args.intent_head,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, model_checkpoint = LanguageModel.load_checkpoint(
        args.base_model, device=device
    )
    intent_head, head_checkpoint, labels = load_intent_head(
        args.intent_head, model, device
    )
    validate_injection_point(model, args.inject_after)

    model.eval()
    intent_head.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in intent_head.parameters():
        parameter.requires_grad_(False)

    projection = MidLayerIntentProjection(
        len(labels),
        model.d_model,
        alpha=args.alpha,
        inject_after=args.inject_after,
    ).to(device)

    conversation = parse_dialogues(Path(args.data).read_text(encoding="utf-8"))
    instruction = parse_instruction_pairs(
        Path(args.instruction_data).read_text(encoding="utf-8")
    )
    base_pairs = deduplicate_pairs(conversation + instruction)
    rows = augment_pairs(base_pairs, variants_per_intent=args.variants_per_intent)

    train_rows, val_rows = stratified_split(rows, args.validation_ratio, SEED)
    base_train_count = len(train_rows)
    train_rows = oversample(train_rows, args.technical_repeat, TECHNICAL_TAGS)
    train_rows = oversample(train_rows, args.replay_repeat, REPLAY_TAGS)

    train_set = MidIntentDataset(train_rows, tokenizer, model.context_length)
    val_set = MidIntentDataset(val_rows, tokenizer, model.context_length)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        projection.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    print()
    print("==========================================")
    print(" LLM_GPU v0.9 Mid-Layer Intent SFT")
    print("==========================================")
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Base model         :", args.base_model)
    print("Base model loss    :", model_checkpoint.get("loss"))
    print("Intent-head loss   :", head_checkpoint.get("loss"))
    print("Intent labels      :", len(labels))
    print("d_model            :", model.d_model)
    print("Inject after block :", args.inject_after)
    print("Blocks after inject:", model.num_layers - args.inject_after)
    print("Alpha              :", args.alpha)
    print("Trainable params   :", sum(p.numel() for p in projection.parameters()))
    print("Backbone frozen    : True")
    print("Intent head frozen : True")
    print("Train rows (base)  :", base_train_count)
    print("Train rows (final) :", len(train_rows))
    print("Validation rows    :", len(val_rows))
    print()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        projection.train()
        total = 0.0
        batches = 0
        started = time.perf_counter()

        for input_ids, targets, mask, prompt_index in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            prompt_index = prompt_index.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = conditioned_loss(
                model, intent_head, projection,
                input_ids, targets, mask, prompt_index,
                args.label_smoothing,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            optimizer.step()

            total += float(loss.item())
            batches += 1

        train_loss = total / max(1, batches)
        val_loss = evaluate(
            model, intent_head, projection,
            val_loader, device, args.label_smoothing,
        )
        elapsed = time.perf_counter() - started

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train={train_loss:.4f} val={val_loss:.4f} "
            f"| {elapsed:.2f}s"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in projection.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("No valid mid-layer projection produced.")

    projection.load_state_dict(best_state)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "llm-gpu-v0.9-mid-intent",
            "labels": labels,
            "d_model": model.d_model,
            "alpha": args.alpha,
            "inject_after": args.inject_after,
            "state_dict": projection.state_dict(),
            "epoch": best_epoch,
            "loss": best_val,
            "base_model": args.base_model,
            "intent_head": args.intent_head,
        },
        args.output,
    )

    print()
    print("Mid-layer intent projection completed.")
    print("Best epoch        :", best_epoch)
    print("Best val LM loss  :", f"{best_val:.6f}")
    print("Projection saved  :", args.output)
    print()
    print("Next:")
    print("  python evaluate_mid_intent_v09.py")


if __name__ == "__main__":
    main()
