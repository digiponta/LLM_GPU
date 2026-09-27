# train_sft_v08.py
#
# LLM_GPU v0.8 multi-task SFT:
#   1) assistant answer next-token loss
#   2) prompt intent classification auxiliary loss
# plus deterministic intent-aware paraphrase augmentation.

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from augment_sft_v07 import augment_pairs, classify_intent
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-pretrain.pt"
DEFAULT_OUTPUT_MODEL = "model/model-gpu-v0.8-chat.pt"
DEFAULT_INTENT_HEAD = "model/model-gpu-v0.8-intent-head.pt"
DEFAULT_DATA = "data/conversation-ja.txt"
DEFAULT_INSTRUCTION_DATA = "data/instruction-ja.txt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "
SEED = 42
REQUIRE_CUDA = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-task augmented SFT for LLM_GPU v0.8."
    )
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--instruction-data", default=DEFAULT_INSTRUCTION_DATA)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--output", default=DEFAULT_OUTPUT_MODEL)
    p.add_argument("--intent-head-output", default=DEFAULT_INTENT_HEAD)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=8e-6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--intent-loss-weight", type=float, default=0.25)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument(
        "--technical-repeat",
        type=int,
        default=1,
        help=(
            "Training-only repeat factor for rows carrying technical concept "
            "tags. Default 1 disables broad oversampling so v0.8 reverse-"
            "definition binding can be evaluated independently."
        ),
    )
    p.add_argument(
        "--replay-repeat",
        type=int,
        default=2,
        help=(
            "Training-only replay factor for protected nontechnical intents "
            "(debug/error, repeat, topic). Default 2 adds one replay copy."
        ),
    )
    return p.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if REQUIRE_CUDA:
        raise RuntimeError("CUDA is not available. Run python check_gpu.py.")
    return torch.device("cpu")


def parse_dialogues(text: str) -> List[Tuple[str, str]]:
    pairs = []
    pending = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
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
    for raw_line in text.splitlines():
        line = raw_line.strip()
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


def deduplicate_pairs(
    pairs: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    seen = set()
    output = []
    for pair in pairs:
        key = (pair[0].strip(), pair[1].strip())
        if key not in seen:
            seen.add(key)
            output.append(key)
    return output


def stratified_split(
    rows: Sequence[Tuple[str, str, Tuple[str, ...]]],
    validation_ratio: float,
    seed: int,
):
    groups: Dict[str, List[Tuple[str, str, Tuple[str, ...]]]] = {}
    for row in rows:
        groups.setdefault(classify_intent(row[0]), []).append(row)

    rng = random.Random(seed)
    train_rows = []
    val_rows = []

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


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Tuple[str, str, Tuple[str, ...]]],
        tokenizer: Tokenizer,
        context_length: int,
        label_to_id: Dict[str, int],
    ):
        self.rows = []

        for user_text, answer_text, tags in rows:
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
            lm_mask = [
                1.0 if (i + 1) >= answer_start else 0.0
                for i in range(len(y))
            ]

            # Last prompt token in x is the representation used for intent.
            prompt_index = min(max(0, answer_start - 1), len(x) - 1)

            pad_count = context_length - len(x)
            if pad_count < 0:
                raise RuntimeError("SFT row exceeded context length.")
            x += [tokenizer.pad_id] * pad_count
            y += [tokenizer.pad_id] * pad_count
            lm_mask += [0.0] * pad_count

            self.rows.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(lm_mask, dtype=torch.float32),
                torch.tensor(prompt_index, dtype=torch.long),
                torch.tensor(
                    [
                        1.0 if label in tags else 0.0
                        for label in label_to_id
                    ],
                    dtype=torch.float32,
                ),
            ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def forward_losses(
    model,
    intent_head,
    input_ids,
    targets,
    lm_mask,
    prompt_index,
    intent_targets,
    label_smoothing,
    intent_loss_weight,
    pos_weight,
):
    hidden = model.forward_hidden(input_ids)
    logits = model.lm_head(hidden)

    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(targets)
    lm_loss = (token_losses * lm_mask).sum() / lm_mask.sum().clamp_min(1.0)

    batch_index = torch.arange(hidden.size(0), device=hidden.device)
    prompt_repr = hidden[batch_index, prompt_index]
    intent_logits = intent_head(prompt_repr)
    intent_loss = F.binary_cross_entropy_with_logits(
        intent_logits,
        intent_targets,
        pos_weight=pos_weight,
    )
    intent_pred = (torch.sigmoid(intent_logits) >= 0.5).float()

    tp = (intent_pred * intent_targets).sum()
    fp = (intent_pred * (1.0 - intent_targets)).sum()
    fn = ((1.0 - intent_pred) * intent_targets).sum()
    micro_f1 = (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1.0)

    total_loss = lm_loss + intent_loss_weight * intent_loss
    return total_loss, lm_loss, intent_loss, micro_f1


@torch.no_grad()
def evaluate(
    model,
    intent_head,
    loader,
    device,
    label_smoothing,
    intent_loss_weight,
    pos_weight,
):
    model.eval()
    intent_head.eval()
    sums = [0.0, 0.0, 0.0, 0.0]
    batches = 0

    for input_ids, targets, mask, prompt_index, intent_targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        prompt_index = prompt_index.to(device)
        intent_targets = intent_targets.to(device)

        values = forward_losses(
            model,
            intent_head,
            input_ids,
            targets,
            mask,
            prompt_index,
            intent_targets,
            label_smoothing,
            intent_loss_weight,
            pos_weight,
        )
        for i, value in enumerate(values):
            sums[i] += float(value.item())
        batches += 1

    return tuple(value / max(1, batches) for value in sums)



def compute_pos_weight(
    rows: Sequence[Tuple[str, str, Tuple[str, ...]]],
    labels: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    counts = {label: 0 for label in labels}
    for _, _, tags in rows:
        for label in tags:
            if label in counts:
                counts[label] += 1

    total = max(1, len(rows))
    values = []
    for label in labels:
        pos = max(1, counts[label])
        neg = max(1, total - counts[label])
        # Cap extreme weights so very rare tags do not dominate LM learning.
        values.append(min(10.0, neg / pos))

    return torch.tensor(values, dtype=torch.float32, device=device)


TECHNICAL_TAGS = {
    "tech_gpu",
    "tech_cpu",
    "tech_llm",
    "tech_transformer",
    "tech_cuda",
    "tech_python",
}


def oversample_technical_rows(
    rows: Sequence[Tuple[str, str, Tuple[str, ...]]],
    repeat: int,
) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """Repeat technical rows in training only; keep validation untouched."""
    if repeat < 1:
        raise ValueError("--technical-repeat must be >= 1.")

    output = []
    for row in rows:
        output.append(row)
        if any(tag in TECHNICAL_TAGS for tag in row[2]):
            for _ in range(repeat - 1):
                output.append(row)
    return output



REPLAY_TAGS = {
    "debug_error",
    "control_repeat",
    "control_topic",
}


def oversample_replay_rows(
    rows: Sequence[Tuple[str, str, Tuple[str, ...]]],
    repeat: int,
) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """Replay protected nontechnical intents in training only."""
    if repeat < 1:
        raise ValueError("--replay-repeat must be >= 1.")

    output = []
    for row in rows:
        output.append(row)
        if any(tag in REPLAY_TAGS for tag in row[2]):
            for _ in range(repeat - 1):
                output.append(row)
    return output


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    print()
    print("====================================")
    print(" LLM_GPU v0.8 Augmented Multi-task SFT")
    print("====================================")
    print()

    for filename in (
        args.data,
        args.instruction_data,
        args.tokenizer,
        args.base_model,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"Required file not found: {filename}")

    device = select_device()
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.base_model,
        device=device,
    )

    conversation_pairs = parse_dialogues(
        Path(args.data).read_text(encoding="utf-8")
    )
    instruction_pairs = parse_instruction_pairs(
        Path(args.instruction_data).read_text(encoding="utf-8")
    )
    base_pairs = deduplicate_pairs(conversation_pairs + instruction_pairs)

    augmented_rows = augment_pairs(
        base_pairs,
        variants_per_intent=args.variants_per_intent,
    )
    labels = sorted({
        label
        for _, _, tags in augmented_rows
        for label in tags
    })
    label_to_id = {label: i for i, label in enumerate(labels)}

    train_rows, val_rows = stratified_split(
        augmented_rows,
        validation_ratio=args.validation_ratio,
        seed=SEED,
    )
    original_train_count = len(train_rows)
    train_rows = oversample_technical_rows(
        train_rows,
        repeat=args.technical_repeat,
    )
    technical_train_count = len(train_rows)
    train_rows = oversample_replay_rows(
        train_rows,
        repeat=args.replay_repeat,
    )
    pos_weight = compute_pos_weight(train_rows, labels, device)

    train_set = MultiTaskDataset(
        train_rows, tokenizer, model.context_length, label_to_id
    )
    val_set = MultiTaskDataset(
        val_rows, tokenizer, model.context_length, label_to_id
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    intent_head = nn.Sequential(
        nn.Linear(model.d_model, model.d_model),
        nn.GELU(),
        nn.Linear(model.d_model, len(labels)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(intent_head.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Base loss          :", checkpoint.get("loss"))
    print("Base unique pairs  :", len(base_pairs))
    print("Augmented pairs    :", len(augmented_rows))
    print("Intent tags        :", len(labels))
    print("Intent tag names   :", ", ".join(labels))
    print("Train rows (base)  :", original_train_count)
    print("After technical    :", technical_train_count)
    print("Train rows (final) :", len(train_rows))
    print("Technical repeat   :", args.technical_repeat)
    print("Replay repeat      :", args.replay_repeat)
    print("Replay tags        :", ", ".join(sorted(REPLAY_TAGS)))
    print("Validation rows    :", len(val_rows))
    print("Multi-label weight :", args.intent_loss_weight)
    print(
        "Positive weights   :",
        f"min={pos_weight.min().item():.2f}",
        f"max={pos_weight.max().item():.2f}",
    )
    print("Label smoothing    :", args.label_smoothing)
    print("Learning rate      :", args.learning_rate)
    print()

    best_val = float("inf")
    best_model_state = None
    best_head_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        intent_head.train()
        total = lm_total = intent_total = acc_total = 0.0
        batches = 0
        started = time.perf_counter()

        for input_ids, targets, mask, prompt_index, intent_targets in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            prompt_index = prompt_index.to(device, non_blocking=True)
            intent_targets = intent_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss, lm_loss, intent_loss, intent_acc = forward_losses(
                model,
                intent_head,
                input_ids,
                targets,
                mask,
                prompt_index,
                intent_targets,
                args.label_smoothing,
                args.intent_loss_weight,
                pos_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(intent_head.parameters()),
                1.0,
            )
            optimizer.step()

            total += float(loss.item())
            lm_total += float(lm_loss.item())
            intent_total += float(intent_loss.item())
            acc_total += float(intent_acc.item())
            batches += 1

        train_loss = total / max(1, batches)
        train_lm = lm_total / max(1, batches)
        train_intent = intent_total / max(1, batches)
        train_f1 = acc_total / max(1, batches)

        val_loss, val_lm, val_intent, val_f1 = evaluate(
            model,
            intent_head,
            val_loader,
            device,
            args.label_smoothing,
            args.intent_loss_weight,
            pos_weight,
        )
        elapsed = time.perf_counter() - started

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train={train_loss:.4f} lm={train_lm:.4f} "
            f"intent={train_intent:.4f} tag_f1={train_f1:.1%} "
            f"| val={val_loss:.4f} lm={val_lm:.4f} "
            f"intent={val_intent:.4f} tag_f1={val_f1:.1%} "
            f"| {elapsed:.2f}s"
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_head_state = {
                k: v.detach().cpu().clone()
                for k, v in intent_head.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_model_state is None or best_head_state is None:
        raise RuntimeError("No valid multi-task checkpoint produced.")

    model.load_state_dict(best_model_state)
    model.to(device)
    intent_head.load_state_dict(best_head_state)

    model.save_checkpoint(
        args.output,
        optimizer=None,
        epoch=best_epoch,
        loss=best_val,
    )

    Path(args.intent_head_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "labels": labels,
            "multi_label": True,
            "threshold": 0.5,
            "state_dict": intent_head.state_dict(),
            "d_model": model.d_model,
            "epoch": best_epoch,
            "loss": best_val,
            "intent_loss_weight": args.intent_loss_weight,
        },
        args.intent_head_output,
    )

    print()
    print("Multi-task SFT completed.")
    print("Best epoch        :", best_epoch)
    print("Best val total    :", f"{best_val:.6f}")
    print("Chat model saved  :", args.output)
    print("Intent head saved :", args.intent_head_output)
    print()
    print("Next:")
    print("  python evaluate_chat.py")
    print("  python evaluate_generalization_v07.py")


if __name__ == "__main__":
    main()
