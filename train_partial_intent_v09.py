# train_partial_intent_v09.py
#
# LLM_GPU v0.9 Partial Fine-Tuning with Mid-Layer Intent Conditioning.
#
# Intent path:
#   frozen v0.8 model -> frozen intent head -> soft 24-d probabilities
#
# Generation path:
#   Blocks 1-3 frozen
#   inject projected intent after Block 3
#   Blocks 4-6 trainable
#   FinalNorm trainable
#   LM Head frozen
#
# Parameter groups:
#   projection lr : 1e-3
#   Blocks 4-6 + FinalNorm lr : 2e-6

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
    validate_injection_point,
)
from model import LanguageModel
from partial_intent_conditioning_v09 import (
    configure_partial_finetune,
    forward_partial_conditioned,
    infer_intent_prob,
    save_partial_checkpoint,
    trainable_model_parameters,
    trainable_parameter_count,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INTENT_HEAD = "model/model-gpu-v0.8-intent-head.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9-partial-intent.pt"
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
PROTECTED_BOUNDARY_REPLAY_TAGS = {"protected_boundary_replay"}
BALANCED_STABLE_REPLAY_TAGS = {"balanced_stable_replay"}
BALANCED_CONTROL_REPLAY_TAGS = {"balanced_control_replay"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train v0.9 partial fine-tuned mid-layer intent model."
    )
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--instruction-data", default=DEFAULT_INSTRUCTION_DATA)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT_HEAD)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--variants-per-intent", type=int, default=24)
    p.add_argument("--technical-repeat", type=int, default=1)
    p.add_argument("--replay-repeat", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--inject-after", type=int, default=3)
    p.add_argument("--projection-lr", type=float, default=1e-3)
    p.add_argument("--block-lr", type=float, default=2e-6)
    p.add_argument(
        "--targeted-boundary",
        action="store_true",
        help="Include v0.9 targeted boundary v1 rows in SFT augmentation.",
    )
    p.add_argument(
        "--targeted-boundary-v2",
        action="store_true",
        help="Include v0.9 targeted boundary v2 rows in SFT augmentation.",
    )
    p.add_argument(
        "--protected-boundary-replay",
        action="store_true",
        help="Include protected replay rows for Boundary v3.",
    )
    p.add_argument(
        "--protected-replay-repeat",
        type=int,
        default=3,
        help="Training repeat factor for protected Boundary v3 rows.",
    )
    p.add_argument(
        "--balanced-control-replay",
        action="store_true",
        help="Include Boundary v4 balanced stable/control replay rows.",
    )
    p.add_argument(
        "--stable-replay-repeat",
        type=int,
        default=2,
        help="Replay factor for stable protected capabilities in Boundary v4.",
    )
    p.add_argument(
        "--control-replay-repeat",
        type=int,
        default=3,
        help="Replay factor for short/repeat control rows in Boundary v4.",
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


class PartialIntentDataset(Dataset):
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
    generation_model,
    intent_model,
    intent_head,
    projection,
    input_ids,
    targets,
    lm_mask,
    prompt_index,
    label_smoothing,
):
    with torch.no_grad():
        intent_prob = infer_intent_prob(
            intent_model,
            intent_head,
            input_ids,
            prompt_index,
        )

    intent_bias = projection(intent_prob)

    hidden = forward_partial_conditioned(
        generation_model,
        input_ids,
        intent_bias,
        projection.inject_after,
    )
    logits = generation_model.lm_head(hidden)

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(targets)

    return (token_loss * lm_mask).sum() / lm_mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    generation_model,
    intent_model,
    intent_head,
    projection,
    loader,
    device,
    label_smoothing,
):
    generation_model.eval()
    projection.eval()

    total = 0.0
    batches = 0

    for input_ids, targets, mask, prompt_index in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        prompt_index = prompt_index.to(device)

        loss = conditioned_loss(
            generation_model,
            intent_model,
            intent_head,
            projection,
            input_ids,
            targets,
            mask,
            prompt_index,
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
        args.data,
        args.instruction_data,
        args.tokenizer,
        args.base_model,
        args.intent_head,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)

    # Separate models intentionally keep the intent representation frozen.
    intent_model, base_checkpoint = LanguageModel.load_checkpoint(
        args.base_model,
        device=device,
    )
    generation_model, _ = LanguageModel.load_checkpoint(
        args.base_model,
        device=device,
    )

    intent_head, head_checkpoint, labels = load_intent_head(
        args.intent_head,
        intent_model,
        device,
    )

    validate_injection_point(generation_model, args.inject_after)

    intent_model.eval()
    for parameter in intent_model.parameters():
        parameter.requires_grad_(False)
    intent_head.eval()
    for parameter in intent_head.parameters():
        parameter.requires_grad_(False)

    configure_partial_finetune(
        generation_model,
        inject_after=args.inject_after,
    )

    projection = MidLayerIntentProjection(
        num_labels=len(labels),
        d_model=generation_model.d_model,
        alpha=args.alpha,
        inject_after=args.inject_after,
    ).to(device)

    conversation = parse_dialogues(Path(args.data).read_text(encoding="utf-8"))
    instruction = parse_instruction_pairs(
        Path(args.instruction_data).read_text(encoding="utf-8")
    )
    base_pairs = deduplicate_pairs(conversation + instruction)
    rows = augment_pairs(
        base_pairs,
        variants_per_intent=args.variants_per_intent,
        include_targeted_boundary=args.targeted_boundary,
        include_targeted_boundary_v2=args.targeted_boundary_v2,
        include_protected_boundary_replay=args.protected_boundary_replay,
        include_balanced_control_replay=args.balanced_control_replay,
    )

    train_rows, val_rows = stratified_split(
        rows,
        args.validation_ratio,
        SEED,
    )
    base_train_count = len(train_rows)
    train_rows = oversample(
        train_rows,
        args.technical_repeat,
        TECHNICAL_TAGS,
    )
    train_rows = oversample(
        train_rows,
        args.replay_repeat,
        REPLAY_TAGS,
    )
    if args.protected_boundary_replay:
        train_rows = oversample(
            train_rows,
            args.protected_replay_repeat,
            PROTECTED_BOUNDARY_REPLAY_TAGS,
        )
    if args.balanced_control_replay:
        train_rows = oversample(
            train_rows,
            args.stable_replay_repeat,
            BALANCED_STABLE_REPLAY_TAGS,
        )
        train_rows = oversample(
            train_rows,
            args.control_replay_repeat,
            BALANCED_CONTROL_REPLAY_TAGS,
        )

    train_set = PartialIntentDataset(
        train_rows,
        tokenizer,
        generation_model.context_length,
    )
    val_set = PartialIntentDataset(
        val_rows,
        tokenizer,
        generation_model.context_length,
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

    model_trainable = trainable_model_parameters(generation_model)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": projection.parameters(),
                "lr": args.projection_lr,
            },
            {
                "params": model_trainable,
                "lr": args.block_lr,
            },
        ],
        weight_decay=0.01,
    )

    print()
    print("============================================")
    print(" LLM_GPU v0.9 Partial Intent Fine-Tuning")
    print("============================================")
    print("Device                :", device)
    if device.type == "cuda":
        print("GPU                   :", torch.cuda.get_device_name(0))
    print("Base model loss       :", base_checkpoint.get("loss"))
    print("Intent-head loss      :", head_checkpoint.get("loss"))
    print("Intent labels         :", len(labels))
    print("Inject after block    :", args.inject_after)
    print("Frozen blocks         :", f"1-{args.inject_after}")
    print(
        "Trainable blocks      :",
        f"{args.inject_after + 1}-{generation_model.num_layers}",
    )
    print("FinalNorm trainable   : True")
    print("LM Head trainable     : False")
    print("Projection alpha      :", args.alpha)
    print("Projection LR         :", args.projection_lr)
    print("Block/FinalNorm LR    :", args.block_lr)
    print("Targeted boundary v1  :", args.targeted_boundary)
    print("Targeted boundary v2  :", args.targeted_boundary_v2)
    print("Protected replay      :", args.protected_boundary_replay)
    print("Protected repeat      :", args.protected_replay_repeat)
    print("Balanced control      :", args.balanced_control_replay)
    print("Stable replay repeat  :", args.stable_replay_repeat)
    print("Control replay repeat :", args.control_replay_repeat)
    print(
        "Projection params     :",
        sum(p.numel() for p in projection.parameters()),
    )
    print(
        "Model trainable params:",
        trainable_parameter_count(generation_model),
    )
    print("Train rows (base)     :", base_train_count)
    print("Train rows (final)    :", len(train_rows))
    print("Validation rows       :", len(val_rows))
    print()

    best_val = float("inf")
    best_model_state = None
    best_projection_state = None
    best_epoch = 0
    bad_epochs = 0

    trainable_for_clip = list(projection.parameters()) + model_trainable

    for epoch in range(1, args.epochs + 1):
        generation_model.train()
        # Frozen early blocks contain no dropout, but keep intent model/head eval.
        intent_model.eval()
        intent_head.eval()
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
                generation_model,
                intent_model,
                intent_head,
                projection,
                input_ids,
                targets,
                mask,
                prompt_index,
                args.label_smoothing,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_for_clip,
                1.0,
            )
            optimizer.step()

            total += float(loss.item())
            batches += 1

        train_loss = total / max(1, batches)
        val_loss = evaluate(
            generation_model,
            intent_model,
            intent_head,
            projection,
            val_loader,
            device,
            args.label_smoothing,
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
            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in generation_model.state_dict().items()
            }
            best_projection_state = {
                k: v.detach().cpu().clone()
                for k, v in projection.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_model_state is None or best_projection_state is None:
        raise RuntimeError("No valid partial fine-tuned checkpoint produced.")

    generation_model.load_state_dict(best_model_state)
    projection.load_state_dict(best_projection_state)

    save_partial_checkpoint(
        args.output,
        generation_model,
        projection,
        labels,
        epoch=best_epoch,
        loss=best_val,
        base_model=args.base_model,
        intent_head=args.intent_head,
        block_learning_rate=args.block_lr,
        projection_learning_rate=args.projection_lr,
    )

    print()
    print("Partial fine-tuning completed.")
    print("Best epoch       :", best_epoch)
    print("Best val LM loss :", f"{best_val:.6f}")
    print("Checkpoint saved :", args.output)
    print()
    print("Next:")
    print("  python evaluate_partial_intent_v09.py")


if __name__ == "__main__":
    main()
