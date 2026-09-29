# online_train.py
#
# LLM_GPU v1.6.2 deduplicated incremental-only conversational learning.
#
# - Reads accepted/manual dialogue pairs saved by chat.py as JSONL.
# - Replays the existing conversation corpus to reduce catastrophic forgetting.
# - Computes loss only on the AI answer.
# - Uses a low learning rate and validation/early stopping.
# - Writes a new checkpoint; it never overwrites the base checkpoint.

from __future__ import annotations

import argparse
import json
import hashlib
import math
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import LanguageModel
from tokenizer_bpe import Tokenizer


USER_PREFIX = "人: "
AI_PREFIX = "AI: "
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forgetting-aware incremental conversational training for LLM_GPU v1.6.19."
    )
    p.add_argument("--chat-data", default="data/chat_history.jsonl")
    p.add_argument("--replay-data", default="data/conversation-ja.txt")
    p.add_argument("--tokenizer", default="model/tokenizer-v0.7-bpe.json")
    p.add_argument("--base-model", default="model/model-gpu-v0.8-chat-clean.pt")
    p.add_argument("--output", default="model/model-gpu-v1.6.2-online.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument(
        "--replay-ratio",
        type=float,
        default=1.0,
        help="Replay pairs per new chat pair.",
    )
    p.add_argument("--manual-weight", type=int, default=4)
    p.add_argument("--recovery-weight", type=int, default=8)
    p.add_argument("--recovery-stabilization-epochs", type=int, default=6)
    p.add_argument("--recovery-learning-rate", type=float, default=1e-5)
    p.add_argument("--auto-weight", type=int, default=0)
    p.add_argument(
        "--state",
        default="data/chat_learning_state.json",
        help="Persistent fingerprints of trusted pairs already consumed by training.",
    )
    p.add_argument(
        "--trusted-replay-weight",
        type=int,
        default=1,
        help="Weak replay weight for previously trained trusted pairs.",
    )
    p.add_argument("--replay-weight", type=int, default=1)
    p.add_argument(
        "--tiny-threshold",
        type=int,
        default=10,
        help="Disable validation/early stopping when new chat pairs are below this count.",
    )
    return p.parse_args()


def load_chat_jsonl(path: Path) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    if not path.exists():
        return pairs
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
        user = str(row.get("user", "")).strip()
        answer = str(row.get("assistant", "")).strip()
        source = str(row.get("source", "chat-auto")).strip() or "chat-auto"
        if user and answer:
            pairs.append((user, answer, source))
    return pairs



def normalize_pair_text(text: str) -> str:
    return " ".join(text.strip().split())


def pair_fingerprint(user: str, answer: str) -> str:
    payload = normalize_pair_text(user) + "\n" + normalize_pair_text(answer)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deduplicate_trusted_rows(
    rows: List[Tuple[str, str, str]],
) -> List[Tuple[str, str, str]]:
    priority = {
        "chat-approved": 1,
        "chat-manual": 2,
        "chat-recovery": 3,
    }
    selected: dict[str, Tuple[str, str, str]] = {}

    for user, answer, source in rows:
        if source not in priority:
            continue
        fp = pair_fingerprint(user, answer)
        previous = selected.get(fp)
        if (
            previous is None
            or priority[source] > priority[previous[2]]
        ):
            selected[fp] = (user, answer, source)

    return list(selected.values())


def load_training_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("trained_fingerprints", [])
    return {str(x) for x in values}


def save_training_state(path: Path, fingerprints: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v1.6.2",
        "trained_fingerprints": sorted(fingerprints),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

def load_replay_pairs(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        return []
    pairs: List[Tuple[str, str]] = []
    pending_user: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(USER_PREFIX):
            pending_user = line[len(USER_PREFIX):].strip()
        elif line.startswith(AI_PREFIX) and pending_user is not None:
            answer = line[len(AI_PREFIX):].strip()
            if pending_user and answer:
                pairs.append((pending_user, answer))
            pending_user = None
    return pairs


class ConversationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        tokenizer: Tokenizer,
        context_length: int,
    ):
        self.rows = []
        self.pad_id = tokenizer.pad_id

        for user_text, answer_text in pairs:
            prompt_ids = tokenizer.encode(
                f"{USER_PREFIX}{user_text}\n{AI_PREFIX}",
                add_bos=True,
            )
            answer_ids = tokenizer.encode(answer_text, add_eos=True)
            max_sequence = context_length + 1

            if len(prompt_ids) + len(answer_ids) > max_sequence:
                keep_prompt = max(1, max_sequence - len(answer_ids))
                prompt_ids = prompt_ids[-keep_prompt:]

            if len(prompt_ids) + len(answer_ids) > max_sequence:
                room = max_sequence - len(prompt_ids)
                answer_ids = answer_ids[:max(1, room)]
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

            pad_count = context_length - len(x)
            x += [tokenizer.pad_id] * pad_count
            y += [tokenizer.pad_id] * pad_count
            mask += [0.0] * pad_count

            self.rows.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(mask, dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def masked_loss(model, x, y, mask):
    logits = model(x)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).view_as(y)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        total += float(masked_loss(model, x, y, mask).item())
        count += 1
    return total / max(1, count)


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chat_path = Path(args.chat_data)
    replay_path = Path(args.replay_data)
    tokenizer_path = Path(args.tokenizer)
    base_path = Path(args.base_model)
    output_path = Path(args.output)

    new_rows = load_chat_jsonl(chat_path)
    if len(new_rows) < 1:
        raise ValueError("At least 1 chat learning pair is required.")

    state_path = Path(args.state)
    trained_fingerprints = load_training_state(state_path)
    trusted_rows = deduplicate_trusted_rows(new_rows)

    pending_rows = [
        row for row in trusted_rows
        if pair_fingerprint(row[0], row[1]) not in trained_fingerprints
    ]
    historical_rows = [
        row for row in trusted_rows
        if pair_fingerprint(row[0], row[1]) in trained_fingerprints
    ]

    if not pending_rows:
        print("No new trusted learning pairs. Incremental training skipped.")
        raise SystemExit(3)

    manual_rows = [(u, a) for u, a, s in pending_rows if s == "chat-manual"]
    approved_rows = [(u, a) for u, a, s in pending_rows if s == "chat-approved"]
    recovery_rows = [(u, a) for u, a, s in pending_rows if s == "chat-recovery"]
    auto_rows = [
        (u, a) for u, a, s in new_rows
        if s not in ("chat-manual", "chat-approved", "chat-recovery")
    ]

    weighted_new_pairs: List[Tuple[str, str]] = []
    for pair in manual_rows:
        weighted_new_pairs.extend([pair] * max(1, args.manual_weight))
    for pair in approved_rows:
        weighted_new_pairs.extend([pair] * max(1, args.manual_weight))
    for pair in recovery_rows:
        weighted_new_pairs.extend([pair] * max(1, args.recovery_weight))

    trusted_replay: List[Tuple[str, str]] = []
    for user, answer, _ in historical_rows:
        trusted_replay.extend(
            [(user, answer)] * max(0, args.trusted_replay_weight)
        )

    replay = load_replay_pairs(replay_path)
    replay_count = min(
        len(replay),
        max(0, int(round(len(pending_rows) * args.replay_ratio))),
    )
    random.shuffle(replay)
    replay_pairs = replay[:replay_count]
    weighted_replay: List[Tuple[str, str]] = []
    for pair in replay_pairs:
        weighted_replay.extend([pair] * max(1, args.replay_weight))

    pairs = list(weighted_new_pairs) + trusted_replay + weighted_replay
    random.shuffle(pairs)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(str(base_path), device=device)

    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    tiny_mode = len(pending_rows) < args.tiny_threshold
    if tiny_mode:
        val_pairs: List[Tuple[str, str]] = []
        train_pairs = pairs
    else:
        val_count = max(1, int(round(len(pairs) * args.validation_ratio)))
        val_count = min(val_count, len(pairs) - 1)
        val_pairs = pairs[:val_count]
        train_pairs = pairs[val_count:]

    train_ds = ConversationDataset(train_pairs, tokenizer, model.context_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if val_pairs:
        val_ds = ConversationDataset(val_pairs, tokenizer, model.context_length)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    print("=" * 72)
    print(" LLM_GPU v1.6.20 Recovery Stabilization Learning")
    print("=" * 72)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Base model      :", base_path)
    print("Base loss       :", checkpoint.get("loss"))
    print("Chat log rows   :", len(new_rows))
    print("Unique trusted  :", len(trusted_rows))
    print("New trusted     :", len(pending_rows))
    print("Prior trusted   :", len(historical_rows), f"(x{args.trusted_replay_weight})")
    print("Manual new      :", len(manual_rows), f"(x{args.manual_weight})")
    print("Approved new    :", len(approved_rows), f"(x{args.manual_weight})")
    print("Recovery new    :", len(recovery_rows), f"(x{args.recovery_weight})")
    print("Legacy auto     :", len(auto_rows), "(ignored)")
    print("Corpus replay   :", replay_count, f"(x{args.replay_weight})")
    print("Tiny-data mode  :", tiny_mode)
    print("Train pairs     :", len(train_pairs))
    print("Validation pairs:", len(val_pairs))
    print("Learning rate   :", args.learning_rate)
    print("Epoch limit     :", args.epochs)
    print()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        started = time.perf_counter()

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model, x, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            batches += 1

        train_loss = total / max(1, batches)
        elapsed = time.perf_counter() - started

        if val_loader is None:
            print(
                f"epoch={epoch:02d} train={train_loss:.6f} "
                f"time={elapsed:.2f}s"
            )
            if train_loss < best_val:
                best_val = train_loss
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            continue

        val_loss = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch:02d} train={train_loss:.6f} "
            f"val={val_loss:.6f} ppl={math.exp(min(val_loss, 20.0)):.2f} "
            f"time={elapsed:.2f}s"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced.")

    model.load_state_dict(best_state)
    model.to(device)

    recovery_stabilization_loss = None
    if recovery_rows and args.recovery_stabilization_epochs > 0:
        recovery_ds = ConversationDataset(
            recovery_rows,
            tokenizer,
            model.context_length,
        )
        recovery_loader = DataLoader(
            recovery_ds,
            batch_size=1,
            shuffle=True,
        )
        recovery_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.recovery_learning_rate,
            weight_decay=0.01,
        )

        print()
        print("Recovery stabilization")
        print("----------------------")
        print("Recovery pairs   :", len(recovery_rows))
        print("Recovery LR      :", args.recovery_learning_rate)
        print("Recovery epochs  :", args.recovery_stabilization_epochs)

        for recovery_epoch in range(
            1,
            args.recovery_stabilization_epochs + 1,
        ):
            model.train()
            total = 0.0
            batches = 0

            for x, y, mask in recovery_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                recovery_optimizer.zero_grad(set_to_none=True)
                loss = masked_loss(model, x, y, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                recovery_optimizer.step()
                total += float(loss.item())
                batches += 1

            recovery_stabilization_loss = total / max(1, batches)
            print(
                f"recovery_epoch={recovery_epoch:02d} "
                f"loss={recovery_stabilization_loss:.6f}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        str(output_path),
        optimizer=optimizer,
        epoch=best_epoch,
        loss=best_val,
    )

    trained_fingerprints.update(
        pair_fingerprint(user, answer)
        for user, answer, _ in pending_rows
    )
    save_training_state(state_path, trained_fingerprints)

    print()
    print("Training completed.")
    print("Best epoch :", best_epoch)
    print("Best metric:", f"{best_val:.6f}", "(train loss in tiny-data mode)" if tiny_mode else "(validation loss)")
    if recovery_stabilization_loss is not None:
        print("Recovery loss:", f"{recovery_stabilization_loss:.6f}")
    print("Saved      :", output_path)
    print("State      :", state_path)
    print("Consumed   :", len(pending_rows), "new trusted pair(s)")


if __name__ == "__main__":
    main()
