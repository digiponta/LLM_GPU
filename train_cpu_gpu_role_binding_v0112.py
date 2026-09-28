# train_cpu_gpu_role_binding_v0112.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from augment_sft_v07 import (
    CPU_GPU_RELATIONAL_ROWS,
    REVERSE_DEFINITION_ROWS,
    TARGETED_BOUNDARY_ROWS,
    TARGETED_BOUNDARY_V2_ROWS,
    BALANCED_CONTROL_REPLAY_ROWS,
    RELATION_AUGMENT_ROWS,
)
from cpu_gpu_role_binding_v0112 import RoleHead, RoleEntityBinding, save_checkpoint
from evaluate_generalization_v07 import CASES
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt"

SEED = 42
MARGIN = 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--role-weight", type=float, default=0.50)
    p.add_argument("--margin-weight", type=float, default=1.00)
    p.add_argument("--l2-weight", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=25)
    return p.parse_args()


def cpu_gpu_rows():
    rows = []
    for prompt, answer, rel in CPU_GPU_RELATIONAL_ROWS:
        if answer.startswith("CPU"):
            rows.append((prompt, "controller", "CPU"))
        elif answer.startswith("GPU"):
            rows.append((prompt, "executor", "GPU"))

    for group in (REVERSE_DEFINITION_ROWS, TARGETED_BOUNDARY_ROWS, TARGETED_BOUNDARY_V2_ROWS):
        for prompt, answer, label in group:
            if label == "tech_cpu" and answer.startswith("CPU"):
                rows.append((prompt, "controller", "CPU"))
            elif label == "tech_gpu" and answer.startswith("GPU"):
                rows.append((prompt, "executor", "GPU"))

    seen = set()
    out = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            out.append(row)
    return out


def negative_rows():
    rows = []

    # Non-CPU/GPU technical relations.
    for prompt, _answer in RELATION_AUGMENT_ROWS:
        if "CPU" not in prompt and "GPU" not in prompt:
            rows.append(prompt)

    # Conversational controls.
    for prompt, _answer, _label in BALANCED_CONTROL_REPLAY_ROWS:
        rows.append(prompt)

    # Fixed, non-CPU/GPU held-out style negatives are not copied from DEV.
    rows.extend([
        "文章生成モデルについて説明してください。",
        "Attentionを使う構造について教えてください。",
        "プログラミング言語について説明してください。",
        "エラー原因を調べたいです。",
        "別の話題へ移りたいです。",
        "回答を短くしてください。",
        "日本の首都について教えてください。",
    ])
    return list(dict.fromkeys(rows))


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model, args.intent_head):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    positives = cpu_gpu_rows()
    negatives = negative_rows()

    dev = {str(case["prompt"]) for case in CASES}
    overlap = [p for p, _, _ in positives if p in dev] + [p for p in negatives if p in dev]
    if overlap:
        raise RuntimeError("Exact DEV overlap: " + repr(overlap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ck = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()

    intent_head, intent_ck, intent_labels = load_intent_head(args.intent_head, model, device)
    for module in (model, intent_head):
        for p in module.parameters():
            p.requires_grad_(False)

    cpu_id = int(tokenizer.encode("CPU")[0])
    gpu_id = int(tokenizer.encode("GPU")[0])

    hidden_rows = []
    intent_rows = []
    base_logits_rows = []
    role_targets = []
    entity_targets = []
    is_positive = []

    @torch.no_grad()
    def encode_prompt(prompt):
        ids = tokenizer.encode(f"人: {prompt}\nAI: ", add_bos=True)[-model.context_length:]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        hidden = model.forward_hidden(x)
        h = hidden[:, -1, :]
        ip = torch.sigmoid(intent_head(h))
        logits = model.lm_head(h)
        return h[0], ip[0], logits[0]

    for prompt, role, entity in positives:
        h, ip, logits = encode_prompt(prompt)
        hidden_rows.append(h)
        intent_rows.append(ip)
        base_logits_rows.append(logits)
        role_targets.append(torch.tensor([1.0, 0.0] if role == "controller" else [0.0, 1.0], device=device))
        entity_targets.append(0 if entity == "CPU" else 1)
        is_positive.append(True)

    for prompt in negatives:
        h, ip, logits = encode_prompt(prompt)
        hidden_rows.append(h)
        intent_rows.append(ip)
        base_logits_rows.append(logits)
        role_targets.append(torch.tensor([0.0, 0.0], device=device))
        entity_targets.append(-1)
        is_positive.append(False)

    hidden_rows = torch.stack(hidden_rows)
    intent_rows = torch.stack(intent_rows)
    base_logits_rows = torch.stack(base_logits_rows)
    role_targets = torch.stack(role_targets)
    entity_targets = torch.tensor(entity_targets, dtype=torch.long, device=device)
    is_positive = torch.tensor(is_positive, dtype=torch.bool, device=device)

    # deterministic validation: one CPU positive, one GPU positive, and two negatives
    cpu_idx = [i for i, x in enumerate(entity_targets.tolist()) if x == 0]
    gpu_idx = [i for i, x in enumerate(entity_targets.tolist()) if x == 1]
    neg_idx = [i for i, x in enumerate(entity_targets.tolist()) if x < 0]

    val_list = [cpu_idx[-1], gpu_idx[-1]]
    if len(neg_idx) >= 2:
        val_list += neg_idx[-2:]
    elif neg_idx:
        val_list += [neg_idx[-1]]

    val_set = set(val_list)
    train_list = [i for i in range(len(entity_targets)) if i not in val_set]

    train_idx = torch.tensor(train_list, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_list, dtype=torch.long, device=device)

    role_head = RoleHead(model.d_model).to(device)
    binding = RoleEntityBinding(
        len(intent_labels), 2, model.vocab_size, rank=args.rank, beta=args.beta
    ).to(device)

    params = list(role_head.parameters()) + list(binding.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    def compute(indices):
        h = hidden_rows[indices]
        ip = intent_rows[indices]
        base = base_logits_rows[indices]
        rt = role_targets[indices]
        et = entity_targets[indices]

        role_logits = role_head(h)
        role_prob = torch.sigmoid(role_logits)
        role_loss = F.binary_cross_entropy_with_logits(role_logits, rt)

        bias = binding(ip, role_prob)
        combined = base + bias

        pos_mask = et >= 0
        if pos_mask.any():
            pos_logits = combined[pos_mask]
            pos_target = et[pos_mask]
            cpu = pos_logits[:, cpu_id]
            gpu = pos_logits[:, gpu_id]

            target_logit = torch.where(pos_target == 0, cpu, gpu)
            other_logit = torch.where(pos_target == 0, gpu, cpu)
            margin_loss = F.relu(MARGIN - (target_logit - other_logit)).mean()
        else:
            margin_loss = torch.tensor(0.0, device=device)

        reg = bias.pow(2).mean()
        total = (
            args.role_weight * role_loss
            + args.margin_weight * margin_loss
            + args.l2_weight * reg
        )
        return total, role_loss, margin_loss, reg

    print("====================================================")
    print(" CPU-GPU Role Binding v0.11.2")
    print("====================================================")
    print("Device:", device)
    print("Base model / clean intent head: frozen")
    print("Roles: controller / executor")
    print("Positive rows:", len(positives))
    print("Negative role rows:", len(negatives))
    print("Train rows:", len(train_idx), "Validation rows:", len(val_idx))
    print("Exact DEV overlap:", len(overlap))
    print("CPU token:", cpu_id, "GPU token:", gpu_id)
    print("Direct CPU/GPU margin:", MARGIN)

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        role_head.train()
        binding.train()
        optimizer.zero_grad(set_to_none=True)

        loss, role_loss, margin_loss, reg = compute(train_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        role_head.eval()
        binding.eval()
        with torch.no_grad():
            val, vrole, vmargin, vreg = compute(val_idx)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"| train={loss.item():.4f} role={role_loss.item():.4f} margin={margin_loss.item():.4f} "
                f"| val={val.item():.4f} role={vrole.item():.4f} margin={vmargin.item():.4f}"
            )

        value = float(val.item())
        if value < best - 1e-6:
            best = value
            best_epoch = epoch
            bad = 0
            best_state = {
                "role": {k: v.detach().cpu().clone() for k, v in role_head.state_dict().items()},
                "binding": {k: v.detach().cpu().clone() for k, v in binding.state_dict().items()},
            }
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    role_head.load_state_dict(best_state["role"])
    binding.load_state_dict(best_state["binding"])

    save_checkpoint(
        args.output,
        role_head,
        binding,
        intent_labels,
        {"CPU": cpu_id, "GPU": gpu_id},
        epoch=best_epoch,
        loss=best,
        base_model=args.model,
        intent_head=args.intent_head,
        learning_rate=args.lr,
        margin=MARGIN,
        role_weight=args.role_weight,
        margin_weight=args.margin_weight,
        role_gate_threshold=0.55,
        entity_margin_threshold=0.10,
    )

    print("Best epoch:", best_epoch)
    print("Best val loss:", f"{best:.6f}")
    print("Checkpoint:", args.output)


if __name__ == "__main__":
    main()
