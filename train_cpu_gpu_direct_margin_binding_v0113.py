# train_cpu_gpu_direct_margin_binding_v0113.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from augment_sft_v07 import CPU_GPU_RELATIONAL_ROWS, REVERSE_DEFINITION_ROWS, TARGETED_BOUNDARY_ROWS, TARGETED_BOUNDARY_V2_ROWS
from cpu_gpu_direct_margin_binding_v0113 import DirectGapBinding, save_checkpoint
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from evaluate_generalization_v07 import CASES
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_ROLE = "model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.11.3-direct-cpu-gpu-gap-binding.pt"

SEED = 42
TARGET_MARGIN = 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--role-checkpoint", default=DEFAULT_ROLE)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=240)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--max-delta", type=float, default=12.0)
    p.add_argument("--l2-weight", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=30)
    return p.parse_args()


def build_rows():
    rows = []
    for prompt, answer, _rel in CPU_GPU_RELATIONAL_ROWS:
        if answer.startswith("CPU"):
            rows.append((prompt, 1))
        elif answer.startswith("GPU"):
            rows.append((prompt, -1))

    for group in (REVERSE_DEFINITION_ROWS, TARGETED_BOUNDARY_ROWS, TARGETED_BOUNDARY_V2_ROWS):
        for prompt, answer, label in group:
            if label == "tech_cpu" and answer.startswith("CPU"):
                rows.append((prompt, 1))
            elif label == "tech_gpu" and answer.startswith("GPU"):
                rows.append((prompt, -1))

    seen = set()
    out = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            out.append(row)
    return out


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model, args.intent_head, args.role_checkpoint):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    rows = build_rows()
    dev = {str(case["prompt"]) for case in CASES}
    overlap = [prompt for prompt, _sign in rows if prompt in dev]
    if overlap:
        raise RuntimeError("Exact DEV overlap: " + repr(overlap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ck = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()

    intent_head, intent_ck, intent_labels = load_intent_head(args.intent_head, model, device)
    role_head, _old_binding, role_ck = load_role_checkpoint(args.role_checkpoint, intent_labels, device)
    role_head.eval()

    for module in (model, intent_head, role_head):
        for p in module.parameters():
            p.requires_grad_(False)

    cpu_id = int(tokenizer.encode("CPU")[0])
    gpu_id = int(tokenizer.encode("GPU")[0])

    features = []
    base_gap = []
    signs = []

    with torch.no_grad():
        for prompt, sign in rows:
            ids = tokenizer.encode(f"人: {prompt}\nAI: ", add_bos=True)[-model.context_length:]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            hidden = model.forward_hidden(x)
            h = hidden[:, -1, :]
            ip = torch.sigmoid(intent_head(h))
            rp = torch.sigmoid(role_head(h))
            logits = model.lm_head(h)[0]

            features.append(torch.cat([ip[0], rp[0]], dim=-1))
            base_gap.append(float((logits[cpu_id] - logits[gpu_id]).item()))
            signs.append(sign)

    features = torch.stack(features)
    base_gap = torch.tensor(base_gap, dtype=torch.float32, device=device)
    signs = torch.tensor(signs, dtype=torch.float32, device=device)

    cpu_idx = [i for i, s in enumerate(signs.tolist()) if s > 0]
    gpu_idx = [i for i, s in enumerate(signs.tolist()) if s < 0]
    val_list = [cpu_idx[-1], gpu_idx[-1]]
    train_list = [i for i in range(len(rows)) if i not in set(val_list)]

    train_idx = torch.tensor(train_list, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_list, dtype=torch.long, device=device)

    adapter = DirectGapBinding(
        input_dim=features.size(1),
        hidden_dim=args.hidden_dim,
        max_delta=args.max_delta,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)

    def compute(indices):
        delta = adapter(features[indices])
        corrected_gap = base_gap[indices] + delta
        signed_gap = signs[indices] * corrected_gap
        hinge = F.relu(TARGET_MARGIN - signed_gap).mean()
        reg = delta.pow(2).mean()
        total = hinge + args.l2_weight * reg
        return total, hinge, reg, delta, corrected_gap

    print("====================================================")
    print(" Direct CPU/GPU Logit Margin Binding v0.11.3")
    print("====================================================")
    print("Device:", device)
    print("Base model / intent head / role head: frozen")
    print("Rows:", len(rows), "Train:", len(train_idx), "Val:", len(val_idx))
    print("Exact DEV overlap:", len(overlap))
    print("Target signed margin:", TARGET_MARGIN)
    print("CPU token:", cpu_id, "GPU token:", gpu_id)
    print("Input dim:", features.size(1), "Hidden dim:", args.hidden_dim)

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        loss, hinge, reg, delta, corrected_gap = compute(train_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        adapter.eval()
        with torch.no_grad():
            val, vhinge, vreg, vdelta, vgap = compute(val_idx)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"| train={loss.item():.4f} hinge={hinge.item():.4f} "
                f"| val={val.item():.4f} hinge={vhinge.item():.4f}"
            )

        value = float(val.item())
        if value < best - 1e-6:
            best = value
            best_epoch = epoch
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    adapter.load_state_dict(best_state)

    save_checkpoint(
        args.output,
        adapter,
        intent_labels,
        args.role_checkpoint,
        epoch=best_epoch,
        loss=best,
        base_model=args.model,
        intent_head=args.intent_head,
        cpu_token_id=cpu_id,
        gpu_token_id=gpu_id,
        target_margin=TARGET_MARGIN,
        intent_margin_threshold=0.10,
        role_margin_threshold=0.05,
        learning_rate=args.lr,
    )

    print("Best epoch:", best_epoch)
    print("Best val loss:", f"{best:.6f}")
    print("Checkpoint:", args.output)


if __name__ == "__main__":
    main()
