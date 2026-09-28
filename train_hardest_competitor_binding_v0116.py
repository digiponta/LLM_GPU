# train_hardest_competitor_binding_v0116.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from augment_sft_v07 import CPU_GPU_RELATIONAL_ROWS, REVERSE_DEFINITION_ROWS, TARGETED_BOUNDARY_ROWS, TARGETED_BOUNDARY_V2_ROWS
from hardest_competitor_binding_v0116 import HardestCompetitorBoost, save_checkpoint
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from evaluate_generalization_v07 import CASES
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_ROLE = "model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.11.6-hardest-competitor-binding.pt"

SEED = 42


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
    p.add_argument("--max-boost", type=float, default=12.0)
    p.add_argument("--target-margin", type=float, default=0.5)
    p.add_argument("--l2-weight", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=30)
    return p.parse_args()


def build_rows():
    rows = []
    for prompt, answer, _rel in CPU_GPU_RELATIONAL_ROWS:
        if answer.startswith("CPU"):
            rows.append((prompt, "CPU"))
        elif answer.startswith("GPU"):
            rows.append((prompt, "GPU"))

    for group in (REVERSE_DEFINITION_ROWS, TARGETED_BOUNDARY_ROWS, TARGETED_BOUNDARY_V2_ROWS):
        for prompt, answer, label in group:
            if label == "tech_cpu" and answer.startswith("CPU"):
                rows.append((prompt, "CPU"))
            elif label == "tech_gpu" and answer.startswith("GPU"):
                rows.append((prompt, "GPU"))

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
    overlap = [prompt for prompt, _entity in rows if prompt in dev]
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

    cpu_token = int(tokenizer.encode("CPU")[0])
    gpu_token = int(tokenizer.encode("GPU")[0])  # first token is G

    features = []
    base_logits_rows = []
    target_ids = []
    target_entities = []

    with torch.no_grad():
        for prompt, entity in rows:
            ids = tokenizer.encode(f"人: {prompt}\nAI: ", add_bos=True)[-model.context_length:]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            hidden = model.forward_hidden(x)
            h = hidden[:, -1, :]
            ip = torch.sigmoid(intent_head(h))
            rp = torch.sigmoid(role_head(h))
            logits = model.lm_head(h)[0]

            features.append(torch.cat([ip[0], rp[0]], dim=-1))
            base_logits_rows.append(logits)
            target_ids.append(cpu_token if entity == "CPU" else gpu_token)
            target_entities.append(entity)

    features = torch.stack(features)
    base_logits_rows = torch.stack(base_logits_rows)
    target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)

    cpu_idx = [i for i, e in enumerate(target_entities) if e == "CPU"]
    gpu_idx = [i for i, e in enumerate(target_entities) if e == "GPU"]
    val_list = [cpu_idx[-1], gpu_idx[-1]]
    train_list = [i for i in range(len(rows)) if i not in set(val_list)]

    train_idx = torch.tensor(train_list, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_list, dtype=torch.long, device=device)

    adapter = HardestCompetitorBoost(
        input_dim=features.size(1),
        hidden_dim=args.hidden_dim,
        max_boost=args.max_boost,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)

    def compute(indices):
        feats = features[indices]
        base = base_logits_rows[indices]
        tids = target_ids[indices]

        boost = adapter(feats)
        corrected = base.clone()
        corrected[torch.arange(len(indices), device=device), tids] += boost

        target = corrected[torch.arange(len(indices), device=device), tids]
        masked = corrected.clone()
        masked[torch.arange(len(indices), device=device), tids] = -torch.inf
        competitor, competitor_ids = masked.max(dim=1)

        hinge = F.relu(args.target_margin - (target - competitor)).mean()
        reg = boost.pow(2).mean()
        total = hinge + args.l2_weight * reg
        return total, hinge, reg, boost, target, competitor, competitor_ids

    print("====================================================")
    print(" Hardest-Competitor Margin Binding v0.11.6")
    print("====================================================")
    print("Device:", device)
    print("Base model / intent head / role head: frozen")
    print("Rows:", len(rows), "Train:", len(train_idx), "Val:", len(val_idx))
    print("Exact DEV overlap:", len(overlap))
    print("CPU first token:", cpu_token, tokenizer.decode([cpu_token], skip_special_tokens=False))
    print("GPU first token:", gpu_token, tokenizer.decode([gpu_token], skip_special_tokens=False))
    print("Target margin over full vocabulary:", args.target_margin)

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        loss, hinge, reg, boost, target, competitor, comp_ids = compute(train_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        adapter.eval()
        with torch.no_grad():
            val, vhinge, vreg, vboost, vtarget, vcompetitor, vcomp_ids = compute(val_idx)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"| train={loss.item():.4f} hinge={hinge.item():.4f} "
                f"| val={val.item():.4f} hinge={vhinge.item():.4f} "
                f"| boost={boost.mean().item():.3f}"
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
        cpu_token_id=cpu_token,
        gpu_token_id=gpu_token,
        target_margin=args.target_margin,
        intent_margin_threshold=0.10,
        role_margin_threshold=0.05,
        learning_rate=args.lr,
    )

    print("Best epoch:", best_epoch)
    print("Best val loss:", f"{best:.6f}")
    print("Checkpoint:", args.output)


if __name__ == "__main__":
    main()
