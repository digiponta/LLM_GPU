# diagnose_first_token_competitors_v0115.py
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from cpu_gpu_direct_margin_binding_v0113 import load_checkpoint
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_ROLE = "model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt"
DEFAULT_BINDING = "model/model-gpu-v0.11.4-d16_lr2e3_m15.pt"

G05 = "コンピュータの中心で多様な命令を処理する装置は何ですか。"

def token_text(tok, token_id: int) -> str:
    try:
        text = tok.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        text = ""
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

def rank_of(logits: torch.Tensor, token_id: int) -> int:
    target = logits[int(token_id)]
    return int((logits > target).sum().item()) + 1

def print_top(title: str, logits: torch.Tensor, tok: Tokenizer, k: int):
    vals, ids = torch.topk(logits, min(k, logits.numel()))
    print(title)
    print("-" * len(title))
    print(f"{'Rank':>4s} {'TokenID':>7s} {'Logit':>10s}  Token")
    for rank, (value, token_id) in enumerate(zip(vals.tolist(), ids.tolist()), start=1):
        print(f"{rank:4d} {int(token_id):7d} {float(value):+10.4f}  {token_text(tok, int(token_id))!r}")
    print()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--role-checkpoint", default=DEFAULT_ROLE)
    p.add_argument("--binding", default=DEFAULT_BINDING)
    p.add_argument("--prompt", default=G05)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    for filename in (args.tokenizer, args.model, args.intent_head, args.role_checkpoint, args.binding):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    model, base_ck = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    intent_head, intent_ck, intent_labels = load_intent_head(args.intent_head, model, device)
    role_head, _old_binding, role_ck = load_role_checkpoint(args.role_checkpoint, intent_labels, device)
    adapter, bind_ck = load_checkpoint(args.binding, intent_labels, device)

    for module in (model, intent_head, role_head, adapter):
        module.eval()

    text = f"人: {args.prompt}\nAI: "
    ids = tok.encode(text, add_bos=True)[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        hidden = model.forward_hidden(x)
        h = hidden[:, -1, :]
        base_logits = model.lm_head(h)[0].clone()
        ip = torch.sigmoid(intent_head(h))
        rp = torch.sigmoid(role_head(h))

    cpu_idx = intent_labels.index("tech_cpu")
    gpu_idx = intent_labels.index("tech_gpu")
    cpu = float(ip[0, cpu_idx].item())
    gpu = float(ip[0, gpu_idx].item())
    controller = float(rp[0, 0].item())
    executor = float(rp[0, 1].item())

    intent_margin = float(bind_ck.get("intent_margin_threshold", 0.10))
    role_margin = float(bind_ck.get("role_margin_threshold", 0.05))

    cpu_case = cpu > gpu and (cpu-gpu) >= intent_margin and controller > executor and (controller-executor) >= role_margin
    gpu_case = gpu > cpu and (gpu-cpu) >= intent_margin and executor > controller and (executor-controller) >= role_margin
    active = cpu_case or gpu_case

    after_logits = base_logits.clone()
    delta = 0.0
    if active:
        with torch.no_grad():
            features = torch.cat([ip[0], rp[0]], dim=-1).unsqueeze(0)
            delta = float(adapter(features)[0].item())
        after_logits[int(bind_ck["cpu_token_id"])] += delta / 2.0
        after_logits[int(bind_ck["gpu_token_id"])] -= delta / 2.0

    cpu_id = int(bind_ck["cpu_token_id"])
    gpu_id = int(bind_ck["gpu_token_id"])
    dpu_ids = tok.encode("DPU")
    dpu_first = int(dpu_ids[0]) if dpu_ids else None

    print("====================================================")
    print(" First-Token Competitor Diagnostic v0.11.5")
    print("====================================================")
    print("Device            :", device)
    print("Prompt            :", args.prompt)
    print("Binding checkpoint:", args.binding)
    print("Binding loss      :", bind_ck.get("loss"))
    print("Binding active    :", active)
    print(f"Intent            : CPU={cpu:.6f} GPU={gpu:.6f}")
    print(f"Role              : controller={controller:.6f} executor={executor:.6f}")
    print(f"Signed delta      : {delta:+.6f}")
    print()

    print("Candidate tokenization")
    print("----------------------")
    for label in ("CPU", "GPU", "DPU"):
        tids = tok.encode(label)
        pieces = [token_text(tok, tid) for tid in tids]
        print(f"{label:4s}: ids={tids} pieces={pieces}")
    print()

    print_top("Base first-token top-k", base_logits, tok, args.top_k)
    print_top("After-binding first-token top-k", after_logits, tok, args.top_k)

    print("Tracked candidates")
    print("------------------")
    tracked = [("CPU", cpu_id), ("GPU", gpu_id)]
    if dpu_first is not None:
        tracked.append(("DPU-first", dpu_first))
    seen = set()
    for name, tid in tracked:
        key=(name,tid)
        if key in seen:
            continue
        seen.add(key)
        print(
            f"{name:10s} id={tid:5d} token={token_text(tok, tid)!r} "
            f"base_logit={float(base_logits[tid]):+.4f} base_rank={rank_of(base_logits, tid):4d} "
            f"after_logit={float(after_logits[tid]):+.4f} after_rank={rank_of(after_logits, tid):4d}"
        )

    base_top_id = int(torch.argmax(base_logits).item())
    after_top_id = int(torch.argmax(after_logits).item())
    print()
    print("Winner")
    print("------")
    print(f"Base  : id={base_top_id} token={token_text(tok, base_top_id)!r} logit={float(base_logits[base_top_id]):+.4f}")
    print(f"After : id={after_top_id} token={token_text(tok, after_top_id)!r} logit={float(after_logits[after_top_id]):+.4f}")
    print(f"CPU vs winner after gap: {float(after_logits[cpu_id]-after_logits[after_top_id]):+.4f}")

if __name__ == "__main__":
    main()
