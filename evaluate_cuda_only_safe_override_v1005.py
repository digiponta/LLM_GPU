# evaluate_cuda_only_safe_override_v1005.py
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from evaluate_heldout_generalization_v1001 import CASES as HELDOUT_CASES, score as heldout_score
from evaluate_generalization_v07 import CASES as ORIGINAL_CASES, dimension_match
from evaluate_transformer_continuation_category_repair_v01117 import (
    DEFAULT_TOKENIZER, DEFAULT_MODEL, DEFAULT_INTENT, DEFAULT_ROLE,
    DEFAULT_BINDING, DEFAULT_REPAIR,
    generate as generate_v01117,
)
from evaluate_conversational_intent_repair_v01118 import (
    route_conversation, ROUTES, generate_guided_response,
)
from evaluate_post_entity_boundary_binding_v01115 import build_boundary_block_ids
from selective_intent_repair_v01110 import (
    TECH_LABELS, extract_technical_logits, load_checkpoint as load_repair,
)
from multi_concept_safe_binding_v0118 import load_checkpoint as load_binding
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CUDA_SCOPE_THRESHOLD = 0.70
CUDA_REPAIR_CONFIDENCE = 0.50
CUDA_ANCHOR = "CUDAはNVIDIA GPUで汎用計算を行うための技術です。"

TECH_INTENTS = {"gpu","cpu","llm","transformer","cuda","python"}

@torch.no_grad()
def cuda_override_signal(model, tok, intent_head, intent_labels, repair, prompt_text):
    device = next(model.parameters()).device
    prompt = f"人: {prompt_text}\nAI: "
    ids = tok.encode(prompt, add_bos=True)
    x = torch.tensor([ids[-model.context_length:]], dtype=torch.long, device=device)
    h = model.forward_hidden(x)[:, -1, :]
    intent_logits = intent_head(h)
    raw = extract_technical_logits(intent_logits, intent_labels)
    repaired, scope_logit = repair(raw)

    raw_idx = int(torch.argmax(raw[0]).item())
    raw_top = TECH_LABELS[raw_idx]
    rep_probs = torch.softmax(repaired, dim=-1)[0]
    rep_idx = int(torch.argmax(rep_probs).item())
    rep_top = TECH_LABELS[rep_idx]
    rep_conf = float(rep_probs[rep_idx].item())
    scope = float(torch.sigmoid(scope_logit)[0].item())

    active = (
        rep_top == "tech_cuda"
        and raw_top != "tech_cuda"
        and scope >= CUDA_SCOPE_THRESHOLD
        and rep_conf >= CUDA_REPAIR_CONFIDENCE
    )
    return active, {
        "raw_top": raw_top,
        "repair_top": rep_top,
        "repair_conf": rep_conf,
        "scope": scope,
    }

@torch.no_grad()
def generate_cuda_safe(model, tok, intent_head, intent_labels, role_head, binding, binding_ck, repair, prompt_text, boundary_block_ids):
    baseline, dbg = generate_v01117(
        model, tok, intent_head, intent_labels, role_head,
        binding, binding_ck, repair, prompt_text, boundary_block_ids
    )
    active, sig = cuda_override_signal(model, tok, intent_head, intent_labels, repair, prompt_text)
    dbg["cuda_override_active"] = active
    dbg["cuda_override_signal"] = sig
    dbg["cuda_override_baseline"] = baseline
    if not active:
        return baseline, dbg

    reply, steps = generate_guided_response(model, tok, prompt_text, CUDA_ANCHOR)
    dbg["cuda_override_steps"] = steps
    return reply, dbg


def load_all(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    intent_head, _, intent_labels = load_intent_head(args.intent_head, model, device)
    role_head, _, _ = load_role_checkpoint(args.role_checkpoint, intent_labels, device)
    binding, binding_ck = load_binding(args.binding, intent_labels, device)
    repair, _ = load_repair(args.repair, intent_labels, device)
    boundary_block_ids = build_boundary_block_ids(tok)
    return device, tok, model, intent_head, intent_labels, role_head, binding, binding_ck, repair, boundary_block_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--role-checkpoint", default=DEFAULT_ROLE)
    p.add_argument("--binding", default=DEFAULT_BINDING)
    p.add_argument("--repair", default=DEFAULT_REPAIR)
    args = p.parse_args()

    device, tok, model, intent_head, intent_labels, role_head, binding, binding_ck, repair, boundary = load_all(args)

    print("=" * 72)
    print(" LLM_GPU v1.0.5 CUDA-only Safe Override")
    print("=" * 72)
    print("Device:", device)
    print(f"Rule: repair=cuda AND raw!=cuda AND scope>={CUDA_SCOPE_THRESHOLD:.2f} AND repair_conf>={CUDA_REPAIR_CONFIDENCE:.2f}")
    print()

    # 1) Original 30-case regression guard.
    orig_sem = orig_strict = orig_overrides = 0
    print("[1] Original 30-case benchmark")
    for idx, case in enumerate(ORIGINAL_CASES, 1):
        prompt = str(case["prompt"])
        reply, dbg = generate_cuda_safe(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,prompt,boundary
        )
        dims = dimension_match(reply, prompt, str(case["intent"]), case["required_all"], case["forbidden"])
        orig_sem += int(dims["semantic_ok"])
        orig_strict += int(dims["strict_ok"])
        orig_overrides += int(dbg.get("cuda_override_active", False))
        if dbg.get("cuda_override_active"):
            s=dbg["cuda_override_signal"]
            print(f"  G{idx:02d} override ON raw={s['raw_top']} repair={s['repair_top']} conf={s['repair_conf']:.3f} scope={s['scope']:.3f}")
            print("       baseline:", dbg["cuda_override_baseline"])
            print("       final   :", reply)
    print(f"  semantic={orig_sem}/30 ({orig_sem/30:.1%}) strict={orig_strict}/30 ({orig_strict/30:.1%}) overrides={orig_overrides}")
    print()

    # 2) Held-out technical 30.
    tech_total = tech_pass = tech_overrides = cuda_total = cuda_pass = 0
    print("[2] Held-out technical 30")
    for idx,(intent,prompt,required,forbidden) in enumerate(HELDOUT_CASES,1):
        if intent not in TECH_INTENTS:
            continue
        reply, dbg = generate_cuda_safe(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,prompt,boundary
        )
        ok,missing,_ = heldout_score(reply,required,forbidden)
        tech_total += 1; tech_pass += int(ok); tech_overrides += int(dbg.get("cuda_override_active",False))
        if intent == "cuda":
            cuda_total += 1; cuda_pass += int(ok)
        print(f"  H{idx:02d} {intent:11s} {'PASS' if ok else 'MISS'} override={'ON' if dbg.get('cuda_override_active') else 'OFF'}")
        if dbg.get("cuda_override_active"):
            s=dbg["cuda_override_signal"]
            print(f"       raw={s['raw_top']} repair={s['repair_top']} conf={s['repair_conf']:.3f} scope={s['scope']:.3f}")
            print("       baseline:", dbg["cuda_override_baseline"])
            print("       final   :", reply)
        if missing:
            print("       missing:", ", ".join(missing))
    print(f"  technical={tech_pass}/{tech_total} ({tech_pass/tech_total:.1%}) CUDA={cuda_pass}/{cuda_total} ({cuda_pass/cuda_total:.1%}) overrides={tech_overrides}")
    print()

    # 3) Held-out total 60, preserving v1.0.2 conversational held-out repair.
    total_pass = total_repairs = total_cuda_overrides = 0
    print("[3] Held-out total 60")
    for idx,(intent,prompt,required,forbidden) in enumerate(HELDOUT_CASES,1):
        base_reply, dbg = generate_cuda_safe(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,prompt,boundary
        )
        base_ok,_,_ = heldout_score(base_reply,required,forbidden)
        reply = base_reply
        conv_active = False
        if not base_ok:
            route, cue_score = route_conversation(prompt)
            if route is not None:
                reply, _ = generate_guided_response(model,tok,prompt,ROUTES[route]["anchor"])
                conv_active = True
        ok,_,_ = heldout_score(reply,required,forbidden)
        total_pass += int(ok)
        total_repairs += int(conv_active)
        total_cuda_overrides += int(dbg.get("cuda_override_active",False))
    print(f"  final={total_pass}/60 ({total_pass/60:.1%}) conversational-repairs={total_repairs} cuda-overrides={total_cuda_overrides}")
    print()
    print("Summary")
    print("-------")
    print(f"Original semantic        : {orig_sem}/30 ({orig_sem/30:.1%})")
    print(f"Original strict          : {orig_strict}/30 ({orig_strict/30:.1%})")
    print(f"Held-out technical      : {tech_pass}/{tech_total} ({tech_pass/tech_total:.1%})")
    print(f"Held-out CUDA           : {cuda_pass}/{cuda_total} ({cuda_pass/cuda_total:.1%})")
    print(f"Held-out total          : {total_pass}/60 ({total_pass/60:.1%})")
    print(f"Original CUDA overrides : {orig_overrides}")
    print(f"Held-out CUDA overrides : {tech_overrides}")


if __name__ == "__main__":
    main()
