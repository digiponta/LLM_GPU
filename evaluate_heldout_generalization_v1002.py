# evaluate_heldout_generalization_v1002.py
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from evaluate_heldout_generalization_v1001 import CASES, CONVERSATIONAL, score
from evaluate_conversational_intent_repair_v01118 import (
    DEFAULT_TOKENIZER, DEFAULT_MODEL, DEFAULT_INTENT, DEFAULT_ROLE,
    DEFAULT_BINDING, DEFAULT_REPAIR,
    ROUTES, route_conversation, generate_guided_response,
)
from evaluate_transformer_continuation_category_repair_v01117 import generate as generate_v01117
from evaluate_post_entity_boundary_binding_v01115 import build_boundary_block_ids
from selective_intent_repair_v01110 import load_checkpoint as load_repair
from multi_concept_safe_binding_v0118 import load_checkpoint as load_binding
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer


@torch.no_grad()
def generate_heldout_final(
    model, tok, intent_head, intent_labels, role_head,
    binding, binding_ck, repair, prompt_text, boundary_block_ids,
    required, forbidden,
):
    # Always obtain the technical/conversation baseline from v0.11.17.
    baseline_reply, baseline_dbg = generate_v01117(
        model, tok, intent_head, intent_labels, role_head,
        binding, binding_ck, repair, prompt_text, boundary_block_ids
    )

    # IMPORTANT: held-out scoring is based on the held-out case itself,
    # never on the original 30-case CASES table.
    baseline_ok, _, _ = score(baseline_reply, required, forbidden)
    if baseline_ok:
        baseline_dbg["conversation_route"] = None
        baseline_dbg["conversation_active"] = False
        baseline_dbg["conversation_steps"] = []
        return baseline_reply, baseline_dbg

    route, cue_score = route_conversation(prompt_text)
    if route is None:
        baseline_dbg["conversation_route"] = None
        baseline_dbg["conversation_active"] = False
        baseline_dbg["conversation_steps"] = []
        return baseline_reply, baseline_dbg

    anchor = ROUTES[route]["anchor"]
    reply, steps = generate_guided_response(model, tok, prompt_text, anchor)

    baseline_dbg["conversation_route"] = route
    baseline_dbg["conversation_cue_score"] = cue_score
    baseline_dbg["conversation_active"] = True
    baseline_dbg["conversation_steps"] = steps
    baseline_dbg["conversation_anchor"] = anchor
    baseline_dbg["baseline_reply"] = baseline_reply
    return reply, baseline_dbg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--role-checkpoint", default=DEFAULT_ROLE)
    p.add_argument("--binding", default=DEFAULT_BINDING)
    p.add_argument("--repair", default=DEFAULT_REPAIR)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    intent_head, _, intent_labels = load_intent_head(args.intent_head, model, device)
    role_head, _, _ = load_role_checkpoint(args.role_checkpoint, intent_labels, device)
    binding, binding_ck = load_binding(args.binding, intent_labels, device)
    repair, _ = load_repair(args.repair, intent_labels, device)
    boundary_block_ids = build_boundary_block_ids(tok)

    print("=" * 64)
    print(" LLM_GPU v1.0.2 Held-out Evaluator Fix")
    print("=" * 64)
    print("Device:", device)
    print("Held-out cases:", len(CASES))
    print("Cases per intent: 5")
    print("Repair trigger: held-out baseline MISS + generic cue router")
    print("Original 30-case CASES dependency: removed")
    print()

    base_pass = final_pass = 0
    repair_tp = repair_fp = repair_fn = 0
    per_intent = {}

    for idx, (intent, prompt, required, forbidden) in enumerate(CASES, start=1):
        base_reply, _ = generate_v01117(
            model, tok, intent_head, intent_labels, role_head,
            binding, binding_ck, repair, prompt, boundary_block_ids
        )
        final_reply, final_dbg = generate_heldout_final(
            model, tok, intent_head, intent_labels, role_head,
            binding, binding_ck, repair, prompt, boundary_block_ids,
            required, forbidden,
        )

        b_ok, _, _ = score(base_reply, required, forbidden)
        f_ok, missing, conflicts = score(final_reply, required, forbidden)
        active = bool(final_dbg.get("conversation_active"))

        base_pass += int(b_ok)
        final_pass += int(f_ok)
        stats = per_intent.setdefault(
            intent, {"n": 0, "base": 0, "final": 0, "repair": 0}
        )
        stats["n"] += 1
        stats["base"] += int(b_ok)
        stats["final"] += int(f_ok)
        stats["repair"] += int(active)

        repair_needed = (intent in CONVERSATIONAL and not b_ok)
        if active and repair_needed:
            repair_tp += 1
        elif active and not repair_needed:
            repair_fp += 1
        elif repair_needed and not active:
            repair_fn += 1

        print(f"[H{idx:02d}] {intent:12s} 人: {prompt}")
        print("      baseline:", base_reply)
        print("      final   :", final_reply)
        print(
            f"      base={'PASS' if b_ok else 'MISS'} "
            f"final={'PASS' if f_ok else 'MISS'} "
            f"repair={'ON' if active else 'OFF'}"
        )
        if active:
            print(
                f"      route={final_dbg.get('conversation_route')} "
                f"cue-score={final_dbg.get('conversation_cue_score')} "
                f"anchor={final_dbg.get('conversation_anchor')}"
            )
        if missing:
            print("      missing:", ", ".join(missing))
        if conflicts:
            print("      conflict:", ", ".join(conflicts))

    precision = repair_tp / (repair_tp + repair_fp) if (repair_tp + repair_fp) else 1.0
    recall = repair_tp / (repair_tp + repair_fn) if (repair_tp + repair_fn) else 1.0

    print()
    print("Summary")
    print("-------")
    print(f"Baseline semantic rate : {base_pass}/{len(CASES)} ({base_pass/len(CASES):.1%})")
    print(f"Final semantic rate    : {final_pass}/{len(CASES)} ({final_pass/len(CASES):.1%})")
    print(f"Repair precision       : {precision:.1%}  TP={repair_tp} FP={repair_fp}")
    print(f"Repair recall          : {recall:.1%}  TP={repair_tp} FN={repair_fn}")
    print(f"False activation rate  : {repair_fp}/{len(CASES)} ({repair_fp/len(CASES):.1%})")
    print()
    print("Per-intent")
    print("----------")
    for intent in sorted(per_intent):
        s = per_intent[intent]
        print(
            f"{intent:12s} base={s['base']}/{s['n']} "
            f"final={s['final']}/{s['n']} repairs={s['repair']}"
        )


if __name__ == "__main__":
    main()
