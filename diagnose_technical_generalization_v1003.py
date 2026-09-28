# diagnose_technical_generalization_v1003.py
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch

from evaluate_heldout_generalization_v1001 import CASES, score
from evaluate_transformer_continuation_category_repair_v01117 import generate as generate_v01117
from evaluate_post_entity_boundary_binding_v01115 import (
    DEFAULT_TOKENIZER, DEFAULT_MODEL, DEFAULT_INTENT, DEFAULT_ROLE,
    DEFAULT_BINDING, DEFAULT_REPAIR, build_boundary_block_ids,
)
from selective_intent_repair_v01110 import load_checkpoint as load_repair
from multi_concept_safe_binding_v0118 import CANONICAL, load_checkpoint as load_binding
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

TECH_INTENT_TO_LABEL = {
    "gpu": "tech_gpu",
    "cpu": "tech_cpu",
    "llm": "tech_llm",
    "transformer": "tech_transformer",
    "cuda": "tech_cuda",
    "python": "tech_python",
}
LABEL_TO_SHORT = {v: k for k, v in TECH_INTENT_TO_LABEL.items()}


def first_generated_entity(reply: str):
    hits = []
    lower = reply.lower()
    for label, canonical in CANONICAL.items():
        pos = lower.find(canonical.lower())
        if pos >= 0:
            hits.append((pos, label, canonical))
    if not hits:
        return None, None, None
    hits.sort(key=lambda x: x[0])
    return hits[0]


def classify_failure(expected_label, dbg, actual_label, semantic_ok):
    if semantic_ok:
        return "pass"
    chosen = dbg.get("chosen")
    if chosen != expected_label:
        return "recognition"
    if actual_label != expected_label:
        return "binding"
    return "generation"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--role-checkpoint", default=DEFAULT_ROLE)
    p.add_argument("--binding", default=DEFAULT_BINDING)
    p.add_argument("--repair", default=DEFAULT_REPAIR)
    p.add_argument("--csv", default="results/technical_generalization_v1003/diagnostic.csv")
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

    tech_cases = [
        (idx, intent, prompt, required, forbidden)
        for idx, (intent, prompt, required, forbidden) in enumerate(CASES, start=1)
        if intent in TECH_INTENT_TO_LABEL
    ]

    print("=" * 70)
    print(" LLM_GPU v1.0.3 Technical Semantic Generalization Diagnostic")
    print("=" * 70)
    print("Device:", device)
    print("Technical held-out cases:", len(tech_cases))
    print("Failure stages: recognition / binding / generation")
    print()

    rows = []
    stage_counts = Counter()
    per_intent = defaultdict(Counter)
    chosen_confusion = Counter()

    for idx, intent, prompt, required, forbidden in tech_cases:
        expected_label = TECH_INTENT_TO_LABEL[intent]
        expected_entity = CANONICAL[expected_label]

        reply, dbg = generate_v01117(
            model, tok, intent_head, intent_labels, role_head,
            binding, binding_ck, repair, prompt, boundary_block_ids
        )
        semantic_ok, missing, conflicts = score(reply, required, forbidden)
        pos, actual_label, actual_entity = first_generated_entity(reply)

        stage = classify_failure(expected_label, dbg, actual_label, semantic_ok)
        stage_counts[stage] += 1
        per_intent[intent][stage] += 1
        chosen_confusion[(expected_label, dbg.get("chosen"))] += 1

        raw_top = dbg.get("raw_top")
        repair_top = dbg.get("repair_top")
        chosen = dbg.get("chosen")

        print(f"[H{idx:02d}] expected={intent:11s} entity={expected_entity:11s} stage={stage}")
        print("      人:", prompt)
        print("      AI:", reply)
        print(
            f"      raw={LABEL_TO_SHORT.get(raw_top, raw_top)} "
            f"margin={dbg.get('raw_margin', 0.0):.3f} | "
            f"repair={LABEL_TO_SHORT.get(repair_top, repair_top)} "
            f"conf={dbg.get('repair_conf', 0.0):.3f}"
        )
        print(
            f"      scope={dbg.get('scope', 0.0):.3f} "
            f"chosen={LABEL_TO_SHORT.get(chosen, chosen)} "
            f"chosen-conf={dbg.get('chosen_conf', 0.0):.3f} "
            f"binding={'ON' if dbg.get('active') else 'OFF'}"
        )
        print(
            f"      actual-entity={actual_entity or 'NONE'} "
            f"entity-pos={pos if pos is not None else '-'} "
            f"semantic={'PASS' if semantic_ok else 'MISS'}"
        )
        if missing:
            print("      missing:", ", ".join(missing))
        if conflicts:
            print("      conflict:", ", ".join(conflicts))

        rows.append({
            "case": f"H{idx:02d}",
            "intent": intent,
            "prompt": prompt,
            "expected_label": expected_label,
            "expected_entity": expected_entity,
            "raw_top": raw_top,
            "raw_margin": dbg.get("raw_margin"),
            "repair_top": repair_top,
            "repair_conf": dbg.get("repair_conf"),
            "scope": dbg.get("scope"),
            "chosen": chosen,
            "chosen_conf": dbg.get("chosen_conf"),
            "repair_applied": dbg.get("repair_applied"),
            "binding_active": dbg.get("active"),
            "canonical": dbg.get("canonical"),
            "actual_label": actual_label,
            "actual_entity": actual_entity,
            "actual_entity_pos": pos,
            "semantic_ok": semantic_ok,
            "failure_stage": stage,
            "reply": reply,
            "missing": " | ".join(missing),
            "conflicts": " | ".join(conflicts),
        })

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    passes = stage_counts["pass"]
    failures = total - passes

    print()
    print("Summary")
    print("-------")
    print(f"Technical semantic pass : {passes}/{total} ({passes/total:.1%})")
    print(f"Failures                : {failures}/{total} ({failures/total:.1%})")
    print(f"Recognition failures    : {stage_counts['recognition']}")
    print(f"Binding failures        : {stage_counts['binding']}")
    print(f"Generation failures     : {stage_counts['generation']}")
    print()
    print("Per-intent failure stages")
    print("-------------------------")
    for intent in TECH_INTENT_TO_LABEL:
        c = per_intent[intent]
        print(
            f"{intent:11s} pass={c['pass']} "
            f"recognition={c['recognition']} "
            f"binding={c['binding']} "
            f"generation={c['generation']}"
        )

    print()
    print("Chosen-label confusion")
    print("----------------------")
    for (expected, chosen), count in sorted(
        chosen_confusion.items(),
        key=lambda x: (LABEL_TO_SHORT.get(x[0][0], x[0][0]), -x[1], str(x[0][1]))
    ):
        print(
            f"{LABEL_TO_SHORT.get(expected, expected):11s} -> "
            f"{LABEL_TO_SHORT.get(chosen, chosen):11s} : {count}"
        )

    print()
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
