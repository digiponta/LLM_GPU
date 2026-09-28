# repair_override_policy_sweep_v1004.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from evaluate_heldout_generalization_v1001 import CASES as HELDOUT_CASES
from evaluate_generalization_v07 import CASES as ORIGINAL_CASES
from evaluate_post_entity_boundary_binding_v01115 import (
    DEFAULT_TOKENIZER, DEFAULT_MODEL, DEFAULT_INTENT, DEFAULT_REPAIR,
    REPAIR_MARGIN, REPAIR_CONFIDENCE,
)
from selective_intent_repair_v01110 import (
    TECH_LABELS, extract_technical_logits, load_checkpoint as load_repair,
)
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

INTENT_TO_LABEL = {
    "gpu": "tech_gpu",
    "cpu": "tech_cpu",
    "llm": "tech_llm",
    "transformer": "tech_transformer",
    "cuda": "tech_cuda",
    "python": "tech_python",
}

POLICIES = [
    "P0_current",
    "P1_conf_050",
    "P1_conf_055",
    "P1_conf_060",
    "P1_conf_065",
    "P1_conf_070",
    "P2_scope070_conf050",
    "P2_scope070_conf055",
    "P2_scope070_conf060",
    "P2_scope080_conf055",
    "P2_scope080_conf060",
    "P3_disagree_conf050",
    "P3_disagree_conf055",
    "P3_disagree_conf060",
    "P4_concept_specific",
]


def choose(policy, raw_top, raw_margin, rep_top, rep_conf, scope):
    # P0: current v0.11.10 policy.
    if policy == "P0_current":
        if raw_margin <= REPAIR_MARGIN and rep_top != raw_top and rep_conf >= REPAIR_CONFIDENCE:
            return rep_top, True
        return raw_top, False

    # P1: confidence-only override.
    if policy.startswith("P1_conf_"):
        th = int(policy.rsplit("_", 1)[1]) / 100.0
        if rep_top != raw_top and rep_conf >= th:
            return rep_top, True
        return raw_top, False

    # P2: technical scope + repaired confidence.
    if policy.startswith("P2_scope"):
        parts = policy.split("_")
        scope_th = int(parts[1].replace("scope", "")) / 100.0
        conf_th = int(parts[2].replace("conf", "")) / 100.0
        if rep_top != raw_top and scope >= scope_th and rep_conf >= conf_th:
            return rep_top, True
        return raw_top, False

    # P3: disagreement is required, with a confidence threshold.
    if policy.startswith("P3_disagree_conf"):
        th = int(policy.rsplit("conf", 1)[1]) / 100.0
        if rep_top != raw_top and rep_conf >= th:
            return rep_top, True
        return raw_top, False

    # P4: lower thresholds only for the two concepts strongly suggested by
    # v1.0.3 diagnostics (LLM/CUDA); conservative elsewhere.
    if policy == "P4_concept_specific":
        thresholds = {
            "tech_llm": 0.45,
            "tech_cuda": 0.50,
            "tech_gpu": 0.65,
            "tech_cpu": 0.65,
            "tech_transformer": 0.60,
            "tech_python": 0.60,
        }
        th = thresholds[rep_top]
        # Require moderate technical scope except for LLM, whose held-out scope
        # was frequently low despite correct repair direction.
        scope_ok = True if rep_top == "tech_llm" else scope >= 0.50
        if rep_top != raw_top and rep_conf >= th and scope_ok:
            return rep_top, True
        return raw_top, False

    raise ValueError(policy)


@torch.no_grad()
def features(model, tok, intent_head, intent_labels, repair, prompt_text):
    device = next(model.parameters()).device
    prompt = f"人: {prompt_text}\nAI: "
    ids = tok.encode(prompt, add_bos=True)
    x = torch.tensor([ids[-model.context_length:]], dtype=torch.long, device=device)
    h = model.forward_hidden(x)[:, -1, :]
    intent_logits = intent_head(h)
    raw = extract_technical_logits(intent_logits, intent_labels)
    repaired, scope_logit = repair(raw)

    order = torch.argsort(raw[0], descending=True)
    top = int(order[0].item())
    second = int(order[1].item())
    raw_top = TECH_LABELS[top]
    raw_margin = float((raw[0, top] - raw[0, second]).item())

    rep_probs = torch.softmax(repaired, dim=-1)[0]
    rep_idx = int(torch.argmax(rep_probs).item())
    rep_top = TECH_LABELS[rep_idx]
    rep_conf = float(rep_probs[rep_idx].item())
    scope = float(torch.sigmoid(scope_logit)[0].item())
    return raw_top, raw_margin, rep_top, rep_conf, scope


def collect_cases():
    heldout = []
    for idx, (intent, prompt, required, forbidden) in enumerate(HELDOUT_CASES, 1):
        if intent in INTENT_TO_LABEL:
            heldout.append(("heldout", f"H{idx:02d}", intent, prompt, INTENT_TO_LABEL[intent]))

    original = []
    for idx, case in enumerate(ORIGINAL_CASES, 1):
        intent = str(case["intent"])
        if intent in INTENT_TO_LABEL:
            original.append(("original", f"G{idx:02d}", intent, str(case["prompt"]), INTENT_TO_LABEL[intent]))
    return heldout, original


def evaluate_set(rows, policies):
    out = {}
    for policy in policies:
        correct = overrides = wrong_overrides = helpful_overrides = 0
        details = []
        for row in rows:
            expected = row["expected"]
            chosen, overridden = choose(
                policy,
                row["raw_top"], row["raw_margin"],
                row["rep_top"], row["rep_conf"], row["scope"],
            )
            ok = chosen == expected
            correct += int(ok)
            overrides += int(overridden)
            if overridden:
                raw_ok = row["raw_top"] == expected
                if ok and not raw_ok:
                    helpful_overrides += 1
                if not ok:
                    wrong_overrides += 1
            details.append((chosen, overridden, ok))
        out[policy] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows) if rows else 0.0,
            "overrides": overrides,
            "helpful_overrides": helpful_overrides,
            "wrong_overrides": wrong_overrides,
            "details": details,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--repair", default=DEFAULT_REPAIR)
    p.add_argument("--csv", default="results/repair_override_policy_sweep_v1004/policy_sweep.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    intent_head, _, intent_labels = load_intent_head(args.intent_head, model, device)
    repair, _ = load_repair(args.repair, intent_labels, device)

    held_defs, original_defs = collect_cases()
    all_defs = held_defs + original_defs

    rows = []
    for dataset, case_id, intent, prompt, expected in all_defs:
        raw_top, raw_margin, rep_top, rep_conf, scope = features(
            model, tok, intent_head, intent_labels, repair, prompt
        )
        rows.append({
            "dataset": dataset,
            "case": case_id,
            "intent": intent,
            "prompt": prompt,
            "expected": expected,
            "raw_top": raw_top,
            "raw_margin": raw_margin,
            "rep_top": rep_top,
            "rep_conf": rep_conf,
            "scope": scope,
        })

    held_rows = [r for r in rows if r["dataset"] == "heldout"]
    original_rows = [r for r in rows if r["dataset"] == "original"]
    held_eval = evaluate_set(held_rows, POLICIES)
    orig_eval = evaluate_set(original_rows, POLICIES)

    print("=" * 78)
    print(" LLM_GPU v1.0.4 Repair Override Policy Sweep")
    print("=" * 78)
    print("Device:", device)
    print("Held-out technical cases:", len(held_rows))
    print("Original technical guard cases:", len(original_rows))
    print("Policies:", len(POLICIES))
    print()
    print("Policy                         Held-out       Original       Overrides  Helpful  Wrong")
    print("-" * 86)

    ranked = []
    for policy in POLICIES:
        h = held_eval[policy]
        o = orig_eval[policy]
        guard_ok = o["correct"] == len(original_rows)
        ranked.append((guard_ok, h["correct"], -h["wrong_overrides"], policy))
        print(
            f"{policy:30s} "
            f"{h['correct']:2d}/{h['total']:<2d} {h['accuracy']:6.1%}   "
            f"{o['correct']:2d}/{o['total']:<2d} {o['accuracy']:6.1%}   "
            f"{h['overrides']:3d}        "
            f"{h['helpful_overrides']:3d}      "
            f"{h['wrong_overrides']:3d}"
        )

    ranked.sort(reverse=True)
    best_guard = next((x for x in ranked if x[0]), None)
    if best_guard is None:
        # Fall back to the strongest held-out policy if no policy preserves
        # every original technical selection.
        best = max(ranked, key=lambda x: (x[1], x[2]))
        reason = "no policy preserved all original technical selections"
    else:
        best = best_guard
        reason = "preserves all original technical selections"

    best_policy = best[3]
    print()
    print("Best guarded candidate")
    print("----------------------")
    print("Policy:", best_policy)
    print("Reason:", reason)
    print(
        f"Held-out: {held_eval[best_policy]['correct']}/{len(held_rows)} "
        f"({held_eval[best_policy]['accuracy']:.1%})"
    )
    print(
        f"Original : {orig_eval[best_policy]['correct']}/{len(original_rows)} "
        f"({orig_eval[best_policy]['accuracy']:.1%})"
    )

    # Detailed changes relative to P0.
    print()
    print("Changes vs P0_current on held-out")
    print("---------------------------------")
    p0 = held_eval["P0_current"]["details"]
    pb = held_eval[best_policy]["details"]
    for row, base_d, best_d in zip(held_rows, p0, pb):
        if base_d[0] != best_d[0]:
            print(
                f"{row['case']} {row['intent']:11s} "
                f"expected={row['expected']:16s} "
                f"P0={base_d[0]:16s} -> {best_d[0]:16s} "
                f"{'FIX' if best_d[2] and not base_d[2] else 'REGRESSION' if base_d[2] and not best_d[2] else 'CHANGE'}"
            )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "policy", "heldout_correct", "heldout_total", "heldout_accuracy",
            "original_correct", "original_total", "original_accuracy",
            "heldout_overrides", "heldout_helpful_overrides", "heldout_wrong_overrides",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for policy in POLICIES:
            h = held_eval[policy]
            o = orig_eval[policy]
            w.writerow({
                "policy": policy,
                "heldout_correct": h["correct"],
                "heldout_total": h["total"],
                "heldout_accuracy": h["accuracy"],
                "original_correct": o["correct"],
                "original_total": o["total"],
                "original_accuracy": o["accuracy"],
                "heldout_overrides": h["overrides"],
                "heldout_helpful_overrides": h["helpful_overrides"],
                "heldout_wrong_overrides": h["wrong_overrides"],
            })

    print()
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
