# run_selective_repair_threshold_sweep_v01111.py
from __future__ import annotations

import csv
from pathlib import Path

import torch

from evaluate_selective_intent_repair_v01110 import (
    DIRECT,
    generate,
)
from selective_intent_repair_v01110 import load_checkpoint as load_repair
from multi_concept_safe_binding_v0118 import load_checkpoint as load_binding
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from evaluate_generalization_v07 import CASES, dimension_match
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

TOKENIZER = "model/tokenizer-v0.7-bpe.json"
MODEL = "model/model-gpu-v0.8-chat-clean.pt"
INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"
ROLE = "model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt"
BINDING = "model/model-gpu-v0.11.8-multi-concept-safe-binding.pt"
REPAIR = "model/model-gpu-v0.11.10-selective-intent-repair.pt"

RESULTS = Path("results/selective_repair_threshold_sweep_v01111")
SUMMARY_CSV = RESULTS / "threshold_sweep.csv"
DETAIL_CSV = RESULTS / "threshold_sweep_detail.csv"

REPAIR_MARGINS = (0.20, 0.25, 0.30)
SCOPE_THRESHOLDS = (0.50, 0.60, 0.70)
BINDING_CONFIDENCES = (0.45, 0.55, 0.70)

# Mandatory preservation guards.
GUARD_DIRECT = {5: "CPU", 9: "CUDA", 27: "GPU", 28: "CPU"}
TARGET_REPAIR = {7: "LLM", 8: "Transformer", 10: "Python"}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    for filename in (TOKENIZER, MODEL, INTENT, ROLE, BINDING, REPAIR):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(TOKENIZER)
    model, base_ck = LanguageModel.load_checkpoint(MODEL, device=device)
    model.eval()
    intent_head, intent_ck, intent_labels = load_intent_head(INTENT, model, device)
    role_head, _old_binding, role_ck = load_role_checkpoint(ROLE, intent_labels, device)
    binding, binding_ck = load_binding(BINDING, intent_labels, device)
    repair, repair_ck_base = load_repair(REPAIR, intent_labels, device)

    print("====================================================")
    print(" Selective Repair Threshold Sweep v0.11.11")
    print("====================================================")
    print("Device:", device)
    print("Repair margins      :", REPAIR_MARGINS)
    print("Scope thresholds    :", SCOPE_THRESHOLDS)
    print("Binding confidences :", BINDING_CONFIDENCES)
    print("Combinations        :", len(REPAIR_MARGINS) * len(SCOPE_THRESHOLDS) * len(BINDING_CONFIDENCES))
    print()

    summaries = []
    details = []

    for repair_margin in REPAIR_MARGINS:
        for scope_threshold in SCOPE_THRESHOLDS:
            for binding_confidence in BINDING_CONFIDENCES:
                repair_ck = dict(repair_ck_base)
                repair_ck["repair_margin_threshold"] = repair_margin
                repair_ck["scope_threshold"] = scope_threshold
                repair_ck["binding_confidence_threshold"] = binding_confidence

                semantic = strict = fluency = 0
                direct_hits = 0
                replies = {}
                dbgs = {}

                for idx, case in enumerate(CASES, start=1):
                    prompt = str(case["prompt"])
                    reply, dbg = generate(
                        model,
                        tokenizer,
                        intent_head,
                        intent_labels,
                        role_head,
                        binding,
                        binding_ck,
                        repair,
                        repair_ck,
                        prompt,
                    )
                    dims = dimension_match(
                        reply,
                        prompt,
                        str(case["intent"]),
                        case["required_all"],
                        case["forbidden"],
                    )
                    semantic += int(dims["semantic_ok"])
                    strict += int(dims["strict_ok"])
                    fluency += int(dims["fluent_ok"])
                    replies[idx] = reply
                    dbgs[idx] = dbg

                    if idx in DIRECT:
                        direct_hits += int(reply.startswith(DIRECT[idx]))

                    details.append({
                        "repair_margin": repair_margin,
                        "scope_threshold": scope_threshold,
                        "binding_confidence": binding_confidence,
                        "case": idx,
                        "intent": str(case["intent"]),
                        "reply": reply,
                        "semantic_ok": int(dims["semantic_ok"]),
                        "strict_ok": int(dims["strict_ok"]),
                        "raw_top": dbg["raw_top"],
                        "raw_margin": dbg["raw_margin"],
                        "repair_top": dbg["repair_top"],
                        "repair_conf": dbg["repair_conf"],
                        "chosen": dbg["chosen"],
                        "chosen_conf": dbg["chosen_conf"],
                        "scope": dbg["scope"],
                        "repair_applied": int(dbg["repair_applied"]),
                        "binding_active": int(dbg["active"]),
                    })

                guard_direct_ok = all(
                    replies[i].startswith(expected)
                    for i, expected in GUARD_DIRECT.items()
                )
                g21_safe = not dbgs[21]["active"]
                g30_safe = (
                    not dbgs[30]["active"]
                    and "LLM" in replies[30]
                    and "Transformer" in replies[30]
                )
                guards_ok = guard_direct_ok and g21_safe and g30_safe

                target_hits = sum(
                    int(replies[i].startswith(expected))
                    for i, expected in TARGET_REPAIR.items()
                )

                # Prefer valid guards first, then target repairs, then strict,
                # semantic, direct. Smaller thresholds are not rewarded.
                score = (
                    int(guards_ok),
                    target_hits,
                    strict,
                    semantic,
                    direct_hits,
                )

                summaries.append({
                    "repair_margin": repair_margin,
                    "scope_threshold": scope_threshold,
                    "binding_confidence": binding_confidence,
                    "guards_ok": int(guards_ok),
                    "g05_ok": int(replies[5].startswith("CPU")),
                    "g07_ok": int(replies[7].startswith("LLM")),
                    "g08_ok": int(replies[8].startswith("Transformer")),
                    "g09_ok": int(replies[9].startswith("CUDA")),
                    "g10_ok": int(replies[10].startswith("Python")),
                    "g21_safe": int(g21_safe),
                    "g27_ok": int(replies[27].startswith("GPU")),
                    "g28_ok": int(replies[28].startswith("CPU")),
                    "g30_safe": int(g30_safe),
                    "target_hits": target_hits,
                    "semantic": semantic,
                    "strict": strict,
                    "fluency": fluency,
                    "direct_hits": direct_hits,
                    "_score": score,
                })

                print(
                    f"margin={repair_margin:.2f} scope={scope_threshold:.2f} conf={binding_confidence:.2f} "
                    f"| guards={'PASS' if guards_ok else 'FAIL'} "
                    f"| G07/G08/G10={target_hits}/3 "
                    f"| sem={semantic:02d} strict={strict:02d} direct={direct_hits}/7 "
                    f"| G21={'SAFE' if g21_safe else 'ON'}"
                )

    best = max(summaries, key=lambda row: row["_score"])

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        fields = [k for k in summaries[0].keys() if k != "_score"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summaries:
            w.writerow({k: row[k] for k in fields})

    with DETAIL_CSV.open("w", encoding="utf-8", newline="") as f:
        fields = list(details[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(details)

    print()
    print("Best configuration")
    print("------------------")
    print(f"repair_margin       : {best['repair_margin']:.2f}")
    print(f"scope_threshold     : {best['scope_threshold']:.2f}")
    print(f"binding_confidence  : {best['binding_confidence']:.2f}")
    print(f"guards              : {'PASS' if best['guards_ok'] else 'FAIL'}")
    print(f"G05 CPU             : {'PASS' if best['g05_ok'] else 'MISS'}")
    print(f"G07 LLM             : {'PASS' if best['g07_ok'] else 'MISS'}")
    print(f"G08 Transformer     : {'PASS' if best['g08_ok'] else 'MISS'}")
    print(f"G09 CUDA            : {'PASS' if best['g09_ok'] else 'MISS'}")
    print(f"G10 Python          : {'PASS' if best['g10_ok'] else 'MISS'}")
    print(f"G21 false binding   : {'NO' if best['g21_safe'] else 'YES'}")
    print(f"G27 GPU             : {'PASS' if best['g27_ok'] else 'MISS'}")
    print(f"G28 CPU             : {'PASS' if best['g28_ok'] else 'MISS'}")
    print(f"G30 protected       : {'PASS' if best['g30_safe'] else 'MISS'}")
    print(f"Semantic-content    : {best['semantic']}/30 ({best['semantic']/30:.1%})")
    print(f"Strict              : {best['strict']}/30 ({best['strict']/30:.1%})")
    print(f"Fluency             : {best['fluency']}/30 ({best['fluency']/30:.1%})")
    print(f"Tracked direct      : {best['direct_hits']}/7 ({best['direct_hits']/7:.1%})")
    print()
    print("Summary CSV:", SUMMARY_CSV)
    print("Detail CSV :", DETAIL_CSV)


if __name__ == "__main__":
    main()
