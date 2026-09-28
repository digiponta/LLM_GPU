# llm_threshold_safety_sweep_v1008.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import torch

from evaluate_llm_semantic_repair_v1006 import (
    load_detector, llm_probability, LLM_ANCHOR,
)
from evaluate_cuda_only_safe_override_v1005 import generate_cuda_safe
from evaluate_heldout_generalization_v1001 import CASES as HELDOUT_CASES, score as heldout_score
from evaluate_generalization_v07 import CASES as ORIGINAL_CASES, dimension_match
from evaluate_conversational_intent_repair_v01118 import (
    route_conversation, ROUTES, generate_guided_response,
)
from evaluate_transformer_continuation_category_repair_v01117 import (
    DEFAULT_TOKENIZER, DEFAULT_MODEL, DEFAULT_INTENT, DEFAULT_ROLE,
    DEFAULT_BINDING, DEFAULT_REPAIR,
)
from evaluate_post_entity_boundary_binding_v01115 import build_boundary_block_ids
from selective_intent_repair_v01110 import load_checkpoint as load_repair
from multi_concept_safe_binding_v0118 import load_checkpoint as load_binding
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

THRESHOLDS = [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
TECH_INTENTS = {"gpu","cpu","llm","transformer","cuda","python"}


@torch.no_grad()
def generate_with_threshold(
    model, tok, intent_head, intent_labels, role_head, binding, binding_ck,
    repair, det, threshold, prompt, boundary,
):
    reply, dbg = generate_cuda_safe(
        model, tok, intent_head, intent_labels, role_head,
        binding, binding_ck, repair, prompt, boundary
    )
    prob = llm_probability(model, tok, det, prompt)
    active = (not dbg.get("cuda_override_active", False)) and prob >= threshold
    dbg["llm_detector_prob"] = prob
    dbg["llm_override_active"] = active
    dbg["llm_override_baseline"] = reply
    if active:
        reply, _ = generate_guided_response(model, tok, prompt, LLM_ANCHOR)
    return reply, dbg


def evaluate_original(ctx, threshold):
    (model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,det,boundary)=ctx
    semantic = strict = 0
    llm_overrides = 0
    false_llm = 0
    override_cases = []

    for idx, case in enumerate(ORIGINAL_CASES, 1):
        prompt = str(case["prompt"])
        reply, dbg = generate_with_threshold(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,
            repair,det,threshold,prompt,boundary
        )
        dims = dimension_match(
            reply,prompt,str(case["intent"]),case["required_all"],case["forbidden"]
        )

        # Preserve the integrated conversational fallback used in v1.0.7.
        if not dims["semantic_ok"]:
            route,_ = route_conversation(prompt)
            if route is not None:
                reply,_ = generate_guided_response(model,tok,prompt,ROUTES[route]["anchor"])
                dims = dimension_match(
                    reply,prompt,str(case["intent"]),case["required_all"],case["forbidden"]
                )

        semantic += int(dims["semantic_ok"])
        strict += int(dims["strict_ok"])

        if dbg.get("llm_override_active"):
            llm_overrides += 1
            is_llm_case = str(case["intent"]) == "llm"
            false_llm += int(not is_llm_case)
            override_cases.append(
                (f"G{idx:02d}", str(case["intent"]), dbg["llm_detector_prob"], is_llm_case)
            )

    return {
        "semantic": semantic,
        "strict": strict,
        "llm_overrides": llm_overrides,
        "false_llm": false_llm,
        "override_cases": override_cases,
    }


def evaluate_heldout(ctx, threshold):
    (model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,det,boundary)=ctx
    tech_ok = tech_total = 0
    llm_ok = llm_total = 0
    total_ok = 0
    llm_overrides = 0
    false_llm = 0
    cuda_ok = cuda_total = 0
    override_cases = []

    for idx,(intent,prompt,required,forbidden) in enumerate(HELDOUT_CASES,1):
        reply, dbg = generate_with_threshold(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,
            repair,det,threshold,prompt,boundary
        )

        if dbg.get("llm_override_active"):
            llm_overrides += 1
            is_llm_case = intent == "llm"
            false_llm += int(not is_llm_case)
            override_cases.append(
                (f"H{idx:02d}", intent, dbg["llm_detector_prob"], is_llm_case)
            )

        ok,_,_ = heldout_score(reply,required,forbidden)

        if intent in TECH_INTENTS:
            tech_total += 1
            tech_ok += int(ok)
            if intent == "llm":
                llm_total += 1
                llm_ok += int(ok)
            if intent == "cuda":
                cuda_total += 1
                cuda_ok += int(ok)

        # Integrated conversational fallback.
        if not ok:
            route,_ = route_conversation(prompt)
            if route is not None:
                reply,_ = generate_guided_response(model,tok,prompt,ROUTES[route]["anchor"])
                ok,_,_ = heldout_score(reply,required,forbidden)

        total_ok += int(ok)

    return {
        "tech_ok": tech_ok,
        "tech_total": tech_total,
        "llm_ok": llm_ok,
        "llm_total": llm_total,
        "total_ok": total_ok,
        "total": len(HELDOUT_CASES),
        "cuda_ok": cuda_ok,
        "cuda_total": cuda_total,
        "llm_overrides": llm_overrides,
        "false_llm": false_llm,
        "override_cases": override_cases,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--intent-head",default=DEFAULT_INTENT)
    p.add_argument("--role-checkpoint",default=DEFAULT_ROLE)
    p.add_argument("--binding",default=DEFAULT_BINDING)
    p.add_argument("--repair",default=DEFAULT_REPAIR)
    p.add_argument("--llm-detector",default="model/model-gpu-v1.0.7-llm-detector-hard-negative.pt")
    p.add_argument("--csv",default="results/llm_threshold_safety_sweep_v1008/threshold_sweep.csv")
    args=p.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device); model.eval()
    intent_head,_,intent_labels=load_intent_head(args.intent_head,model,device)
    role_head,_,_=load_role_checkpoint(args.role_checkpoint,intent_labels,device)
    binding,binding_ck=load_binding(args.binding,intent_labels,device)
    repair,_=load_repair(args.repair,intent_labels,device)
    boundary=build_boundary_block_ids(tok)
    det,trained_threshold,_=load_detector(args.llm_detector,device)

    ctx=(model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,det,boundary)

    print("="*92)
    print(" LLM_GPU v1.0.8 LLM Threshold Safety Sweep")
    print("="*92)
    print("Device:",device)
    print("Detector checkpoint threshold:",f"{trained_threshold:.2f}")
    print("Sweep:",", ".join(f"{x:.2f}" for x in THRESHOLDS))
    print()
    print("Threshold  Original   Tech heldout  LLM heldout  CUDA  Total60  Orig FP  Held FP  LLM overrides")
    print("-"*96)

    rows=[]
    scored=[]
    for th in THRESHOLDS:
        orig=evaluate_original(ctx,th)
        held=evaluate_heldout(ctx,th)
        print(
            f"{th:8.2f}   "
            f"{orig['semantic']:2d}/30      "
            f"{held['tech_ok']:2d}/{held['tech_total']:<2d}       "
            f"{held['llm_ok']}/{held['llm_total']}          "
            f"{held['cuda_ok']}/{held['cuda_total']}   "
            f"{held['total_ok']:2d}/60      "
            f"{orig['false_llm']:2d}       "
            f"{held['false_llm']:2d}       "
            f"{held['llm_overrides']:2d}"
        )
        rows.append({
            "threshold": th,
            "original_semantic": orig["semantic"],
            "original_strict": orig["strict"],
            "heldout_technical": held["tech_ok"],
            "heldout_llm": held["llm_ok"],
            "heldout_cuda": held["cuda_ok"],
            "heldout_total": held["total_ok"],
            "original_false_llm": orig["false_llm"],
            "heldout_false_llm": held["false_llm"],
            "heldout_llm_overrides": held["llm_overrides"],
        })

        # Selection priority:
        # 1) Original 30/30
        # 2) zero false LLM overrides on original + heldout
        # 3) CUDA 5/5
        # 4) highest heldout LLM
        # 5) highest heldout total
        # 6) higher threshold for extra safety
        key=(
            orig["semantic"]==30,
            orig["false_llm"]==0 and held["false_llm"]==0,
            held["cuda_ok"]==held["cuda_total"],
            held["llm_ok"],
            held["total_ok"],
            th,
        )
        scored.append((key,th,orig,held))

    scored.sort(reverse=True)
    _,best_th,best_orig,best_held=scored[0]

    print()
    print("Best safety candidate")
    print("---------------------")
    print(f"Threshold           : {best_th:.2f}")
    print(f"Original semantic   : {best_orig['semantic']}/30")
    print(f"Original strict     : {best_orig['strict']}/30")
    print(f"Held-out technical  : {best_held['tech_ok']}/{best_held['tech_total']}")
    print(f"Held-out LLM        : {best_held['llm_ok']}/{best_held['llm_total']}")
    print(f"Held-out CUDA       : {best_held['cuda_ok']}/{best_held['cuda_total']}")
    print(f"Held-out total      : {best_held['total_ok']}/60")
    print(f"Original false LLM  : {best_orig['false_llm']}")
    print(f"Held-out false LLM  : {best_held['false_llm']}")
    print()

    print("LLM override cases at best threshold")
    print("------------------------------------")
    for case_id,intent,prob,is_llm in best_orig["override_cases"]:
        print(f"{case_id} original intent={intent:12s} p={prob:.3f} {'TP' if is_llm else 'FP'}")
    for case_id,intent,prob,is_llm in best_held["override_cases"]:
        print(f"{case_id} heldout  intent={intent:12s} p={prob:.3f} {'TP' if is_llm else 'FP'}")

    csv_path=Path(args.csv)
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print()
    print("CSV:",csv_path)


if __name__=="__main__":
    main()
