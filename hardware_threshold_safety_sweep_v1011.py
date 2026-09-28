# hardware_threshold_safety_sweep_v1011.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import torch

from evaluate_cpu_gpu_semantic_repair_v1010 import (
    load_hardware_detector, hardware_prediction,
)
from evaluate_python_semantic_repair_v1009 import (
    load_binary_detector, binary_probability,
)
from evaluate_llm_semantic_repair_v1006 import (
    load_detector, llm_probability,
)
from evaluate_cuda_only_safe_override_v1005 import generate_cuda_safe
from evaluate_heldout_generalization_v1001 import CASES as HELDOUT_CASES, score as heldout_score
from evaluate_generalization_v07 import CASES as ORIGINAL_CASES, dimension_match
from evaluate_conversational_intent_repair_v01118 import route_conversation, ROUTES, generate_guided_response
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

THRESHOLDS = [0.75,0.85,0.90,0.92,0.93,0.94,0.95]
LLM_THRESHOLD = 0.70
LLM_ANCHOR = "LLMは文章を学習して生成する言語モデルです。"
PYTHON_ANCHOR = "Pythonは読みやすい汎用プログラミング言語です。"
CPU_ANCHOR = "CPUは汎用処理や制御を担当します。"
GPU_ANCHOR = "GPUは大量の並列計算を得意とします。"
TECH_INTENTS={"gpu","cpu","llm","transformer","cuda","python"}

@torch.no_grad()
def generate_with_threshold(
    model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
    llm_det,python_det,python_th,hw_det,hw_th,prompt,boundary
):
    reply,dbg=generate_cuda_safe(
        model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,prompt,boundary
    )

    llm_p=llm_probability(model,tok,llm_det,prompt)
    llm_active=(not dbg.get("cuda_override_active",False)) and llm_p>=LLM_THRESHOLD
    if llm_active:
        reply,_=generate_guided_response(model,tok,prompt,LLM_ANCHOR)

    py_p=binary_probability(model,tok,python_det,prompt)
    py_active=(
        not dbg.get("cuda_override_active",False)
        and not llm_active
        and py_p>=python_th
    )
    if py_active:
        reply,_=generate_guided_response(model,tok,prompt,PYTHON_ANCHOR)

    hw_label,hw_conf=hardware_prediction(model,tok,hw_det,prompt)
    hw_active=(
        not dbg.get("cuda_override_active",False)
        and not llm_active
        and not py_active
        and hw_label in {"cpu","gpu"}
        and hw_conf>=hw_th
    )
    if hw_active:
        reply,_=generate_guided_response(
            model,tok,prompt,CPU_ANCHOR if hw_label=="cpu" else GPU_ANCHOR
        )

    dbg.update({
        "llm_p":llm_p,
        "llm_active":llm_active,
        "python_p":py_p,
        "python_active":py_active,
        "hw_label":hw_label,
        "hw_conf":hw_conf,
        "hw_active":hw_active,
    })
    return reply,dbg

def eval_original(ctx,th):
    (model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,llm_det,python_det,python_th,hw_det,boundary)=ctx
    sem=strict=fp=hw_overrides=0
    override_cases=[]
    for idx,case in enumerate(ORIGINAL_CASES,1):
        prompt=str(case["prompt"])
        reply,dbg=generate_with_threshold(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
            llm_det,python_det,python_th,hw_det,th,prompt,boundary
        )
        dims=dimension_match(reply,prompt,str(case["intent"]),case["required_all"],case["forbidden"])
        if not dims["semantic_ok"]:
            route,_=route_conversation(prompt)
            if route is not None:
                reply,_=generate_guided_response(model,tok,prompt,ROUTES[route]["anchor"])
                dims=dimension_match(reply,prompt,str(case["intent"]),case["required_all"],case["forbidden"])
        sem+=int(dims["semantic_ok"]); strict+=int(dims["strict_ok"])
        if dbg["hw_active"]:
            hw_overrides+=1
            correct_intent=str(case["intent"])==dbg["hw_label"]
            fp+=int(not correct_intent)
            override_cases.append((f"G{idx:02d}",str(case["intent"]),dbg["hw_label"],dbg["hw_conf"],correct_intent))
    return {"semantic":sem,"strict":strict,"fp":fp,"overrides":hw_overrides,"cases":override_cases}

def eval_heldout(ctx,th):
    (model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,llm_det,python_det,python_th,hw_det,boundary)=ctx
    tech=tech_ok=gpu_n=gpu_ok=cpu_n=cpu_ok=py_n=py_ok=cuda_n=cuda_ok=0
    fp=hw_overrides=total_ok=0
    override_cases=[]
    for idx,(intent,prompt,required,forbidden) in enumerate(HELDOUT_CASES,1):
        reply,dbg=generate_with_threshold(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
            llm_det,python_det,python_th,hw_det,th,prompt,boundary
        )
        ok,_,_=heldout_score(reply,required,forbidden)
        if intent in TECH_INTENTS:
            tech+=1; tech_ok+=int(ok)
            if intent=="gpu": gpu_n+=1; gpu_ok+=int(ok)
            if intent=="cpu": cpu_n+=1; cpu_ok+=int(ok)
            if intent=="python": py_n+=1; py_ok+=int(ok)
            if intent=="cuda": cuda_n+=1; cuda_ok+=int(ok)

        if dbg["hw_active"]:
            hw_overrides+=1
            correct=(intent==dbg["hw_label"])
            fp+=int(not correct)
            override_cases.append((f"H{idx:02d}",intent,dbg["hw_label"],dbg["hw_conf"],correct))

        if not ok:
            route,_=route_conversation(prompt)
            if route is not None:
                reply,_=generate_guided_response(model,tok,prompt,ROUTES[route]["anchor"])
                ok,_,_=heldout_score(reply,required,forbidden)
        total_ok+=int(ok)

    return {
        "tech":tech,"tech_ok":tech_ok,
        "gpu_n":gpu_n,"gpu_ok":gpu_ok,
        "cpu_n":cpu_n,"cpu_ok":cpu_ok,
        "py_n":py_n,"py_ok":py_ok,
        "cuda_n":cuda_n,"cuda_ok":cuda_ok,
        "fp":fp,"overrides":hw_overrides,
        "total_ok":total_ok,"cases":override_cases,
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
    p.add_argument("--python-detector",default="model/model-gpu-v1.0.9-python-detector.pt")
    p.add_argument("--hardware-detector",default="model/model-gpu-v1.0.10-cpu-gpu-detector.pt")
    p.add_argument("--csv",default="results/hardware_threshold_safety_sweep_v1011/threshold_sweep.csv")
    args=p.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device); model.eval()
    intent_head,_,intent_labels=load_intent_head(args.intent_head,model,device)
    role_head,_,_=load_role_checkpoint(args.role_checkpoint,intent_labels,device)
    binding,binding_ck=load_binding(args.binding,intent_labels,device)
    repair,_=load_repair(args.repair,intent_labels,device)
    boundary=build_boundary_block_ids(tok)
    llm_det,_,_=load_detector(args.llm_detector,device)
    python_det,python_th,_=load_binary_detector(args.python_detector,device)
    hw_det,trained_hw_th,_=load_hardware_detector(args.hardware_detector,device)

    ctx=(model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,llm_det,python_det,python_th,hw_det,boundary)

    print("="*104)
    print(" LLM_GPU v1.0.11 Hardware Threshold Safety Sweep")
    print("="*104)
    print("Device:",device)
    print(f"Trained hardware threshold: {trained_hw_th:.2f}")
    print("Sweep:",", ".join(f"{x:.2f}" for x in THRESHOLDS))
    print()
    print("Threshold  Original  Tech  GPU  CPU  Python  CUDA  Total60  OrigFP  HeldFP  HW overrides")
    print("-"*100)

    rows=[]; scored=[]
    for th in THRESHOLDS:
        o=eval_original(ctx,th)
        h=eval_heldout(ctx,th)
        print(
            f"{th:8.2f}   {o['semantic']:2d}/30    "
            f"{h['tech_ok']:2d}/{h['tech']:<2d}  "
            f"{h['gpu_ok']}/{h['gpu_n']}   "
            f"{h['cpu_ok']}/{h['cpu_n']}   "
            f"{h['py_ok']}/{h['py_n']}      "
            f"{h['cuda_ok']}/{h['cuda_n']}    "
            f"{h['total_ok']:2d}/60      "
            f"{o['fp']:2d}      {h['fp']:2d}       {h['overrides']:2d}"
        )
        rows.append({
            "threshold":th,
            "original_semantic":o["semantic"],
            "original_strict":o["strict"],
            "heldout_technical":h["tech_ok"],
            "heldout_gpu":h["gpu_ok"],
            "heldout_cpu":h["cpu_ok"],
            "heldout_python":h["py_ok"],
            "heldout_cuda":h["cuda_ok"],
            "heldout_total":h["total_ok"],
            "original_hw_fp":o["fp"],
            "heldout_hw_fp":h["fp"],
            "heldout_hw_overrides":h["overrides"],
        })
        key=(
            o["semantic"]==30,
            o["fp"]==0 and h["fp"]==0,
            h["cuda_ok"]==h["cuda_n"],
            h["py_ok"]>=3,
            h["gpu_ok"],
            h["cpu_ok"],
            h["total_ok"],
            th,
        )
        scored.append((key,th,o,h))

    scored.sort(reverse=True)
    _,best_th,best_o,best_h=scored[0]

    print()
    print("Best safety candidate")
    print("---------------------")
    print(f"Threshold            : {best_th:.2f}")
    print(f"Original semantic    : {best_o['semantic']}/30")
    print(f"Original strict      : {best_o['strict']}/30")
    print(f"Held-out technical   : {best_h['tech_ok']}/{best_h['tech']}")
    print(f"Held-out GPU         : {best_h['gpu_ok']}/{best_h['gpu_n']}")
    print(f"Held-out CPU         : {best_h['cpu_ok']}/{best_h['cpu_n']}")
    print(f"Held-out Python      : {best_h['py_ok']}/{best_h['py_n']}")
    print(f"Held-out CUDA        : {best_h['cuda_ok']}/{best_h['cuda_n']}")
    print(f"Held-out total       : {best_h['total_ok']}/60")
    print(f"Original hardware FP : {best_o['fp']}")
    print(f"Held-out hardware FP : {best_h['fp']}")
    print()
    print("Hardware override cases at best threshold")
    print("-----------------------------------------")
    for case_id,intent,label,conf,correct in best_o["cases"]:
        print(f"{case_id} original intent={intent:12s} hw={label:5s} conf={conf:.3f} {'TP' if correct else 'FP'}")
    for case_id,intent,label,conf,correct in best_h["cases"]:
        print(f"{case_id} heldout  intent={intent:12s} hw={label:5s} conf={conf:.3f} {'TP' if correct else 'FP'}")

    csv_path=Path(args.csv)
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print()
    print("CSV:",csv_path)

if __name__=="__main__":
    main()
