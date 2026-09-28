# evaluate_fresh_generalization_v114.py
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import csv
import torch

from fresh_generalization_cases_v114 import CASES
from evaluate_deterministic_relation_response_v113 import generate_combined
from evaluate_llm_second_stage_repair_v110 import load_second_stage
from evaluate_transformer_semantic_repair_v111 import load_transformer_detector
from evaluate_cpu_semantic_repair_v1012 import load_cpu_detector
from evaluate_cpu_gpu_semantic_repair_v1010 import load_hardware_detector
from evaluate_python_semantic_repair_v1009 import load_binary_detector
from evaluate_llm_semantic_repair_v1006 import load_detector
from evaluate_conversational_intent_repair_v01118 import route_conversation, ROUTES, generate_guided_response
from evaluate_transformer_continuation_category_repair_v01117 import (
    DEFAULT_TOKENIZER, DEFAULT_MODEL, DEFAULT_INTENT, DEFAULT_ROLE, DEFAULT_BINDING, DEFAULT_REPAIR,
)
from evaluate_post_entity_boundary_binding_v01115 import build_boundary_block_ids
from selective_intent_repair_v01110 import load_checkpoint as load_repair
from multi_concept_safe_binding_v0118 import load_checkpoint as load_binding
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

TECH={"gpu","cpu","llm","transformer","cuda","python"}


def score(text,required,forbidden):
    missing=["/".join(g) for g in required if not any(t in text for t in g)]
    conflicts=[t for t in forbidden if t in text]
    return (not missing and not conflicts),missing,conflicts


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
    p.add_argument("--cpu-detector",default="model/model-gpu-v1.0.12-cpu-detector.pt")
    p.add_argument("--llm2-detector",default="model/model-gpu-v1.1.0-llm-second-stage.pt")
    p.add_argument("--transformer-detector",default="model/model-gpu-v1.1.1-transformer-detector.pt")
    p.add_argument("--csv",default="results/fresh_generalization_v114/fresh_v2.csv")
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
    hw_det,_,_=load_hardware_detector(args.hardware_detector,device)
    cpu_det,cpu_th,_=load_cpu_detector(args.cpu_detector,device)
    llm2_det,llm2_th,_=load_second_stage(args.llm2_detector,device)
    tr_det,tr_th,_=load_transformer_detector(args.transformer_detector,device)

    print("="*88)
    print(" LLM_GPU v1.1.4 Fresh Generalization Test v2")
    print("="*88)
    print("Device:",device)
    print("Fresh cases:",len(CASES))
    print("Intents: 12 x 10")
    print("Policy: v1.1.3 frozen; no retraining or threshold tuning")
    print()

    stats=defaultdict(lambda:{"n":0,"pass":0})
    total=tech_n=tech_ok=0
    rows=[]
    for idx,(intent,prompt,required,forbidden) in enumerate(CASES,1):
        reply,dbg=generate_combined(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
            llm_det,python_det,python_th,hw_det,cpu_det,cpu_th,llm2_det,llm2_th,
            tr_det,tr_th,prompt,boundary
        )
        ok,missing,conflicts=score(reply,required,forbidden)

        # Preserve the same conversational fallback used in prior integrated evaluations.
        route=None
        if not ok:
            route,_=route_conversation(prompt)
            if route is not None:
                reply,_=generate_guided_response(model,tok,prompt,ROUTES[route]["anchor"])
                ok,missing,conflicts=score(reply,required,forbidden)

        stats[intent]["n"]+=1
        stats[intent]["pass"]+=int(ok)
        total+=int(ok)
        if intent in TECH:
            tech_n+=1; tech_ok+=int(ok)

        rows.append({
            "id":f"F{idx:03d}","intent":intent,"prompt":prompt,
            "pass":int(ok),"reply":reply,"missing":" | ".join(missing),
            "route":route or "",
            "relation":int(bool(dbg.get("relation_active"))),
            "llm":int(bool(dbg.get("llm_active"))),
            "python":int(bool(dbg.get("python_active"))),
            "hardware":int(bool(dbg.get("hardware_active"))),
            "cpu":int(bool(dbg.get("cpu_active"))),
            "llm2":int(bool(dbg.get("llm2_active"))),
            "transformer":int(bool(dbg.get("transformer_active"))),
        })
        print(f"F{idx:03d} {intent:12s} {'PASS' if ok else 'MISS'}")
        if not ok:
            print("     prompt :",prompt)
            print("     reply  :",reply)
            if missing: print("     missing:",", ".join(missing))

    print()
    print("Per-intent")
    print("----------")
    for intent in ["gpu","cpu","llm","transformer","cuda","python","short","topic","repeat","compare","error","end"]:
        s=stats[intent]
        print(f"{intent:12s} {s['pass']:2d}/{s['n']:2d} ({s['pass']/s['n']:.1%})")

    print()
    print("Summary")
    print("-------")
    print(f"Fresh technical : {tech_ok}/{tech_n} ({tech_ok/tech_n:.1%})")
    print(f"Fresh total     : {total}/{len(CASES)} ({total/len(CASES):.1%})")

    out=Path(args.csv); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("CSV:",out)

if __name__=="__main__":
    main()
