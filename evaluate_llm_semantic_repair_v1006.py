# evaluate_llm_semantic_repair_v1006.py
from __future__ import annotations

import argparse
import torch

from train_llm_semantic_detector_v1006 import LLMDetector
from evaluate_cuda_only_safe_override_v1005 import (
    generate_cuda_safe, CUDA_SCOPE_THRESHOLD, CUDA_REPAIR_CONFIDENCE,
)
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

LLM_ANCHOR="LLMは文章を学習して生成する言語モデルです。"
TECH_INTENTS={"gpu","cpu","llm","transformer","cuda","python"}


def load_detector(filename,device):
    ck=torch.load(filename,map_location=device)
    det=LLMDetector(int(ck["d_model"]),int(ck.get("hidden",32))).to(device)
    det.load_state_dict(ck["state_dict"]); det.eval()
    return det,float(ck["threshold"]),ck


@torch.no_grad()
def llm_probability(model,tok,det,prompt_text):
    device=next(model.parameters()).device
    ids=tok.encode(f"人: {prompt_text}\nAI: ",add_bos=True)
    x=torch.tensor([ids[-model.context_length:]],dtype=torch.long,device=device)
    h=model.forward_hidden(x)[:,-1,:]
    return float(torch.sigmoid(det(h))[0].item())


@torch.no_grad()
def generate_combined(model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,det,llm_th,prompt,boundary):
    reply,dbg=generate_cuda_safe(
        model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,prompt,boundary
    )
    p=llm_probability(model,tok,det,prompt)
    llm_active=(not dbg.get("cuda_override_active",False)) and p>=llm_th
    dbg["llm_detector_prob"]=p
    dbg["llm_override_active"]=llm_active
    dbg["llm_override_baseline"]=reply
    if llm_active:
        reply,steps=generate_guided_response(model,tok,prompt,LLM_ANCHOR)
        dbg["llm_override_steps"]=steps
    return reply,dbg


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--intent-head",default=DEFAULT_INTENT)
    p.add_argument("--role-checkpoint",default=DEFAULT_ROLE)
    p.add_argument("--binding",default=DEFAULT_BINDING)
    p.add_argument("--repair",default=DEFAULT_REPAIR)
    p.add_argument("--llm-detector",default="model/model-gpu-v1.0.6-llm-detector.pt")
    args=p.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device); model.eval()
    intent_head,_,intent_labels=load_intent_head(args.intent_head,model,device)
    role_head,_,_=load_role_checkpoint(args.role_checkpoint,intent_labels,device)
    binding,binding_ck=load_binding(args.binding,intent_labels,device)
    repair,_=load_repair(args.repair,intent_labels,device)
    boundary=build_boundary_block_ids(tok)
    det,llm_th,_=load_detector(args.llm_detector,device)

    print("="*74)
    print(" LLM_GPU v1.0.6 LLM Semantic Recognition Repair")
    print("="*74)
    print("Device:",device)
    print(f"LLM detector threshold: {llm_th:.2f}")
    print(f"CUDA rule: scope>={CUDA_SCOPE_THRESHOLD:.2f} repair_conf>={CUDA_REPAIR_CONFIDENCE:.2f}")
    print()

    orig_sem=orig_strict=orig_llm=orig_cuda=0
    print("[1] Original 30-case integrated guard")
    for idx,case in enumerate(ORIGINAL_CASES,1):
        prompt=str(case["prompt"])
        base,dbg=generate_combined(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,det,llm_th,prompt,boundary
        )
        dims=dimension_match(base,prompt,str(case["intent"]),case["required_all"],case["forbidden"])
        reply=base
        if not dims["semantic_ok"]:
            route,_=route_conversation(prompt)
            if route is not None:
                reply,_=generate_guided_response(model,tok,prompt,ROUTES[route]["anchor"])
                dims=dimension_match(reply,prompt,str(case["intent"]),case["required_all"],case["forbidden"])
        orig_sem+=int(dims["semantic_ok"]); orig_strict+=int(dims["strict_ok"])
        orig_llm+=int(dbg.get("llm_override_active",False)); orig_cuda+=int(dbg.get("cuda_override_active",False))
        if dbg.get("llm_override_active") or dbg.get("cuda_override_active"):
            print(f"  G{idx:02d} intent={case['intent']} llmP={dbg['llm_detector_prob']:.3f} LLM={'ON' if dbg.get('llm_override_active') else 'OFF'} CUDA={'ON' if dbg.get('cuda_override_active') else 'OFF'}")
            print("       final:",reply)
    print(f"  semantic={orig_sem}/30 ({orig_sem/30:.1%}) strict={orig_strict}/30 ({orig_strict/30:.1%}) llm-overrides={orig_llm} cuda-overrides={orig_cuda}")
    print()

    tech=tech_ok=llm_n=llm_ok=llm_overrides=cuda_overrides=0
    print("[2] Held-out technical 30")
    for idx,(intent,prompt,required,forbidden) in enumerate(HELDOUT_CASES,1):
        if intent not in TECH_INTENTS: continue
        reply,dbg=generate_combined(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,det,llm_th,prompt,boundary
        )
        ok,missing,_=heldout_score(reply,required,forbidden)
        tech+=1; tech_ok+=int(ok)
        if intent=="llm": llm_n+=1; llm_ok+=int(ok)
        llm_overrides+=int(dbg.get("llm_override_active",False)); cuda_overrides+=int(dbg.get("cuda_override_active",False))
        print(f"  H{idx:02d} {intent:11s} {'PASS' if ok else 'MISS'} llmP={dbg['llm_detector_prob']:.3f} LLM={'ON' if dbg.get('llm_override_active') else 'OFF'} CUDA={'ON' if dbg.get('cuda_override_active') else 'OFF'}")
        if dbg.get("llm_override_active") or dbg.get("cuda_override_active"):
            print("       baseline:",dbg.get("llm_override_baseline",dbg.get("cuda_override_baseline","")))
            print("       final   :",reply)
        if missing: print("       missing:",", ".join(missing))
    print(f"  technical={tech_ok}/{tech} ({tech_ok/tech:.1%}) LLM={llm_ok}/{llm_n} ({llm_ok/llm_n:.1%}) llm-overrides={llm_overrides} cuda-overrides={cuda_overrides}")
    print()

    total_ok=conv_rep=0
    for idx,(intent,prompt,required,forbidden) in enumerate(HELDOUT_CASES,1):
        reply,dbg=generate_combined(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,det,llm_th,prompt,boundary
        )
        ok,_,_=heldout_score(reply,required,forbidden)
        if not ok:
            route,_=route_conversation(prompt)
            if route is not None:
                reply,_=generate_guided_response(model,tok,prompt,ROUTES[route]["anchor"])
                conv_rep+=1
                ok,_,_=heldout_score(reply,required,forbidden)
        total_ok+=int(ok)

    print("[3] Held-out total 60")
    print(f"  final={total_ok}/60 ({total_ok/60:.1%}) conversational-repairs={conv_rep}")
    print()
    print("Summary")
    print("-------")
    print(f"Original integrated semantic : {orig_sem}/30 ({orig_sem/30:.1%})")
    print(f"Original integrated strict   : {orig_strict}/30 ({orig_strict/30:.1%})")
    print(f"Held-out technical          : {tech_ok}/{tech} ({tech_ok/tech:.1%})")
    print(f"Held-out LLM                : {llm_ok}/{llm_n} ({llm_ok/llm_n:.1%})")
    print(f"Held-out total              : {total_ok}/60 ({total_ok/60:.1%})")
    print(f"LLM overrides (technical)   : {llm_overrides}")
    print(f"CUDA overrides (technical)  : {cuda_overrides}")

if __name__=="__main__":
    main()
