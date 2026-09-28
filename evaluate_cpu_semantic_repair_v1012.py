# evaluate_cpu_semantic_repair_v1012.py
from __future__ import annotations

import argparse
import torch

from train_llm_semantic_detector_v1006 import LLMDetector as BinaryDetector
from evaluate_cpu_gpu_semantic_repair_v1010 import load_hardware_detector, hardware_prediction
from evaluate_python_semantic_repair_v1009 import load_binary_detector, binary_probability
from evaluate_llm_semantic_repair_v1006 import load_detector, llm_probability
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

LLM_THRESHOLD=0.70
HARDWARE_THRESHOLD=0.93
LLM_ANCHOR="LLMは文章を学習して生成する言語モデルです。"
PYTHON_ANCHOR="Pythonは読みやすい汎用プログラミング言語です。"
CPU_ANCHOR="CPUは汎用処理や制御を担当します。"
GPU_ANCHOR="GPUは大量の並列計算を得意とします。"
TECH_INTENTS={"gpu","cpu","llm","transformer","cuda","python"}


def load_cpu_detector(filename,device):
    ck=torch.load(filename,map_location=device)
    det=BinaryDetector(int(ck["d_model"]),int(ck.get("hidden",32))).to(device)
    det.load_state_dict(ck["state_dict"]); det.eval()
    return det,float(ck["threshold"]),ck


@torch.no_grad()
def cpu_probability(model,tok,det,prompt):
    device=next(model.parameters()).device
    ids=tok.encode(f"人: {prompt}\nAI: ",add_bos=True)
    x=torch.tensor([ids[-model.context_length:]],dtype=torch.long,device=device)
    h=model.forward_hidden(x)[:,-1,:]
    return float(torch.sigmoid(det(h))[0].item())


@torch.no_grad()
def generate_combined(model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
                      llm_det,python_det,python_th,hw_det,cpu_det,cpu_th,prompt,boundary):
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
        and hw_conf>=HARDWARE_THRESHOLD
    )
    if hw_active:
        reply,_=generate_guided_response(
            model,tok,prompt,CPU_ANCHOR if hw_label=="cpu" else GPU_ANCHOR
        )

    cpu_p=cpu_probability(model,tok,cpu_det,prompt)
    cpu_active=(
        not dbg.get("cuda_override_active",False)
        and not llm_active
        and not py_active
        and not hw_active
        and cpu_p>=cpu_th
    )
    if cpu_active:
        reply,_=generate_guided_response(model,tok,prompt,CPU_ANCHOR)

    dbg.update({
        "llm_active":llm_active,
        "python_active":py_active,
        "hardware_label":hw_label,
        "hardware_conf":hw_conf,
        "hardware_active":hw_active,
        "cpu_prob":cpu_p,
        "cpu_active":cpu_active,
    })
    return reply,dbg


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

    print("="*80)
    print(" LLM_GPU v1.0.12 CPU Hard-Case Recognition Repair")
    print("="*80)
    print("Device:",device)
    print(f"LLM threshold fixed     : {LLM_THRESHOLD:.2f}")
    print(f"Python threshold        : {python_th:.2f}")
    print(f"Hardware threshold fixed: {HARDWARE_THRESHOLD:.2f}")
    print(f"CPU detector threshold  : {cpu_th:.2f}")
    print()

    orig_sem=orig_strict=orig_cpu_fp=0
    print("[1] Original 30-case integrated guard")
    for idx,case in enumerate(ORIGINAL_CASES,1):
        prompt=str(case["prompt"])
        reply,dbg=generate_combined(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
            llm_det,python_det,python_th,hw_det,cpu_det,cpu_th,prompt,boundary
        )
        dims=dimension_match(reply,prompt,str(case["intent"]),case["required_all"],case["forbidden"])
        if not dims["semantic_ok"]:
            route,_=route_conversation(prompt)
            if route is not None:
                reply,_=generate_guided_response(model,tok,prompt,ROUTES[route]["anchor"])
                dims=dimension_match(reply,prompt,str(case["intent"]),case["required_all"],case["forbidden"])
        orig_sem+=int(dims["semantic_ok"]); orig_strict+=int(dims["strict_ok"])
        if dbg["cpu_active"] and str(case["intent"])!="cpu":
            orig_cpu_fp+=1
        if dbg["cpu_active"]:
            print(f"  G{idx:02d} intent={case['intent']} cpuP={dbg['cpu_prob']:.3f} CPU=ON")
            print("       final:",reply)
    print(f"  semantic={orig_sem}/30 ({orig_sem/30:.1%}) strict={orig_strict}/30 ({orig_strict/30:.1%}) cpu-FP={orig_cpu_fp}")
    print()

    tech=tech_ok=cpu_n=cpu_ok=gpu_n=gpu_ok=py_n=py_ok=cuda_n=cuda_ok=cpu_overrides=cpu_fp=0
    print("[2] Held-out technical 30")
    for idx,(intent,prompt,required,forbidden) in enumerate(HELDOUT_CASES,1):
        if intent not in TECH_INTENTS: continue
        reply,dbg=generate_combined(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
            llm_det,python_det,python_th,hw_det,cpu_det,cpu_th,prompt,boundary
        )
        ok,missing,_=heldout_score(reply,required,forbidden)
        tech+=1; tech_ok+=int(ok)
        if intent=="cpu": cpu_n+=1; cpu_ok+=int(ok)
        if intent=="gpu": gpu_n+=1; gpu_ok+=int(ok)
        if intent=="python": py_n+=1; py_ok+=int(ok)
        if intent=="cuda": cuda_n+=1; cuda_ok+=int(ok)
        if dbg["cpu_active"]:
            cpu_overrides+=1
            if intent!="cpu": cpu_fp+=1
        print(f"  H{idx:02d} {intent:11s} {'PASS' if ok else 'MISS'} cpuP={dbg['cpu_prob']:.3f} CPU={'ON' if dbg['cpu_active'] else 'OFF'} HW={'ON' if dbg['hardware_active'] else 'OFF'}")
        if dbg["cpu_active"]:
            print("       final:",reply)
        if missing: print("       missing:",", ".join(missing))

    print(f"  technical={tech_ok}/{tech} ({tech_ok/tech:.1%}) CPU={cpu_ok}/{cpu_n} GPU={gpu_ok}/{gpu_n} Python={py_ok}/{py_n} CUDA={cuda_ok}/{cuda_n}")
    print(f"  cpu-overrides={cpu_overrides} cpu-FP={cpu_fp}")
    print()

    total_ok=0
    for intent,prompt,required,forbidden in HELDOUT_CASES:
        reply,dbg=generate_combined(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
            llm_det,python_det,python_th,hw_det,cpu_det,cpu_th,prompt,boundary
        )
        ok,_,_=heldout_score(reply,required,forbidden)
        if not ok:
            route,_=route_conversation(prompt)
            if route is not None:
                reply,_=generate_guided_response(model,tok,prompt,ROUTES[route]["anchor"])
                ok,_,_=heldout_score(reply,required,forbidden)
        total_ok+=int(ok)

    print("[3] Held-out total 60")
    print(f"  final={total_ok}/60 ({total_ok/60:.1%})")
    print()
    print("Summary")
    print("-------")
    print(f"Original semantic      : {orig_sem}/30 ({orig_sem/30:.1%})")
    print(f"Original strict        : {orig_strict}/30 ({orig_strict/30:.1%})")
    print(f"Original CPU FP        : {orig_cpu_fp}")
    print(f"Held-out technical    : {tech_ok}/{tech} ({tech_ok/tech:.1%})")
    print(f"Held-out CPU          : {cpu_ok}/{cpu_n} ({cpu_ok/cpu_n:.1%})")
    print(f"Held-out GPU          : {gpu_ok}/{gpu_n} ({gpu_ok/gpu_n:.1%})")
    print(f"Held-out Python       : {py_ok}/{py_n} ({py_ok/py_n:.1%})")
    print(f"Held-out CUDA         : {cuda_ok}/{cuda_n} ({cuda_ok/cuda_n:.1%})")
    print(f"Held-out CPU FP       : {cpu_fp}")
    print(f"Held-out total        : {total_ok}/60 ({total_ok/60:.1%})")

if __name__=="__main__":
    main()
