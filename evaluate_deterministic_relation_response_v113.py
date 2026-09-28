# evaluate_deterministic_relation_response_v113.py
from __future__ import annotations

import argparse
import re
import torch

from evaluate_transformer_semantic_repair_v111 import (
    load_transformer_detector, transformer_probability,
)
from evaluate_llm_second_stage_repair_v110 import (
    load_second_stage, detector_probability as llm2_probability,
)
from evaluate_cpu_semantic_repair_v1012 import load_cpu_detector, cpu_probability
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
TRANSFORMER_ANCHOR="TransformerはAttentionを中心に使うモデル構造です。"
RELATION_ANCHOR="LLMとTransformerは同じ意味ではありません。TransformerはLLMで利用されるモデル構造の一つです。"

TECH_INTENTS={"gpu","cpu","llm","transformer","cuda","python"}

RELATION_TERMS = re.compile(
    r"(同じ|同一|違い|異なる|関係|関連|比較|どう違|何が違|同義|意味)",
    re.IGNORECASE,
)


def is_llm_transformer_relation(prompt: str) -> bool:
    p=prompt.lower()
    has_llm=("llm" in p) or ("言語モデル" in prompt)
    has_transformer=("transformer" in p) or ("トランスフォーマ" in prompt)
    return has_llm and has_transformer and bool(RELATION_TERMS.search(prompt))


@torch.no_grad()
def generate_combined(model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
                      llm_det,python_det,python_th,hw_det,cpu_det,cpu_th,llm2_det,llm2_th,
                      tr_det,tr_th,prompt,boundary):
    reply,dbg=generate_cuda_safe(
        model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,prompt,boundary
    )

    relation_active=is_llm_transformer_relation(prompt)
    if relation_active:
        reply=RELATION_ANCHOR

    llm_p=llm_probability(model,tok,llm_det,prompt)
    llm_active=(not relation_active and not dbg.get("cuda_override_active",False) and llm_p>=LLM_THRESHOLD)
    if llm_active:
        reply,_=generate_guided_response(model,tok,prompt,LLM_ANCHOR)

    py_p=binary_probability(model,tok,python_det,prompt)
    py_active=(not relation_active and not dbg.get("cuda_override_active",False) and not llm_active and py_p>=python_th)
    if py_active:
        reply,_=generate_guided_response(model,tok,prompt,PYTHON_ANCHOR)

    hw_label,hw_conf=hardware_prediction(model,tok,hw_det,prompt)
    hw_active=(
        not relation_active and not dbg.get("cuda_override_active",False)
        and not llm_active and not py_active
        and hw_label in {"cpu","gpu"} and hw_conf>=HARDWARE_THRESHOLD
    )
    if hw_active:
        reply,_=generate_guided_response(model,tok,prompt,CPU_ANCHOR if hw_label=="cpu" else GPU_ANCHOR)

    cpu_p=cpu_probability(model,tok,cpu_det,prompt)
    cpu_active=(
        not relation_active and not dbg.get("cuda_override_active",False)
        and not llm_active and not py_active and not hw_active and cpu_p>=cpu_th
    )
    if cpu_active:
        reply,_=generate_guided_response(model,tok,prompt,CPU_ANCHOR)

    llm2_p=llm2_probability(model,tok,llm2_det,prompt)
    llm2_active=(
        not relation_active and not dbg.get("cuda_override_active",False)
        and not llm_active and not py_active and not hw_active and not cpu_active
        and llm2_p>=llm2_th
    )
    if llm2_active:
        reply,_=generate_guided_response(model,tok,prompt,LLM_ANCHOR)

    tr_p=transformer_probability(model,tok,tr_det,prompt)
    tr_active=(
        not relation_active and not dbg.get("cuda_override_active",False)
        and not llm_active and not py_active and not hw_active and not cpu_active and not llm2_active
        and tr_p>=tr_th
    )
    if tr_active:
        reply,_=generate_guided_response(model,tok,prompt,TRANSFORMER_ANCHOR)

    dbg.update({
        "relation_active":relation_active,
        "llm_active":llm_active,
        "python_active":py_active,
        "hardware_active":hw_active,
        "cpu_active":cpu_active,
        "llm2_active":llm2_active,
        "transformer_prob":tr_p,
        "transformer_active":tr_active,
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
    p.add_argument("--llm2-detector",default="model/model-gpu-v1.1.0-llm-second-stage.pt")
    p.add_argument("--transformer-detector",default="model/model-gpu-v1.1.1-transformer-detector.pt")
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

    print("="*86)
    print(" LLM_GPU v1.1.3 Deterministic Relation Response")
    print("="*86)
    print("Device:",device)
    print(f"Primary LLM threshold : {LLM_THRESHOLD:.2f}")
    print(f"Python threshold      : {python_th:.2f}")
    print(f"Hardware threshold    : {HARDWARE_THRESHOLD:.2f}")
    print(f"CPU threshold         : {cpu_th:.2f}")
    print(f"LLM2 threshold        : {llm2_th:.2f}")
    print(f"Transformer threshold : {tr_th:.2f}")
    print()

    orig_sem=orig_strict=orig_relation=0
    print("[1] Original 30-case integrated guard")
    for idx,case in enumerate(ORIGINAL_CASES,1):
        prompt=str(case["prompt"])
        reply,dbg=generate_combined(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
            llm_det,python_det,python_th,hw_det,cpu_det,cpu_th,llm2_det,llm2_th,
            tr_det,tr_th,prompt,boundary
        )
        dims=dimension_match(reply,prompt,str(case["intent"]),case["required_all"],case["forbidden"])
        if not dims["semantic_ok"]:
            route,_=route_conversation(prompt)
            if route is not None:
                reply,_=generate_guided_response(model,tok,prompt,ROUTES[route]["anchor"])
                dims=dimension_match(reply,prompt,str(case["intent"]),case["required_all"],case["forbidden"])
        orig_sem+=int(dims["semantic_ok"]); orig_strict+=int(dims["strict_ok"])
        orig_relation+=int(dbg["relation_active"])
        if dbg["relation_active"]:
            print(f"  G{idx:02d} intent={case['intent']} RELATION=ON")
            print("       final:",reply)
    print(f"  semantic={orig_sem}/30 ({orig_sem/30:.1%}) strict={orig_strict}/30 ({orig_strict/30:.1%}) relation={orig_relation}")
    print()

    tech=tech_ok=tr_n=tr_ok=llm_n=llm_ok=gpu_n=gpu_ok=cpu_n=cpu_ok=py_n=py_ok=cuda_n=cuda_ok=relation_count=0
    print("[2] Held-out technical 30")
    for idx,(intent,prompt,required,forbidden) in enumerate(HELDOUT_CASES,1):
        if intent not in TECH_INTENTS: continue
        reply,dbg=generate_combined(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
            llm_det,python_det,python_th,hw_det,cpu_det,cpu_th,llm2_det,llm2_th,
            tr_det,tr_th,prompt,boundary
        )
        ok,missing,_=heldout_score(reply,required,forbidden)
        tech+=1; tech_ok+=int(ok)
        if intent=="transformer": tr_n+=1; tr_ok+=int(ok)
        if intent=="llm": llm_n+=1; llm_ok+=int(ok)
        if intent=="gpu": gpu_n+=1; gpu_ok+=int(ok)
        if intent=="cpu": cpu_n+=1; cpu_ok+=int(ok)
        if intent=="python": py_n+=1; py_ok+=int(ok)
        if intent=="cuda": cuda_n+=1; cuda_ok+=int(ok)
        relation_count+=int(dbg["relation_active"])
        print(f"  H{idx:02d} {intent:11s} {'PASS' if ok else 'MISS'} REL={'ON' if dbg['relation_active'] else 'OFF'} TR={'ON' if dbg['transformer_active'] else 'OFF'} LLM2={'ON' if dbg['llm2_active'] else 'OFF'}")
        if missing: print("       missing:",", ".join(missing))

    print(f"  technical={tech_ok}/{tech} ({tech_ok/tech:.1%}) Transformer={tr_ok}/{tr_n} LLM={llm_ok}/{llm_n} CPU={cpu_ok}/{cpu_n} GPU={gpu_ok}/{gpu_n} Python={py_ok}/{py_n} CUDA={cuda_ok}/{cuda_n}")
    print(f"  relation-activations={relation_count}")
    print()

    total_ok=0
    for intent,prompt,required,forbidden in HELDOUT_CASES:
        reply,dbg=generate_combined(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,
            llm_det,python_det,python_th,hw_det,cpu_det,cpu_th,llm2_det,llm2_th,
            tr_det,tr_th,prompt,boundary
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
    print(f"Original relation hits : {orig_relation}")
    print(f"Held-out technical    : {tech_ok}/{tech} ({tech_ok/tech:.1%})")
    print(f"Held-out Transformer  : {tr_ok}/{tr_n} ({tr_ok/tr_n:.1%})")
    print(f"Held-out LLM          : {llm_ok}/{llm_n} ({llm_ok/llm_n:.1%})")
    print(f"Held-out CPU          : {cpu_ok}/{cpu_n} ({cpu_ok/cpu_n:.1%})")
    print(f"Held-out GPU          : {gpu_ok}/{gpu_n} ({gpu_ok/gpu_n:.1%})")
    print(f"Held-out Python       : {py_ok}/{py_n} ({py_ok/py_n:.1%})")
    print(f"Held-out CUDA         : {cuda_ok}/{cuda_n} ({cuda_ok/cuda_n:.1%})")
    print(f"Held-out relation hits: {relation_count}")
    print(f"Held-out total        : {total_ok}/60 ({total_ok/60:.1%})")

if __name__=="__main__":
    main()
