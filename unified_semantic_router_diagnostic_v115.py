# unified_semantic_router_diagnostic_v115.py
from __future__ import annotations

import argparse
import csv
from collections import defaultdict, Counter
from pathlib import Path

import torch

from fresh_generalization_cases_v114 import CASES
from train_cpu_gpu_semantic_detector_v1010 import HardwareDetector
from evaluate_cpu_gpu_semantic_repair_v1010 import load_hardware_detector
from evaluate_cpu_semantic_repair_v1012 import load_cpu_detector
from evaluate_python_semantic_repair_v1009 import load_binary_detector
from evaluate_llm_semantic_repair_v1006 import load_detector
from evaluate_llm_second_stage_repair_v110 import load_second_stage
from evaluate_transformer_semantic_repair_v111 import load_transformer_detector
from selective_intent_repair_v01110 import TECH_LABELS, extract_technical_logits, load_checkpoint as load_repair
from evaluate_transformer_continuation_category_repair_v01117 import (
    DEFAULT_TOKENIZER, DEFAULT_MODEL, DEFAULT_INTENT, DEFAULT_REPAIR,
)
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

TECH = ["gpu","cpu","llm","transformer","cuda","python"]


@torch.no_grad()
def hidden_for_prompt(model,tok,prompt):
    device=next(model.parameters()).device
    ids=tok.encode(f"人: {prompt}\nAI: ",add_bos=True)
    x=torch.tensor([ids[-model.context_length:]],dtype=torch.long,device=device)
    return model.forward_hidden(x)[:,-1,:]


@torch.no_grad()
def sigmoid_score(det,h):
    return float(torch.sigmoid(det(h))[0].item())


@torch.no_grad()
def all_scores(model,tok,intent_head,intent_labels,repair,llm_det,py_det,hw_det,cpu_det,llm2_det,tr_det,prompt):
    h=hidden_for_prompt(model,tok,prompt)

    llm1=sigmoid_score(llm_det,h)
    py=sigmoid_score(py_det,h)
    cpu2=sigmoid_score(cpu_det,h)
    llm2=sigmoid_score(llm2_det,h)
    tr=sigmoid_score(tr_det,h)

    hw_probs=torch.softmax(hw_det(h),dim=-1)[0]
    hw_cpu=float(hw_probs[0].item())
    hw_gpu=float(hw_probs[1].item())
    hw_other=float(hw_probs[2].item())

    intent_logits=intent_head(h)
    raw=extract_technical_logits(intent_logits,intent_labels)
    repaired,scope_logit=repair(raw)
    repair_probs=torch.softmax(repaired,dim=-1)[0]
    scope=float(torch.sigmoid(scope_logit)[0].item())
    rep={TECH_LABELS[i]:float(repair_probs[i].item()) for i in range(len(TECH_LABELS))}
    cuda_rep=rep.get("tech_cuda",0.0)

    # Diagnostic candidate scores only. These heads were trained independently and
    # are NOT calibrated probabilities across classes. The purpose is to expose
    # competition/confusion before training a unified router.
    scores={
        "gpu": hw_gpu,
        "cpu": max(hw_cpu,cpu2),
        "llm": max(llm1,llm2),
        "transformer": tr,
        "cuda": cuda_rep * scope,
        "python": py,
    }
    ranked=sorted(scores.items(),key=lambda kv:kv[1],reverse=True)
    top1,top1_score=ranked[0]
    top2,top2_score=ranked[1]
    return {
        "scores":scores,
        "top1":top1,
        "top1_score":top1_score,
        "top2":top2,
        "top2_score":top2_score,
        "margin":top1_score-top2_score,
        "llm_primary":llm1,
        "llm_second":llm2,
        "python_binary":py,
        "cpu_binary":cpu2,
        "transformer_binary":tr,
        "hardware_cpu":hw_cpu,
        "hardware_gpu":hw_gpu,
        "hardware_other":hw_other,
        "cuda_repair":cuda_rep,
        "repair_scope":scope,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--intent-head",default=DEFAULT_INTENT)
    p.add_argument("--repair",default=DEFAULT_REPAIR)
    p.add_argument("--llm-detector",default="model/model-gpu-v1.0.7-llm-detector-hard-negative.pt")
    p.add_argument("--python-detector",default="model/model-gpu-v1.0.9-python-detector.pt")
    p.add_argument("--hardware-detector",default="model/model-gpu-v1.0.10-cpu-gpu-detector.pt")
    p.add_argument("--cpu-detector",default="model/model-gpu-v1.0.12-cpu-detector.pt")
    p.add_argument("--llm2-detector",default="model/model-gpu-v1.1.0-llm-second-stage.pt")
    p.add_argument("--transformer-detector",default="model/model-gpu-v1.1.1-transformer-detector.pt")
    p.add_argument("--csv",default="results/unified_semantic_router_diagnostic_v115/router_scores.csv")
    p.add_argument("--confusion-csv",default="results/unified_semantic_router_diagnostic_v115/confusion.csv")
    args=p.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device); model.eval()
    intent_head,_,intent_labels=load_intent_head(args.intent_head,model,device)
    repair,_=load_repair(args.repair,intent_labels,device)
    llm_det,_,_=load_detector(args.llm_detector,device)
    py_det,_,_=load_binary_detector(args.python_detector,device)
    hw_det,_,_=load_hardware_detector(args.hardware_detector,device)
    cpu_det,_,_=load_cpu_detector(args.cpu_detector,device)
    llm2_det,_,_=load_second_stage(args.llm2_detector,device)
    tr_det,_,_=load_transformer_detector(args.transformer_detector,device)

    print("="*92)
    print(" LLM_GPU v1.1.5 Unified Semantic Router Diagnostic")
    print("="*92)
    print("Device:",device)
    print("Development set: Fresh Generalization v2 technical subset (60 prompts)")
    print("Important: component heads are independently trained and not cross-calibrated.")
    print()

    rows=[]
    confusion=defaultdict(Counter)
    correct=0
    margins_correct=[]
    margins_wrong=[]
    per_class=defaultdict(lambda:{"n":0,"top1":0,"gold_mean":0.0,"margin":0.0})

    tech_cases=[x for x in CASES if x[0] in TECH]
    for idx,(gold,prompt,required,forbidden) in enumerate(tech_cases,1):
        d=all_scores(model,tok,intent_head,intent_labels,repair,llm_det,py_det,hw_det,cpu_det,llm2_det,tr_det,prompt)
        is_correct=d["top1"]==gold
        correct+=int(is_correct)
        confusion[gold][d["top1"]]+=1
        gold_score=d["scores"][gold]
        s=per_class[gold]
        s["n"]+=1; s["top1"]+=int(is_correct); s["gold_mean"]+=gold_score; s["margin"]+=d["margin"]
        (margins_correct if is_correct else margins_wrong).append(d["margin"])

        row={
            "id":f"U{idx:03d}","gold":gold,"prompt":prompt,
            "top1":d["top1"],"top1_score":d["top1_score"],
            "top2":d["top2"],"top2_score":d["top2_score"],"margin":d["margin"],
            "gold_score":gold_score,
            **{f"score_{k}":d["scores"][k] for k in TECH},
            "llm_primary":d["llm_primary"],"llm_second":d["llm_second"],
            "python_binary":d["python_binary"],"cpu_binary":d["cpu_binary"],
            "transformer_binary":d["transformer_binary"],
            "hardware_cpu":d["hardware_cpu"],"hardware_gpu":d["hardware_gpu"],"hardware_other":d["hardware_other"],
            "cuda_repair":d["cuda_repair"],"repair_scope":d["repair_scope"],
        }
        rows.append(row)
        print(
            f"U{idx:03d} gold={gold:11s} top1={d['top1']:11s} "
            f"{d['top1_score']:.3f} top2={d['top2']:11s} {d['top2_score']:.3f} "
            f"margin={d['margin']:.3f} {'OK' if is_correct else 'MISS'}"
        )

    print()
    print("Raw-score top-1 diagnostic")
    print("--------------------------")
    print(f"Top-1: {correct}/{len(tech_cases)} ({correct/len(tech_cases):.1%})")
    mc=sum(margins_correct)/len(margins_correct) if margins_correct else 0.0
    mw=sum(margins_wrong)/len(margins_wrong) if margins_wrong else 0.0
    print(f"Mean margin correct: {mc:.4f}")
    print(f"Mean margin wrong  : {mw:.4f}")
    print()

    print("Per-class")
    print("---------")
    for cls in TECH:
        s=per_class[cls]
        print(
            f"{cls:11s} top1={s['top1']}/{s['n']} ({s['top1']/s['n']:.1%}) "
            f"mean_gold_score={s['gold_mean']/s['n']:.3f} mean_margin={s['margin']/s['n']:.3f}"
        )

    print()
    print("Confusion matrix: rows=gold, columns=predicted")
    print("------------------------------------------------")
    print("gold\\pred   "+" ".join(f"{x[:5]:>5s}" for x in TECH))
    for g in TECH:
        print(f"{g[:9]:9s} "+" ".join(f"{confusion[g][p]:5d}" for p in TECH))

    out=Path(args.csv); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    cout=Path(args.confusion_csv); cout.parent.mkdir(parents=True,exist_ok=True)
    with cout.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["gold"]+TECH)
        for g in TECH:
            w.writerow([g]+[confusion[g][p] for p in TECH])

    print()
    print("CSV:",out)
    print("Confusion CSV:",cout)
    print()
    print("Interpretation note:")
    print("These raw top-1 results are diagnostic only. Independent detector scores are not")
    print("probability-calibrated against each other. Use this output to design calibration")
    print("or a learned unified router; do not deploy raw argmax as the final policy.")


if __name__=="__main__":
    main()
