# run_llm_hardest_competitor_boost_sweep_v01113.py
from __future__ import annotations

import csv
from pathlib import Path
import torch

from selective_intent_repair_v01110 import TECH_LABELS, extract_technical_logits, load_checkpoint as load_repair
from multi_concept_safe_binding_v0118 import CONCEPTS, CANONICAL, load_checkpoint as load_binding
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from evaluate_generalization_v07 import CASES, dimension_match
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

TOKENIZER="model/tokenizer-v0.7-bpe.json"
MODEL="model/model-gpu-v0.8-chat-clean.pt"
INTENT="model/model-gpu-v0.8-intent-head-clean.pt"
ROLE="model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt"
BINDING="model/model-gpu-v0.11.8-multi-concept-safe-binding.pt"
REPAIR="model/model-gpu-v0.11.10-selective-intent-repair.pt"
RESULTS=Path("results/llm_hardest_competitor_boost_sweep_v01113")
SUMMARY=RESULTS/"boost_sweep.csv"

MULTIPLIERS=(1.0,1.25,1.5,2.0,2.5,3.0)
REPAIR_MARGIN=0.20
SCOPE_THRESHOLD=0.50
BINDING_CONFIDENCE=0.45
REPAIR_CONFIDENCE=0.55
LLM_RESCUE_MARGIN=0.20
LLM_RESCUE_CONFIDENCE=0.70

@torch.no_grad()
def generate(model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,prompt_text,multiplier):
    prompt=f"人: {prompt_text}\nAI: "
    generated=tok.encode(prompt,add_bos=True)
    response=[]
    device=next(model.parameters()).device
    cpu_idx=intent_labels.index("tech_cpu"); gpu_idx=intent_labels.index("tech_gpu")
    pair_intent_margin=float(binding_ck.get("cpu_gpu_intent_margin",0.10))
    pair_role_margin=float(binding_ck.get("cpu_gpu_role_margin",0.05))
    debug=None

    for step in range(96):
        x=torch.tensor([generated[-model.context_length:]],dtype=torch.long,device=device)
        h=model.forward_hidden(x)[:,-1,:]
        logits=model.lm_head(h)[0].clone()

        if step==0:
            intent_logits=intent_head(h)
            ip=torch.sigmoid(intent_logits)
            rp=torch.sigmoid(role_head(h))
            raw=extract_technical_logits(intent_logits,intent_labels)
            repaired,scope_logit=repair(raw)
            raw_scores=torch.sigmoid(raw)[0]
            repaired_probs=torch.softmax(repaired,dim=-1)[0]
            scope=float(torch.sigmoid(scope_logit)[0].item())

            order=torch.argsort(raw[0],descending=True)
            top=int(order[0].item()); second=int(order[1].item())
            raw_top=TECH_LABELS[top]
            raw_margin=float((raw[0,top]-raw[0,second]).item())
            rep_top=int(torch.argmax(repaired_probs).item())
            rep_label=TECH_LABELS[rep_top]
            rep_conf=float(repaired_probs[rep_top].item())

            chosen=top
            if raw_margin<=REPAIR_MARGIN and rep_top!=top and rep_conf>=REPAIR_CONFIDENCE:
                chosen=rep_top

            chosen_label=TECH_LABELS[chosen]
            canonical=CANONICAL[chosen_label]
            cidx=CONCEPTS.index(chosen_label)
            target_id=int(binding_ck["token_ids"][chosen_label])
            chosen_conf=max(float(raw_scores[chosen].item()),float(repaired_probs[chosen].item()))

            cpu=float(ip[0,cpu_idx].item()); gpu=float(ip[0,gpu_idx].item())
            controller=float(rp[0,0].item()); executor=float(rp[0,1].item())

            pair_gate=False; concept_gate=False; rescue=False
            if chosen_label=="tech_cpu":
                pair_gate=cpu>gpu and (cpu-gpu)>=pair_intent_margin and controller>executor and (controller-executor)>=pair_role_margin
            elif chosen_label=="tech_gpu":
                pair_gate=gpu>cpu and (gpu-cpu)>=pair_intent_margin and executor>controller and (executor-controller)>=pair_role_margin
            else:
                normal_scope=scope>=SCOPE_THRESHOLD
                if chosen_label=="tech_llm":
                    rescue=(raw_top=="tech_llm" and rep_label=="tech_llm" and raw_margin<=LLM_RESCUE_MARGIN and chosen_conf>=LLM_RESCUE_CONFIDENCE)
                concept_gate=(normal_scope or rescue) and chosen_conf>=BINDING_CONFIDENCE and canonical.lower() not in prompt_text.lower()

            active=pair_gate or concept_gate
            base_top=int(torch.argmax(logits).item())
            base_target=float(logits[target_id].item())
            boost=0.0
            raw_boost=0.0
            if active:
                feats=torch.cat([ip[0],rp[0]],dim=-1).unsqueeze(0)
                ci=torch.tensor([cidx],dtype=torch.long,device=device)
                raw_boost=float(binding(feats,ci)[0].item())
                boost=raw_boost*(multiplier if rescue and chosen_label=="tech_llm" else 1.0)
                logits[target_id]+=boost
            after_top=int(torch.argmax(logits).item())
            debug={
                "chosen":chosen_label,"scope":scope,"rescue":rescue,"active":active,
                "raw_boost":raw_boost,"boost":boost,"base_top":base_top,"after_top":after_top,
                "target_id":target_id,"base_target":base_target,
                "after_target":float(logits[target_id].item()),
                "base_winner_logit":float(model.lm_head(h)[0,base_top].item()),
                "after_winner_logit":float(logits[after_top].item()),
            }

        if response:
            for tid in set(response):
                if logits[tid]>=0: logits[tid]/=1.05
                else: logits[tid]*=1.05
        nid=int(torch.argmax(logits).item())
        if nid==tok.eos_id: break
        generated.append(nid); response.append(nid)
        if "\n" in tok.decode(response,skip_special_tokens=True): break

    return tok.decode(response,skip_special_tokens=True).split("\n",1)[0].strip(),debug

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(TOKENIZER)
    model,_=LanguageModel.load_checkpoint(MODEL,device=device); model.eval()
    intent_head,_,intent_labels=load_intent_head(INTENT,model,device)
    role_head,_,_=load_role_checkpoint(ROLE,intent_labels,device)
    binding,binding_ck=load_binding(BINDING,intent_labels,device)
    repair,_=load_repair(REPAIR,intent_labels,device)

    print("="*52)
    print(" LLM Hardest-Competitor Boost Sweep v0.11.13")
    print("="*52)
    print("Device:",device)
    print("Multipliers:",MULTIPLIERS)
    print("v0.11.11 thresholds fixed: margin=0.20 scope=0.50 confidence=0.45")
    print()

    rows=[]
    best=None
    for mult in MULTIPLIERS:
        semantic=strict=fluency=0
        replies={}; dbgs={}
        for idx,case in enumerate(CASES,start=1):
            reply,dbg=generate(model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,str(case["prompt"]),mult)
            dims=dimension_match(reply,str(case["prompt"]),str(case["intent"]),case["required_all"],case["forbidden"])
            semantic+=int(dims["semantic_ok"]); strict+=int(dims["strict_ok"]); fluency+=int(dims["fluent_ok"])
            replies[idx]=reply; dbgs[idx]=dbg

        guards=(
            replies[5].startswith("CPU") and replies[9].startswith("CUDA")
            and replies[10].startswith("Python") and replies[27].startswith("GPU")
            and replies[28].startswith("CPU") and not dbgs[21]["active"]
            and not dbgs[30]["active"]
        )
        g07=replies[7].startswith("LLM")
        row={
            "multiplier":mult,"guards_ok":int(guards),"g07_ok":int(g07),
            "semantic":semantic,"strict":strict,"fluency":fluency,
            "g07_reply":replies[7],"g07_raw_boost":dbgs[7]["raw_boost"],
            "g07_boost":dbgs[7]["boost"],"g07_base_target":dbgs[7]["base_target"],
            "g07_after_target":dbgs[7]["after_target"],
            "g07_base_winner_logit":dbgs[7]["base_winner_logit"],
            "g07_after_winner_logit":dbgs[7]["after_winner_logit"],
        }
        rows.append(row)
        print(
            f"x{mult:<4.2f} | guards={'PASS' if guards else 'FAIL'} "
            f"| G07={'PASS' if g07 else 'MISS'} "
            f"| boost={row['g07_boost']:+.3f} "
            f"| sem={semantic}/30 strict={strict}/30"
        )
        if guards and g07 and best is None:
            best=row

    with SUMMARY.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print()
    print("Best configuration")
    print("------------------")
    if best is None:
        print("No multiplier passed G07 plus all safety guards.")
    else:
        print(f"multiplier            : {best['multiplier']:.2f}")
        print("guards                : PASS")
        print("G07 LLM               : PASS")
        print(f"G07 raw boost         : {best['g07_raw_boost']:+.3f}")
        print(f"G07 applied boost     : {best['g07_boost']:+.3f}")
        print(f"G07 base target logit : {best['g07_base_target']:+.3f}")
        print(f"G07 after target      : {best['g07_after_target']:+.3f}")
        print(f"Semantic-content      : {best['semantic']}/30 ({best['semantic']/30:.1%})")
        print(f"Strict                : {best['strict']}/30 ({best['strict']/30:.1%})")
        print("G07 reply             :",best["g07_reply"])
    print("CSV:",SUMMARY)

if __name__=="__main__":
    main()
