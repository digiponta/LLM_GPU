# evaluate_cpu_gpu_relational_binding_v0111.py
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from cpu_gpu_relational_binding_v0111 import RELATION_LABELS, load_checkpoint
from evaluate_generalization_v07 import CASES, dimension_match
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER="model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL="model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT="model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_REL="model/model-gpu-v0.11.1-cpu-gpu-relational-binding.pt"

DIRECT={5:"CPU",27:"GPU",28:"CPU"}

@torch.no_grad()
def generate(model,tok,intent_head,intent_labels,relation_head,binding,prompt_text,max_new_tokens=96):
    prompt=f"人: {prompt_text}\nAI: "
    generated=tok.encode(prompt,add_bos=True)
    response=[]
    device=next(model.parameters()).device
    debug=None

    for step in range(max_new_tokens):
        context=generated[-model.context_length:]
        x=torch.tensor([context],dtype=torch.long,device=device)
        hidden=model.forward_hidden(x)
        prompt_hidden=hidden[:,-1,:]
        logits=model.lm_head(prompt_hidden)[0].clone()

        if step==0:
            intent_prob=torch.sigmoid(intent_head(prompt_hidden))
            relation_prob=torch.sigmoid(relation_head(prompt_hidden))
            label_to_i={x:i for i,x in enumerate(intent_labels)}
            cpu=float(intent_prob[0,label_to_i["tech_cpu"]].item())
            gpu=float(intent_prob[0,label_to_i["tech_gpu"]].item())
            top_entity="CPU" if cpu>gpu else "GPU"
            relation_max=float(relation_prob.max().item())

            # Apply only for CPU/GPU semantic questions with one clearly stronger side.
            active=max(cpu,gpu)>=0.50 and abs(cpu-gpu)>=0.10
            if active:
                logits=logits+binding(intent_prob,relation_prob)[0]

            rvals,ridx=torch.topk(relation_prob[0],len(RELATION_LABELS))
            ranked_rel=[(RELATION_LABELS[int(i)],float(v)) for v,i in zip(rvals.tolist(),ridx.tolist())]
            debug=(cpu,gpu,top_entity,relation_max,active,ranked_rel)

        if response:
            for tid in set(response):
                if logits[tid]>=0: logits[tid]/=1.05
                else: logits[tid]*=1.05

        next_id=int(torch.argmax(logits).item())
        if next_id==tok.eos_id: break
        generated.append(next_id); response.append(next_id)
        dec=tok.decode(response,skip_special_tokens=True)
        if "\n" in dec: break

    reply=tok.decode(response,skip_special_tokens=True).split("\n",1)[0].strip()
    return reply,debug

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--intent-head",default=DEFAULT_INTENT)
    p.add_argument("--relational-binding",default=DEFAULT_REL)
    args=p.parse_args()

    for f in (args.tokenizer,args.model,args.intent_head,args.relational_binding):
        if not Path(f).exists(): raise FileNotFoundError(f)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,base_ck=LanguageModel.load_checkpoint(args.model,device=device); model.eval()
    intent_head,intent_ck,intent_labels=load_intent_head(args.intent_head,model,device)
    relation_head,binding,rel_ck=load_checkpoint(args.relational_binding,intent_labels,device)

    print("====================================================")
    print(" CPU-GPU Relational Binding v0.11.1 Evaluation")
    print("====================================================")
    print("Device:",device)
    print("Base loss:",base_ck.get("loss"))
    print("Relational binding loss:",rel_ck.get("loss"))

    sem=strict=flu=0; direct=0
    for idx,c in enumerate(CASES,1):
        reply,debug=generate(model,tok,intent_head,intent_labels,relation_head,binding,str(c["prompt"]))
        dims=dimension_match(reply,str(c["prompt"]),str(c["intent"]),c["required_all"],c["forbidden"])
        sem+=int(dims["semantic_ok"]); strict+=int(dims["strict_ok"]); flu+=int(dims["fluent_ok"])

        print(f"[G{idx:02d}] {str(c['intent']):14s} 人: {c['prompt']}")
        print("      AI:",reply)
        print("      semantic-content="+("PASS" if dims["semantic_ok"] else "MISS")+" | strict="+("PASS" if dims["strict_ok"] else "MISS"))

        if debug:
            cpu,gpu,top_entity,rmax,active,ranked_rel=debug
            print(f"      CPU={cpu:.3f} GPU={gpu:.3f} top={top_entity} relational-binding={'ON' if active else 'OFF'}")
            print("      relations="+", ".join(f"{n}={v:.3f}" for n,v in ranked_rel))

        if idx in DIRECT:
            ok=reply.startswith(DIRECT[idx])
            direct+=int(ok)
            print("      direct-entity="+("PASS" if ok else "MISS")+" | expected="+DIRECT[idx])

    print()
    print("Summary")
    print("-------")
    print(f"Semantic-content rate : {sem}/30 ({sem/30:.1%})")
    print(f"Fluency rate          : {flu}/30 ({flu/30:.1%})")
    print(f"Strict composite rate : {strict}/30 ({strict/30:.1%})")
    print(f"CPU/GPU direct rate   : {direct}/3 ({direct/3:.1%})")

if __name__=="__main__":
    main()
