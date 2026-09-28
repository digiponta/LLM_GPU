# evaluate_intent_entity_logit_binding_v0110.py
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from evaluate_generalization_v07 import CASES, dimension_match
from intent_conditioning_v09 import load_intent_head
from intent_entity_logit_binding_v0110 import load_binding_checkpoint, technical_gate
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER="model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL="model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT="model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_BINDING="model/model-gpu-v0.11.0-intent-entity-binding.pt"

DIRECT={5:"CPU",8:"Transformer",9:"CUDA",10:"Python",27:"GPU",28:"CPU"}

def generate(model,tok,head,labels,adapter,threshold,prompt_text,max_new_tokens=96):
    prompt=f"人: {prompt_text}\nAI: "
    generated=tok.encode(prompt,add_bos=True)
    response=[]
    device=next(model.parameters()).device
    first_info=None
    for step in range(max_new_tokens):
        context=generated[-model.context_length:]
        x=torch.tensor([context],dtype=torch.long,device=device)
        hidden=model.forward_hidden(x)
        logits=model.lm_head(hidden[:,-1,:])[0].clone()
        if step==0:
            pr=hidden[:,-1,:]
            ip=torch.sigmoid(head(pr))
            conf_t, active_t, count_t = technical_gate(ip, labels, threshold)
            conf=float(conf_t[0].item())
            active=bool(active_t[0].item())
            active_count=int(count_t[0].item())
            if active: logits=logits+adapter(ip)[0]
            topi=torch.topk(ip[0],min(5,len(labels)))
            ranked=[(labels[int(i)],float(v)) for v,i in zip(topi.values.tolist(),topi.indices.tolist())]
            first_info=(conf,active,active_count,ranked)
        if response:
            for tid in set(response):
                if logits[tid]>=0: logits[tid]/=1.05
                else: logits[tid]*=1.05
        nid=int(torch.argmax(logits).item())
        if nid==tok.eos_id: break
        generated.append(nid); response.append(nid)
        dec=tok.decode(response,skip_special_tokens=True)
        if "\n" in dec: break
    reply=tok.decode(response,skip_special_tokens=True).split("\n",1)[0].strip()
    return reply,first_info

def main():
    p=argparse.ArgumentParser(); p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER); p.add_argument("--model",default=DEFAULT_MODEL); p.add_argument("--intent-head",default=DEFAULT_INTENT); p.add_argument("--binding",default=DEFAULT_BINDING)
    args=p.parse_args()
    for f in (args.tokenizer,args.model,args.intent_head,args.binding):
        if not Path(f).exists(): raise FileNotFoundError(f)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer); model,ck=LanguageModel.load_checkpoint(args.model,device=device); model.eval()
    head,hck,labels=load_intent_head(args.intent_head,model,device)
    adapter,bck=load_binding_checkpoint(args.binding,labels,device)
    threshold=float(bck.get("gate_threshold",0.5))
    print("===================================================="); print(" Intent -> Entity Logit Binding v0.11.0 Evaluation"); print("====================================================")
    print("Device:",device); print("Base loss:",ck.get("loss")); print("Binding loss:",bck.get("loss")); print("Gate threshold:",threshold)
    sem=strict=flu=0; direct=0
    for idx,c in enumerate(CASES,1):
        reply,info=generate(model,tok,head,labels,adapter,threshold,str(c["prompt"]))
        dims=dimension_match(reply,str(c["prompt"]),str(c["intent"]),c["required_all"],c["forbidden"])
        sem+=int(dims["semantic_ok"]); strict+=int(dims["strict_ok"]); flu+=int(dims["fluent_ok"])
        print(f"[G{idx:02d}] {str(c['intent']):14s} 人: {c['prompt']}"); print("      AI:",reply)
        print("      semantic-content="+("PASS" if dims["semantic_ok"] else "MISS")+" | entity="+("N/A" if dims["entity_ok"] is None else ("PASS" if dims["entity_ok"] else "MISS"))+" | fluency="+("PASS" if dims["fluent_ok"] else "MISS")+" | strict="+("PASS" if dims["strict_ok"] else "MISS"))
        if info:
            conf,active,active_count,ranked=info
            print(f"      binding-gate={'ON' if active else 'OFF'} conf={conf:.3f} active-tech={active_count} top="+", ".join(f"{n}={v:.3f}" for n,v in ranked))
        if idx in DIRECT:
            ok=reply.startswith(DIRECT[idx]); direct+=int(ok); print("      direct-entity="+("PASS" if ok else "MISS")+" | expected="+DIRECT[idx])
    print(); print("Summary"); print("-------"); print(f"Semantic-content rate : {sem}/30 ({sem/30:.1%})"); print(f"Fluency rate          : {flu}/30 ({flu/30:.1%})"); print(f"Strict composite rate : {strict}/30 ({strict/30:.1%})"); print(f"Direct-entity rate    : {direct}/6 ({direct/6:.1%})")
if __name__=="__main__": main()
