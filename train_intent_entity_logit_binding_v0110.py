# train_intent_entity_logit_binding_v0110.py
from __future__ import annotations
import argparse, random
from pathlib import Path
import torch
import torch.nn.functional as F
from augment_sft_v07 import REVERSE_DEFINITION_ROWS, PAIRWISE_HARD_NEGATIVE_ROWS, TARGETED_BOUNDARY_ROWS, TARGETED_BOUNDARY_V2_ROWS, PROTECTED_BOUNDARY_REPLAY_ROWS
from evaluate_generalization_v07 import CASES
from intent_conditioning_v09 import load_intent_head
from intent_entity_logit_binding_v0110 import IntentEntityLogitBinding, ENTITY_TARGETS, save_binding_checkpoint
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER="model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL="model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT="model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_OUTPUT="model/model-gpu-v0.11.0-intent-entity-binding.pt"
SEED=42

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--intent-head",default=DEFAULT_INTENT)
    p.add_argument("--output",default=DEFAULT_OUTPUT)
    p.add_argument("--epochs",type=int,default=200)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--rank",type=int,default=32)
    p.add_argument("--beta",type=float,default=1.0)
    p.add_argument("--gate-threshold",type=float,default=0.50)
    p.add_argument("--l2-weight",type=float,default=1e-5)
    p.add_argument("--patience",type=int,default=25)
    return p.parse_args()

def rows():
    out=[]
    for group in (REVERSE_DEFINITION_ROWS,PAIRWISE_HARD_NEGATIVE_ROWS,TARGETED_BOUNDARY_ROWS,TARGETED_BOUNDARY_V2_ROWS,PROTECTED_BOUNDARY_REPLAY_ROWS):
        for prompt,answer,label in group:
            if label in ENTITY_TARGETS and answer.startswith(ENTITY_TARGETS[label]):
                out.append((prompt,label))
    seen=set(); ded=[]
    for x in out:
        if x not in seen: seen.add(x); ded.append(x)
    return ded

def main():
    args=parse_args(); torch.manual_seed(SEED); random.seed(SEED)
    for f in (args.tokenizer,args.model,args.intent_head):
        if not Path(f).exists(): raise FileNotFoundError(f)
    dev={str(c["prompt"]) for c in CASES}
    data=rows()
    overlap=[p for p,_ in data if p in dev]
    if overlap: raise RuntimeError("Exact DEV overlap: "+repr(overlap))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,ck=LanguageModel.load_checkpoint(args.model,device=device); model.eval()
    head,hck,labels=load_intent_head(args.intent_head,model,device)
    for m in (model,head):
        for p in m.parameters(): p.requires_grad_(False)
    target_ids={}
    for label,entity in ENTITY_TARGETS.items():
        ids=tok.encode(entity)
        if not ids: raise RuntimeError(entity)
        target_ids[label]=int(ids[0])
    probs=[]; logits=[]; targets=[]; labs=[]
    with torch.no_grad():
        for prompt,label in data:
            ids=tok.encode(f"人: {prompt}\nAI: ",add_bos=True)[-model.context_length:]
            x=torch.tensor([ids],dtype=torch.long,device=device)
            hidden=model.forward_hidden(x)
            pr=hidden[:,-1,:]
            ip=torch.sigmoid(head(pr))
            lg=model.lm_head(hidden[:,-1,:])
            probs.append(ip[0]); logits.append(lg[0]); targets.append(target_ids[label]); labs.append(label)
    probs=torch.stack(probs); logits=torch.stack(logits); targets=torch.tensor(targets,device=device)
    by={}
    for i,l in enumerate(labs): by.setdefault(l,[]).append(i)
    tr=[]; va=[]
    for l,idx in by.items(): va.append(idx[-1]); tr.extend(idx[:-1])
    tr=torch.tensor(tr,device=device); va=torch.tensor(va,device=device)
    adapter=IntentEntityLogitBinding(len(labels),model.vocab_size,args.rank,args.beta).to(device)
    opt=torch.optim.AdamW(adapter.parameters(),lr=args.lr,weight_decay=0.01)
    def loss(ix):
        b=adapter(probs[ix]); combined=logits[ix]+b
        ce=F.cross_entropy(combined,targets[ix]); reg=b.pow(2).mean()
        return ce+args.l2_weight*reg,ce,reg
    print("===================================================="); print(" Intent -> Entity Logit Binding v0.11.0"); print("====================================================")
    print("Device:",device); print("Base model: frozen"); print("Intent head: frozen"); print("Rows:",len(data),"Train:",len(tr),"Val:",len(va)); print("Exact DEV overlap:",len(overlap))
    print("Gate threshold:",args.gate_threshold); print("First-token only: yes")
    for l,e in ENTITY_TARGETS.items(): print(l,"->",e,"token",target_ids[l])
    best=1e9; state=None; be=0; bad=0
    for ep in range(1,args.epochs+1):
        adapter.train(); opt.zero_grad(); lo,ce,reg=loss(tr); lo.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0); opt.step()
        adapter.eval()
        with torch.no_grad(): vl,vce,vreg=loss(va)
        if ep==1 or ep%10==0: print(f"Epoch {ep:03d}/{args.epochs} | train={lo.item():.4f} ce={ce.item():.4f} | val={vl.item():.4f} ce={vce.item():.4f}")
        v=float(vl.item())
        if v<best-1e-6:
            best=v; be=ep; state={k:x.detach().cpu().clone() for k,x in adapter.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=args.patience: print("Early stopping."); break
    adapter.load_state_dict(state)
    save_binding_checkpoint(args.output,adapter,labels,target_ids,epoch=be,loss=best,base_model=args.model,intent_head=args.intent_head,learning_rate=args.lr,gate_threshold=args.gate_threshold)
    print("Best epoch:",be); print("Best val loss:",f"{best:.6f}"); print("Checkpoint:",args.output)
if __name__=="__main__": main()
