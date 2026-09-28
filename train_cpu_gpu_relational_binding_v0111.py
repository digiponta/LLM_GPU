# train_cpu_gpu_relational_binding_v0111.py
from __future__ import annotations
import argparse, random
from pathlib import Path
import torch
import torch.nn.functional as F

from augment_sft_v07 import (
    CPU_GPU_RELATIONAL_ROWS,
    REVERSE_DEFINITION_ROWS,
    TARGETED_BOUNDARY_ROWS,
    TARGETED_BOUNDARY_V2_ROWS,
)
from cpu_gpu_relational_binding_v0111 import (
    RELATION_LABELS,
    RelationHead,
    RelationalEntityBinding,
    save_checkpoint,
)
from evaluate_generalization_v07 import CASES
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER="model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL="model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT="model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_OUTPUT="model/model-gpu-v0.11.1-cpu-gpu-relational-binding.pt"
SEED=42

CPU_RELATIONS={"cpu_controls_gpu","cpu_assigns_work_gpu"}
GPU_RELATIONS={"gpu_controlled_by_cpu","gpu_executes_for_cpu"}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--intent-head",default=DEFAULT_INTENT)
    p.add_argument("--output",default=DEFAULT_OUTPUT)
    p.add_argument("--epochs",type=int,default=240)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--rank",type=int,default=32)
    p.add_argument("--beta",type=float,default=1.0)
    p.add_argument("--relation-weight",type=float,default=0.50)
    p.add_argument("--entity-weight",type=float,default=1.00)
    p.add_argument("--l2-weight",type=float,default=1e-5)
    p.add_argument("--patience",type=int,default=30)
    return p.parse_args()

def build_rows():
    rows=[]
    # Explicit directional relation supervision.
    for prompt,answer,rel in CPU_GPU_RELATIONAL_ROWS:
        entity="CPU" if rel in CPU_RELATIONS else "GPU"
        rows.append((prompt,rel,entity))

    # Add paraphrase support from existing CPU/GPU reverse-definition rows.
    for group in (REVERSE_DEFINITION_ROWS,TARGETED_BOUNDARY_ROWS,TARGETED_BOUNDARY_V2_ROWS):
        for prompt,answer,label in group:
            if label=="tech_cpu" and answer.startswith("CPU"):
                rows.append((prompt,"cpu_controls_gpu","CPU"))
            elif label=="tech_gpu" and answer.startswith("GPU"):
                rows.append((prompt,"gpu_executes_for_cpu","GPU"))

    seen=set(); out=[]
    for row in rows:
        if row not in seen:
            seen.add(row); out.append(row)
    return out

def main():
    args=parse_args()
    torch.manual_seed(SEED); random.seed(SEED)

    for f in (args.tokenizer,args.model,args.intent_head):
        if not Path(f).exists(): raise FileNotFoundError(f)

    rows=build_rows()
    dev={str(c["prompt"]) for c in CASES}
    overlap=[p for p,_,_ in rows if p in dev]
    if overlap: raise RuntimeError("Exact DEV overlap: "+repr(overlap))

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,base_ck=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    intent_head,intent_ck,intent_labels=load_intent_head(args.intent_head,model,device)
    for module in (model,intent_head):
        for p in module.parameters(): p.requires_grad_(False)

    cpu_id=int(tok.encode("CPU")[0])
    gpu_id=int(tok.encode("GPU")[0])
    target_token_ids={"CPU":cpu_id,"GPU":gpu_id}

    hidden_rows=[]; intent_rows=[]; base_logits=[]; relation_targets=[]; entity_targets=[]; relation_names=[]
    rel_to_id={r:i for i,r in enumerate(RELATION_LABELS)}

    with torch.no_grad():
        for prompt,rel,entity in rows:
            ids=tok.encode(f"人: {prompt}\nAI: ",add_bos=True)[-model.context_length:]
            x=torch.tensor([ids],dtype=torch.long,device=device)
            hidden=model.forward_hidden(x)
            prompt_hidden=hidden[:,-1,:]
            intent_prob=torch.sigmoid(intent_head(prompt_hidden))
            logits=model.lm_head(prompt_hidden)
            hidden_rows.append(prompt_hidden[0])
            intent_rows.append(intent_prob[0])
            base_logits.append(logits[0])
            t=torch.zeros(len(RELATION_LABELS),device=device)
            t[rel_to_id[rel]]=1.0
            relation_targets.append(t)
            entity_targets.append(cpu_id if entity=="CPU" else gpu_id)
            relation_names.append(rel)

    hidden_rows=torch.stack(hidden_rows)
    intent_rows=torch.stack(intent_rows)
    base_logits=torch.stack(base_logits)
    relation_targets=torch.stack(relation_targets)
    entity_targets=torch.tensor(entity_targets,dtype=torch.long,device=device)

    # Deterministic split: last row of each relation label for validation.
    by={}
    for i,rel in enumerate(relation_names): by.setdefault(rel,[]).append(i)
    train_idx=[]; val_idx=[]
    for rel,idxs in by.items():
        val_idx.append(idxs[-1]); train_idx.extend(idxs[:-1])
    train_idx=torch.tensor(train_idx,dtype=torch.long,device=device)
    val_idx=torch.tensor(val_idx,dtype=torch.long,device=device)

    relation_head=RelationHead(model.d_model).to(device)
    binding=RelationalEntityBinding(len(intent_labels),len(RELATION_LABELS),model.vocab_size,args.rank,args.beta).to(device)
    params=list(relation_head.parameters())+list(binding.parameters())
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)

    pos_weight=torch.full((len(RELATION_LABELS),),3.0,device=device)

    def compute(ix):
        rlogits=relation_head(hidden_rows[ix])
        rprob=torch.sigmoid(rlogits)
        rel_loss=F.binary_cross_entropy_with_logits(rlogits,relation_targets[ix],pos_weight=pos_weight)
        bias=binding(intent_rows[ix],rprob)
        combined=base_logits[ix]+bias
        ent_loss=F.cross_entropy(combined,entity_targets[ix])
        reg=bias.pow(2).mean()
        total=args.relation_weight*rel_loss+args.entity_weight*ent_loss+args.l2_weight*reg
        return total,rel_loss,ent_loss,reg

    print("====================================================")
    print(" CPU-GPU Relational Binding v0.11.1")
    print("====================================================")
    print("Device:",device)
    print("Base model/head: frozen")
    print("Rows:",len(rows),"Train:",len(train_idx),"Val:",len(val_idx))
    print("Exact DEV overlap:",len(overlap))
    print("Relation labels:",", ".join(RELATION_LABELS))
    print("CPU token:",cpu_id,"GPU token:",gpu_id)

    best=1e9; best_state=None; best_epoch=0; bad=0
    for ep in range(1,args.epochs+1):
        relation_head.train(); binding.train(); opt.zero_grad(set_to_none=True)
        loss,rl,el,reg=compute(train_idx)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()

        relation_head.eval(); binding.eval()
        with torch.no_grad():
            vl,vrl,vel,vreg=compute(val_idx)

        if ep==1 or ep%10==0:
            print(f"Epoch {ep:03d}/{args.epochs} | train={loss.item():.4f} rel={rl.item():.4f} entity={el.item():.4f} | val={vl.item():.4f} rel={vrl.item():.4f} entity={vel.item():.4f}")

        value=float(vl.item())
        if value<best-1e-6:
            best=value; best_epoch=ep; bad=0
            best_state={
                "relation":{k:v.detach().cpu().clone() for k,v in relation_head.state_dict().items()},
                "binding":{k:v.detach().cpu().clone() for k,v in binding.state_dict().items()},
            }
        else:
            bad+=1
            if bad>=args.patience:
                print("Early stopping."); break

    relation_head.load_state_dict(best_state["relation"])
    binding.load_state_dict(best_state["binding"])
    save_checkpoint(
        args.output,relation_head,binding,intent_labels,target_token_ids,
        epoch=best_epoch,loss=best,base_model=args.model,intent_head=args.intent_head,
        learning_rate=args.lr,relation_weight=args.relation_weight,entity_weight=args.entity_weight,
    )
    print("Best epoch:",best_epoch)
    print("Best val loss:",f"{best:.6f}")
    print("Checkpoint:",args.output)

if __name__=="__main__":
    main()
