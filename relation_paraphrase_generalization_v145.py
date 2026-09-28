# relation_paraphrase_generalization_v145.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from semantic_explicit_role_span_v143 import DATASET as CONCEPT_DATASET, CLASSES
from semantic_relation_paraphrase_v145 import RELATION_SAMPLES, FAMILIES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
STAGES = ["block1","block2","block3","block4","block5","block6","final_norm"]

class LinearHead(nn.Module):
    def __init__(self,d_in,n_out):
        super().__init__()
        self.fc=nn.Linear(d_in,n_out)
    def forward(self,x): return self.fc(x)

@torch.no_grad()
def encode_text_stages(model,tok,text):
    device=next(model.parameters()).device
    ids=tok.encode(f"人: {text}\nAI: ",add_bos=True)
    ids=ids[-model.context_length:]
    tid=torch.tensor([ids],dtype=torch.long,device=device)
    x=model.embedding(tid)
    if model.position_embedding is not None:
        pos=torch.arange(tid.shape[1],device=device)
        x=x+model.position_embedding(pos).unsqueeze(0)
    out={}
    for i,block in enumerate(model.blocks,start=1):
        x=block(x)
        out[f"block{i}"]=x[0].mean(0).detach().cpu()
    x=model.final_norm(x)
    out["final_norm"]=x[0].mean(0).detach().cpu()
    return out

def standardize(Xtr,Xte):
    mean=Xtr.mean(0,keepdim=True)
    std=Xtr.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std,(Xte-mean)/std

def train_head(X,y,n_out,args,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    m=LinearHead(X.shape[1],n_out).to(X.device)
    opt=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    loss_fn=nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        m.train(); opt.zero_grad()
        loss=loss_fn(m(X),y); loss.backward(); opt.step()
    m.eval(); return m

def predict_split(X,y,n_out,train_idx,test_idx,args,seeds,offset):
    Xtr,Xte=standardize(X[train_idx],X[test_idx])
    probs=[]
    for seed in seeds:
        m=train_head(Xtr,y[train_idx],n_out,args,seed+offset)
        with torch.no_grad():
            probs.append(torch.softmax(m(Xte),dim=-1).cpu())
    return torch.stack(probs).mean(0).argmax(-1)

def concept_cv_features(model,tok,stage,args,seeds):
    # Reuse v1.4.4-style span concept decoding on all 120 concept samples.
    cache={}
    def enc(text):
        if text not in cache:
            cache[text]=encode_text_stages(model,tok,text)
        return cache[text][stage]

    X1=torch.stack([enc(s["first_desc"]) for s in CONCEPT_DATASET]).to(next(model.parameters()).device)
    X2=torch.stack([enc(s["second_desc"]) for s in CONCEPT_DATASET]).to(next(model.parameters()).device)
    y1=torch.tensor([CLASS_TO_ID[s["first_class"]] for s in CONCEPT_DATASET],dtype=torch.long,device=X1.device)
    y2=torch.tensor([CLASS_TO_ID[s["second_class"]] for s in CONCEPT_DATASET],dtype=torch.long,device=X1.device)

    folds=[[] for _ in range(5)]
    for c in CLASSES:
        for order in (0,1):
            idx=[i for i,s in enumerate(CONCEPT_DATASET) if s["target"]==c and s["surface_order"]==order]
            for k in range(5): folds[k].extend(idx[2*k:2*k+2])

    p1=torch.zeros(len(y1),dtype=torch.long)
    p2=torch.zeros(len(y2),dtype=torch.long)
    for k,test in enumerate(folds):
        ts=set(test); train=[i for i in range(len(y1)) if i not in ts]
        p1[test]=predict_split(X1,y1,6,train,test,args,seeds,10000+1000*k)
        p2[test]=predict_split(X2,y2,6,train,test,args,seeds,20000+1000*k)
    return p1,p2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=250)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--seeds",default="41,42,43")
    ap.add_argument("--output-dir",default="results/relation_paraphrase_generalization_v145")
    args=ap.parse_args()

    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print("="*116)
    print(" LLM_GPU v1.4.5 Relation Phrase Paraphrase Generalization")
    print("="*116)
    print("Device         :",device)
    print("Base           :",args.model)
    print("Relation data  : 20 phrases = 5 families x 2 labels x 2 paraphrases")
    print("Families       :",", ".join(FAMILIES))
    print("Evaluation     : leave-one-relation-family-out")
    print("Labels         : FIRST target / SECOND target both present in every train/test split")
    print("Composition    : Concept1 + Concept2 + predicted TargetPosition -> deterministic Target")
    print("Stages         :",", ".join(STAGES))
    print("Base model     : completely frozen")
    print()

    relation_cache={s["text"]:encode_text_stages(model,tok,s["text"]) for s in RELATION_SAMPLES}
    yrel=torch.tensor([s["target_position"] for s in RELATION_SAMPLES],dtype=torch.long,device=device)

    rows=[]

    for stage in STAGES:
        XR=torch.stack([relation_cache[s["text"]][stage] for s in RELATION_SAMPLES]).to(device)

        pred_rel=torch.zeros(len(RELATION_SAMPLES),dtype=torch.long)
        family_scores={}
        for fi,fam in enumerate(FAMILIES):
            test=[i for i,s in enumerate(RELATION_SAMPLES) if s["family"]==fam]
            train=[i for i,s in enumerate(RELATION_SAMPLES) if s["family"]!=fam]
            pr=predict_split(XR,yrel,2,train,test,args,seeds,30000+1000*fi)
            pred_rel[test]=pr
            family_scores[fam]=float((pr==yrel[test].cpu()).float().mean())

        rel_acc=float((pred_rel==yrel.cpu()).float().mean())

        # Concept heads from the same stage.
        p1,p2=concept_cv_features(model,tok,stage,args,seeds)
        y1=torch.tensor([CLASS_TO_ID[s["first_class"]] for s in CONCEPT_DATASET],dtype=torch.long)
        y2=torch.tensor([CLASS_TO_ID[s["second_class"]] for s in CONCEPT_DATASET],dtype=torch.long)
        concept_acc=(float((p1==y1).float().mean())+float((p2==y2).float().mean()))/2

        # Apply each held-out relation prediction to concept samples with matching target-position label.
        # This isolates whether an unseen relation wording can drive the deterministic binder.
        target_hits=[]
        for i,s in enumerate(CONCEPT_DATASET):
            desired_pos=0 if s["target"]==s["first_class"] else 1
            candidates=[j for j,r in enumerate(RELATION_SAMPLES) if r["target_position"]==desired_pos]
            # Mean correctness over all relation phrases with that semantic label.
            for j in candidates:
                pos=int(pred_rel[j])
                pred_target=int(p1[i]) if pos==0 else int(p2[i])
                target_hits.append(pred_target==CLASS_TO_ID[s["target"]])
        target_acc=sum(target_hits)/len(target_hits)

        # Oracle relation label gives upper bound from concept decoding.
        oracle_hits=[]
        for i,s in enumerate(CONCEPT_DATASET):
            pos=0 if s["target"]==s["first_class"] else 1
            pred_target=int(p1[i]) if pos==0 else int(p2[i])
            oracle_hits.append(pred_target==CLASS_TO_ID[s["target"]])
        oracle_target=sum(oracle_hits)/len(oracle_hits)

        print()
        print(stage.upper())
        print("-"*84)
        print(f"unseen relation phrase accuracy      : {rel_acc:.1%}")
        for fam in FAMILIES:
            print(f"  {fam:10s} {family_scores[fam]:.1%}")
        print(f"mean concept accuracy                : {concept_acc:.1%}")
        print(f"reconstructed target (unseen phrase): {target_acc:.1%}")
        print(f"oracle relation target upper bound   : {oracle_target:.1%}")

        rows.append([
            stage,rel_acc,concept_acc,target_acc,oracle_target,
            *[family_scores[f] for f in FAMILIES]
        ])

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow([
            "stage","unseen_relation_accuracy","mean_concept_accuracy",
            "reconstructed_target_accuracy","oracle_relation_target_accuracy",
            *[f"family_{f}" for f in FAMILIES]
        ])
        w.writerows(rows)

    best_rel=max(rows,key=lambda r:r[1])
    best_target=max(rows,key=lambda r:r[3])

    print()
    print("Best unseen relation generalization")
    print("-----------------------------------")
    print(f"{best_rel[0]} : {best_rel[1]:.1%}")

    print()
    print("Best reconstructed Target")
    print("-------------------------")
    print(f"{best_target[0]} : {best_target[3]:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.4 reconstructed Target CV : 98.3%")
    print("Binary relation chance         : 50.0%")
    print()
    print("Interpretation")
    print("--------------")
    print("A high leave-family-out relation score means TargetPosition semantics generalize")
    print("to unseen wording while both FIRST/SECOND labels remain represented in training.")
    print("Reconstructed Target combines that relation generalization with concept decoding")
    print("and deterministic semantic binding.")
    print("Output dir:",outdir)

if __name__=="__main__":
    main()
