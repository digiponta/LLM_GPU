# aligned_semantic_dataset_v2_eval_v136.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from semantic_aligned_v2_v136 import DATASET, CLASSES, AXES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID={c:i for i,c in enumerate(CLASSES)}

class LinearProbe(nn.Module):
    def __init__(self,d_in,n_out):
        super().__init__()
        self.fc=nn.Linear(d_in,n_out)
    def forward(self,x): return self.fc(x)

@torch.no_grad()
def block1_vector(model,tok,prompt):
    device=next(model.parameters()).device
    ids=tok.encode(f"人: {prompt}\nAI: ",add_bos=True)
    ids=ids[-model.context_length:]
    x=torch.tensor([ids],dtype=torch.long,device=device)
    h=model.embedding(x)
    if model.position_embedding is not None:
        pos=torch.arange(x.shape[1],device=device)
        h=h+model.position_embedding(pos).unsqueeze(0)
    h=model.blocks[0](h)
    return h[0,-1,:].detach()

def standardize(Xtr,Xte):
    mean=Xtr.mean(0,keepdim=True)
    std=Xtr.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std,(Xte-mean)/std

def train_probe(X,y,n_out,epochs,lr,wd,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    p=LinearProbe(X.shape[1],n_out).to(X.device)
    opt=torch.optim.AdamW(p.parameters(),lr=lr,weight_decay=wd)
    loss_fn=nn.CrossEntropyLoss()
    for _ in range(epochs):
        p.train(); opt.zero_grad()
        loss=loss_fn(p(X),y); loss.backward(); opt.step()
    p.eval(); return p

def evaluate_folds(X,y,folds,epochs,lr,wd,seeds,n_out):
    oof=torch.zeros((len(y),n_out),dtype=torch.float32)
    for k,test in enumerate(folds):
        train=[i for i in range(len(y)) if i not in set(test)]
        Xtr,Xte=standardize(X[train],X[test])
        probs=[]
        for seed in seeds:
            p=train_probe(Xtr,y[train],n_out,epochs,lr,wd,seed+1000*k)
            with torch.no_grad(): probs.append(torch.softmax(p(Xte),dim=-1).cpu())
        oof[test]=torch.stack(probs).mean(0)
    pred=oof.argmax(-1)
    yy=y.cpu()
    acc=float((pred==yy).float().mean().item())
    per={}
    for i,c in enumerate(CLASSES):
        mask=yy==i
        per[c]=float((pred[mask]==yy[mask]).float().mean().item())
    return acc,per,pred

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=250)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--seeds",default="41,42,43")
    ap.add_argument("--output-dir",default="results/aligned_semantic_dataset_v2_v136")
    args=ap.parse_args()

    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print("="*108)
    print(" LLM_GPU v1.3.6 Aligned Semantic Dataset v2 Evaluation")
    print("="*108)
    print("Device         :",device)
    print("Base           :",args.model)
    print("Dataset        : 300 prompts = 6 classes x 5 aligned axes x 10")
    print("Axes           :",", ".join(AXES))
    print("Representation : frozen Block1 final-token hidden")
    print("Probe          : Linear 256 -> 6")
    print("Seeds          :",seeds)
    print()

    print("Encoding Block1 features...")
    X=torch.stack([block1_vector(model,tok,p) for _,_,p in DATASET])
    y=torch.tensor([CLASS_TO_ID[l] for l,_,_ in DATASET],dtype=torch.long,device=device)

    # Mixed-axis stratified folds: 2 prompts/class/axis per fold.
    mixed=[[] for _ in range(5)]
    for c in CLASSES:
        for a in AXES:
            idx=[i for i,(l,ax,_) in enumerate(DATASET) if l==c and ax==a]
            for k in range(5):
                mixed[k].extend(idx[2*k:2*k+2])

    mixed_acc,mixed_per,_=evaluate_folds(X,y,mixed,args.epochs,args.lr,args.weight_decay,seeds,len(CLASSES))

    # Leave-one-axis-out: exactly one complete aligned axis held out across all classes.
    loo=[]
    for a in AXES:
        loo.append([i for i,(_,ax,_) in enumerate(DATASET) if ax==a])
    loo_acc,loo_per,loo_pred=evaluate_folds(X,y,loo,args.epochs,args.lr,args.weight_decay,seeds,len(CLASSES))

    print("MIXED-AXIS STRATIFIED 5-FOLD")
    print("----------------------------")
    print(f"accuracy={mixed_acc:.1%}")
    for c in CLASSES: print(f"  {c:11s} {mixed_per[c]:.1%}")

    print()
    print("LEAVE-ONE-AXIS-OUT")
    print("------------------")
    print(f"overall accuracy={loo_acc:.1%}")
    for a in AXES:
        idx=[i for i,(_,ax,_) in enumerate(DATASET) if ax==a]
        correct=sum(int(loo_pred[i])==int(y[i].cpu()) for i in idx)
        print(f"  {a:12s} {correct}/{len(idx)} ({correct/len(idx):.1%})")
    print("Per-class:")
    for c in CLASSES: print(f"  {c:11s} {loo_per[c]:.1%}")

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["evaluation","accuracy"])
        w.writerow(["mixed_axis_stratified_5fold",mixed_acc])
        w.writerow(["leave_one_axis_out",loo_acc])
        for c in CLASSES:
            w.writerow([f"mixed_recall_{c}",mixed_per[c]])
        for c in CLASSES:
            w.writerow([f"loo_recall_{c}",loo_per[c]])

    with (outdir/"leave_axis_breakdown.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["axis","accuracy"])
        for a in AXES:
            idx=[i for i,(_,ax,_) in enumerate(DATASET) if ax==a]
            correct=sum(int(loo_pred[i])==int(y[i].cpu()) for i in idx)
            w.writerow([a,correct/len(idx)])

    print()
    print("Interpretation")
    print("--------------")
    print("This is the first six-class evaluation where every class uses the same ontology axes.")
    print("Mixed-axis measures general routing under aligned data.")
    print("Leave-one-axis-out measures semantic transfer to a completely unseen meaning axis.")
    print("Comparison-axis performance should be interpreted against v1.3.5, where symmetric")
    print("hard-contrast wording greatly improved GPU/CPU comparison behavior.")
    print("Output dir:",outdir)

if __name__=="__main__":
    main()
