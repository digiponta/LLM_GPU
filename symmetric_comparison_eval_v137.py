# symmetric_comparison_eval_v137.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from semantic_aligned_v2_v136 import DATASET as OLD_DATASET, CLASSES, AXES
from semantic_aligned_v3_v137 import DATASET as NEW_DATASET
from semantic_symmetric_comparison_v137 import PAIR_META
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

def evaluate_folds(X,y,dataset,folds,args,seeds):
    oof=torch.zeros((len(dataset),len(CLASSES)),dtype=torch.float32)
    for k,test in enumerate(folds):
        train=[i for i in range(len(dataset)) if i not in set(test)]
        Xtr,Xte=standardize(X[train],X[test])
        probs=[]
        for seed in seeds:
            p=train_probe(Xtr,y[train],len(CLASSES),args.epochs,args.lr,args.weight_decay,seed+1000*k)
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

def build_mixed(dataset):
    folds=[[] for _ in range(5)]
    for c in CLASSES:
        for a in AXES:
            idx=[i for i,(l,ax,_) in enumerate(dataset) if l==c and ax==a]
            for k in range(5): folds[k].extend(idx[2*k:2*k+2])
    return folds

def build_axis_loo(dataset):
    return [[i for i,(_,ax,_) in enumerate(dataset) if ax==a] for a in AXES]

def comparison_within_cv(X,y,dataset,args,seeds):
    idx=[i for i,(_,a,_) in enumerate(dataset) if a=="comparison"]
    folds=[[] for _ in range(5)]
    for c in CLASSES:
        cidx=[i for i in idx if dataset[i][0]==c]
        for k in range(5): folds[k].extend(cidx[2*k:2*k+2])

    probs={}
    correct=0
    total=0
    per_correct={c:0 for c in CLASSES}
    per_total={c:0 for c in CLASSES}

    for k,test in enumerate(folds):
        train=[i for i in idx if i not in set(test)]
        Xtr,Xte=standardize(X[train],X[test])
        ps=[]
        for seed in seeds:
            p=train_probe(Xtr,y[train],len(CLASSES),args.epochs,args.lr,args.weight_decay,seed+5000+k*1000)
            with torch.no_grad(): ps.append(torch.softmax(p(Xte),dim=-1).cpu())
        pred=torch.stack(ps).mean(0).argmax(-1)
        gold=y[test].cpu()
        correct+=int((pred==gold).sum().item())
        total+=len(test)
        for j,i in enumerate(test):
            c=dataset[i][0]
            per_total[c]+=1
            per_correct[c]+=int(pred[j])==int(gold[j])
    return correct/total,{c:per_correct[c]/per_total[c] for c in CLASSES}

def transfer_noncomparison_to_comparison(X,y,dataset,args,seeds):
    train=[i for i,(_,a,_) in enumerate(dataset) if a!="comparison"]
    test=[i for i,(_,a,_) in enumerate(dataset) if a=="comparison"]
    Xtr,Xte=standardize(X[train],X[test])
    ps=[]
    for seed in seeds:
        p=train_probe(Xtr,y[train],len(CLASSES),args.epochs,args.lr,args.weight_decay,seed+30000)
        with torch.no_grad(): ps.append(torch.softmax(p(Xte),dim=-1).cpu())
    pred=torch.stack(ps).mean(0).argmax(-1)
    gold=y[test].cpu()
    acc=float((pred==gold).float().mean().item())
    per={}
    for ci,c in enumerate(CLASSES):
        mask=gold==ci
        per[c]=float((pred[mask]==gold[mask]).float().mean().item())
    return acc,per

def encode_dataset(model,tok,dataset):
    X=torch.stack([block1_vector(model,tok,p) for _,_,p in dataset])
    y=torch.tensor([CLASS_TO_ID[l] for l,_,_ in dataset],dtype=torch.long,device=X.device)
    return X,y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=250)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--seeds",default="41,42,43")
    ap.add_argument("--output-dir",default="results/symmetric_comparison_v137")
    args=ap.parse_args()

    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print("="*110)
    print(" LLM_GPU v1.3.7 Six-Class Symmetric Comparison Dataset")
    print("="*110)
    print("Device         :",device)
    print("Base           :",args.model)
    print("Representation : frozen Block1 final-token hidden")
    print("Probe          : Linear 256 -> 6")
    print("Comparison     : complete 6-class pair graph, 2 reciprocal topics/pair")
    print("Pairs          : 30 reciprocal semantic topics = 60 prompts")
    print("Seeds          :",seeds)
    print()

    print("Encoding v1.3.6 baseline dataset...")
    Xold,yold=encode_dataset(model,tok,OLD_DATASET)
    print("Encoding v1.3.7 symmetric-comparison dataset...")
    Xnew,ynew=encode_dataset(model,tok,NEW_DATASET)

    old_comp,old_comp_per=comparison_within_cv(Xold,yold,OLD_DATASET,args,seeds)
    new_comp,new_comp_per=comparison_within_cv(Xnew,ynew,NEW_DATASET,args,seeds)

    old_transfer,old_transfer_per=transfer_noncomparison_to_comparison(Xold,yold,OLD_DATASET,args,seeds)
    new_transfer,new_transfer_per=transfer_noncomparison_to_comparison(Xnew,ynew,NEW_DATASET,args,seeds)

    mixed_acc,mixed_per,_=evaluate_folds(Xnew,ynew,NEW_DATASET,build_mixed(NEW_DATASET),args,seeds)
    loo_acc,loo_per,loo_pred=evaluate_folds(Xnew,ynew,NEW_DATASET,build_axis_loo(NEW_DATASET),args,seeds)

    axis_scores={}
    for a in AXES:
        idx=[i for i,(_,ax,_) in enumerate(NEW_DATASET) if ax==a]
        correct=sum(int(loo_pred[i])==int(ynew[i].cpu()) for i in idx)
        axis_scores[a]=correct/len(idx)

    print()
    print("COMPARISON WITHIN-AXIS")
    print("----------------------")
    print(f"v1.3.6 old comparison : {old_comp:.1%}")
    print(f"v1.3.7 symmetric      : {new_comp:.1%}")
    print("Per-class symmetric comparison:")
    for c in CLASSES: print(f"  {c:11s} {new_comp_per[c]:.1%}")

    print()
    print("NON-COMPARISON -> COMPARISON TRANSFER")
    print("-------------------------------------")
    print(f"v1.3.6 old comparison : {old_transfer:.1%}")
    print(f"v1.3.7 symmetric      : {new_transfer:.1%}")
    print("Per-class symmetric transfer:")
    for c in CLASSES: print(f"  {c:11s} {new_transfer_per[c]:.1%}")

    print()
    print("FULL v1.3.7 DATASET")
    print("-------------------")
    print(f"mixed-axis stratified 5-fold : {mixed_acc:.1%}")
    print(f"leave-one-axis-out overall   : {loo_acc:.1%}")
    print("Leave-axis breakdown:")
    for a in AXES: print(f"  {a:12s} {axis_scores[a]:.1%}")
    print("Leave-axis per-class:")
    for c in CLASSES: print(f"  {c:11s} {loo_per[c]:.1%}")

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["v136_comparison_within",old_comp])
        w.writerow(["v137_comparison_within",new_comp])
        w.writerow(["v136_noncomparison_to_comparison",old_transfer])
        w.writerow(["v137_noncomparison_to_comparison",new_transfer])
        w.writerow(["v137_mixed_axis",mixed_acc])
        w.writerow(["v137_leave_one_axis_out",loo_acc])
        for a in AXES: w.writerow([f"v137_axis_{a}",axis_scores[a]])
        for c in CLASSES: w.writerow([f"v137_loo_recall_{c}",loo_per[c]])

    with (outdir/"comparison_per_class.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["class","within_symmetric","transfer_to_symmetric"])
        for c in CLASSES: w.writerow([c,new_comp_per[c],new_transfer_per[c]])

    print()
    print("Decision")
    print("--------")
    print("v1.3.6 references: mixed=82.3%, leave-axis-out=75.0%, comparison-axis=55.0%.")
    print("If symmetric comparison raises comparison accuracy and keeps overall metrics stable,")
    print("comparison wording/contrast design is confirmed as the remaining ontology bottleneck.")
    print("If LLM recall also rises, the LLM-vs-Transformer symmetric pairs corrected a major")
    print("class-boundary weakness without changing any base-model weights.")
    print("Output dir:",outdir)

if __name__=="__main__":
    main()
