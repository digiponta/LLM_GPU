# semantic_dataset_redesign_diagnostic_v134.py
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from semantic_router_dataset_v121 import DATASET as OLD_DATASET
from semantic_gpu_cpu_aligned_v134 import DATASET as NEW_DATASET, AXES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer


class LinearProbe(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.fc = nn.Linear(d_in, 2)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def block1_vector(model, tok, prompt):
    device = next(model.parameters()).device
    ids = tok.encode(f"人: {prompt}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)

    x = model.embedding(token_ids)
    if model.position_embedding is not None:
        pos = torch.arange(token_ids.shape[1], device=device)
        x = x + model.position_embedding(pos).unsqueeze(0)
    x = model.blocks[0](x)
    return x[0,-1,:].detach()


def standardize(Xtr, Xte):
    mean = Xtr.mean(0,keepdim=True)
    std = Xtr.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std, (Xte-mean)/std


def train_probe(X,y,epochs,lr,wd,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    p=LinearProbe(X.shape[1]).to(X.device)
    opt=torch.optim.AdamW(p.parameters(),lr=lr,weight_decay=wd)
    loss_fn=nn.CrossEntropyLoss()
    for _ in range(epochs):
        p.train(); opt.zero_grad()
        loss=loss_fn(p(X),y); loss.backward(); opt.step()
    p.eval()
    return p


def ensemble_accuracy(Xtr,ytr,Xte,yte,epochs,lr,wd,seeds):
    Xtr,Xte=standardize(Xtr,Xte)
    probs=[]
    for seed in seeds:
        p=train_probe(Xtr,ytr,epochs,lr,wd,seed)
        with torch.no_grad():
            probs.append(torch.softmax(p(Xte),dim=-1).cpu())
    pred=torch.stack(probs).mean(0).argmax(-1)
    ycpu=yte.cpu()
    acc=float((pred==ycpu).float().mean().item())
    gpu=float((pred[ycpu==0]==0).float().mean().item())
    cpu=float((pred[ycpu==1]==1).float().mean().item())
    return acc,gpu,cpu


def extract_dataset(model,tok,dataset):
    X=torch.stack([block1_vector(model,tok,prompt) for _,_,prompt in dataset])
    y=torch.tensor([0 if label=="gpu" else 1 for label,_,_ in dataset],dtype=torch.long,device=X.device)
    return X,y


def old_gpu_cpu():
    return [x for x in OLD_DATASET if x[0] in ("gpu","cpu")]


def old_leave_family_out(X,y,dataset,epochs,lr,wd,seeds):
    gpu_fams=list(dict.fromkeys(f for l,f,_ in dataset if l=="gpu"))
    cpu_fams=list(dict.fromkeys(f for l,f,_ in dataset if l=="cpu"))
    vals=[]
    for k in range(5):
        test=[i for i,(l,f,_) in enumerate(dataset) if
              (l=="gpu" and f==gpu_fams[k]) or (l=="cpu" and f==cpu_fams[k])]
        train=[i for i in range(len(dataset)) if i not in set(test)]
        vals.append(ensemble_accuracy(X[train],y[train],X[test],y[test],epochs,lr,wd,[s+1000*k for s in seeds]))
    return vals,gpu_fams,cpu_fams


def new_leave_axis_out(X,y,dataset,epochs,lr,wd,seeds):
    rows=[]
    for k,axis in enumerate(AXES):
        test=[i for i,(_,a,_) in enumerate(dataset) if a==axis]
        train=[i for i in range(len(dataset)) if i not in set(test)]
        r=ensemble_accuracy(X[train],y[train],X[test],y[test],epochs,lr,wd,[s+1000*k for s in seeds])
        rows.append((axis,*r))
    return rows


def within_axis_cv(X,y,dataset,epochs,lr,wd,seeds):
    rows=[]
    for ai,axis in enumerate(AXES):
        idx=[i for i,(_,a,_) in enumerate(dataset) if a==axis]
        gpu=[i for i in idx if dataset[i][0]=="gpu"]
        cpu=[i for i in idx if dataset[i][0]=="cpu"]
        probs=[]; gold=[]
        for fold in range(5):
            test=gpu[2*fold:2*fold+2]+cpu[2*fold:2*fold+2]
            train=[i for i in idx if i not in set(test)]
            Xtr,Xte=standardize(X[train],X[test])
            seed_probs=[]
            for seed in seeds:
                p=train_probe(Xtr,y[train],epochs,lr,wd,seed+ai*10000+fold*1000)
                with torch.no_grad():
                    seed_probs.append(torch.softmax(p(Xte),dim=-1).cpu())
            probs.append(torch.stack(seed_probs).mean(0))
            gold.append(y[test].cpu())
        pred=torch.cat(probs).argmax(-1)
        yy=torch.cat(gold)
        acc=float((pred==yy).float().mean().item())
        gpu_rec=float((pred[yy==0]==0).float().mean().item())
        cpu_rec=float((pred[yy==1]==1).float().mean().item())
        rows.append((axis,acc,gpu_rec,cpu_rec))
    return rows


def random_stratified_cv(X,y,dataset,epochs,lr,wd,seeds):
    # Deterministic 5-fold: each axis contributes 2 GPU and 2 CPU to every fold.
    folds=[[] for _ in range(5)]
    for axis in AXES:
        for label in ("gpu","cpu"):
            idx=[i for i,(l,a,_) in enumerate(dataset) if l==label and a==axis]
            for k in range(5):
                folds[k].extend(idx[2*k:2*k+2])

    oof=torch.zeros((len(dataset),2),dtype=torch.float32)
    for k,test in enumerate(folds):
        train=[i for i in range(len(dataset)) if i not in set(test)]
        Xtr,Xte=standardize(X[train],X[test])
        probs=[]
        for seed in seeds:
            p=train_probe(Xtr,y[train],epochs,lr,wd,seed+20000+k*1000)
            with torch.no_grad():
                probs.append(torch.softmax(p(Xte),dim=-1).cpu())
        oof[test]=torch.stack(probs).mean(0)
    pred=oof.argmax(-1)
    yy=y.cpu()
    return (
        float((pred==yy).float().mean().item()),
        float((pred[yy==0]==0).float().mean().item()),
        float((pred[yy==1]==1).float().mean().item()),
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=250)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--seeds",default="41,42,43")
    ap.add_argument("--output-dir",default="results/semantic_dataset_redesign_v134")
    args=ap.parse_args()

    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    old=old_gpu_cpu()
    new=NEW_DATASET

    print("="*108)
    print(" LLM_GPU v1.3.4 Semantic Dataset Redesign Diagnostic")
    print("="*108)
    print("Device             :",device)
    print("Base               :",args.model)
    print("Representation     : frozen Block1 final-token hidden")
    print("Probe              : Linear 256 -> 2")
    print("Old dataset        : 100 GPU/CPU prompts, asymmetric family ontology")
    print("Redesigned dataset : 100 GPU/CPU prompts, aligned axes")
    print("Aligned axes       :",", ".join(AXES))
    print("Seeds              :",seeds)
    print()

    print("Encoding old GPU/CPU dataset...")
    Xold,yold=extract_dataset(model,tok,old)
    print("Encoding redesigned aligned dataset...")
    Xnew,ynew=extract_dataset(model,tok,new)

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)

    old_loo,gf,cf=old_leave_family_out(Xold,yold,old,args.epochs,args.lr,args.weight_decay,seeds)
    print()
    print("OLD DATASET: paired-index leave-family-out")
    print("-------------------------------------------")
    old_rows=[]
    for k,(acc,gpu,cpu) in enumerate(old_loo):
        print(f"fold{k+1}: GPU={gf[k]:12s} CPU={cf[k]:12s} acc={acc:.1%} gpu={gpu:.1%} cpu={cpu:.1%}")
        old_rows.append([k+1,gf[k],cf[k],acc,gpu,cpu])
    old_mean=sum(r[0] for r in old_loo)/5
    print(f"Mean: {old_mean:.1%}")

    new_within=within_axis_cv(Xnew,ynew,new,args.epochs,args.lr,args.weight_decay,seeds)
    print()
    print("REDESIGNED DATASET: within-axis CV")
    print("----------------------------------")
    for axis,acc,gpu,cpu in new_within:
        print(f"{axis:12s} acc={acc:.1%} gpu={gpu:.1%} cpu={cpu:.1%}")
    within_mean=sum(r[1] for r in new_within)/len(new_within)
    print(f"Mean: {within_mean:.1%}")

    new_loo=new_leave_axis_out(Xnew,ynew,new,args.epochs,args.lr,args.weight_decay,seeds)
    print()
    print("REDESIGNED DATASET: leave-one-axis-out")
    print("--------------------------------------")
    for axis,acc,gpu,cpu in new_loo:
        print(f"{axis:12s} acc={acc:.1%} gpu={gpu:.1%} cpu={cpu:.1%}")
    loo_mean=sum(r[1] for r in new_loo)/len(new_loo)
    print(f"Mean: {loo_mean:.1%}")

    random_cv=random_stratified_cv(Xnew,ynew,new,args.epochs,args.lr,args.weight_decay,seeds)
    print()
    print("REDESIGNED DATASET: stratified mixed-axis 5-fold CV")
    print("---------------------------------------------------")
    print(f"acc={random_cv[0]:.1%} gpu={random_cv[1]:.1%} cpu={random_cv[2]:.1%}")

    with (outdir/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["dataset","evaluation","accuracy"])
        w.writerow(["old","paired-index_leave-family-out",old_mean])
        w.writerow(["redesigned","within-axis_mean",within_mean])
        w.writerow(["redesigned","leave-one-axis-out_mean",loo_mean])
        w.writerow(["redesigned","mixed-axis_5fold",random_cv[0]])

    with (outdir/"redesigned_within_axis.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["axis","accuracy","gpu_recall","cpu_recall"]); w.writerows(new_within)

    with (outdir/"redesigned_leave_axis_out.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["axis","accuracy","gpu_recall","cpu_recall"]); w.writerows(new_loo)

    with (outdir/"old_leave_family_out.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["fold","gpu_family","cpu_family","accuracy","gpu_recall","cpu_recall"]); w.writerows(old_rows)

    print()
    print("Diagnostic interpretation")
    print("-------------------------")
    print("If redesigned leave-one-axis-out is much higher than the old family-held-out")
    print("score, ontology alignment was a major source of the earlier apparent failure.")
    print("If within-axis and mixed-axis scores are high but leave-axis-out remains low,")
    print("the model still relies on semantic-axis-specific cues rather than a shared")
    print("GPU-vs-CPU abstraction.")
    print("This experiment changes only the evaluation/data ontology; base weights stay frozen.")
    print("Output dir:",outdir)


if __name__=="__main__":
    main()
