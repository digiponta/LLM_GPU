# semantic_family_invariance_cv_v122.py
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_router_dataset_v121 import DATASET, CLASSES
from fresh_generalization_cases_v114 import CASES as FRESH_V2
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID={c:i for i,c in enumerate(CLASSES)}


class InvarianceRouter(nn.Module):
    def __init__(self,d_model=256,proj=64,n_classes=6):
        super().__init__()
        self.proj=nn.Sequential(
            nn.Linear(d_model,proj),
            nn.GELU(),
            nn.LayerNorm(proj),
        )
        self.head=nn.Linear(proj,n_classes)

    def forward(self,x):
        z=self.proj(x)
        logits=self.head(z)
        return z,logits


@torch.no_grad()
def encode_hidden(model,tok,prompt):
    device=next(model.parameters()).device
    ids=tok.encode(f"人: {prompt}\nAI: ",add_bos=True)
    x=torch.tensor([ids[-model.context_length:]],dtype=torch.long,device=device)
    return model.forward_hidden(x)[:,-1,:][0].detach()


def family_folds():
    fams=defaultdict(list)
    for label,family,_ in DATASET:
        fams[label].append(family)
    order={c:list(dict.fromkeys(fams[c])) for c in CLASSES}
    for c in CLASSES:
        assert len(order[c])==5
    folds=[]
    for k in range(5):
        test=[i for i,(label,family,_) in enumerate(DATASET) if family==order[label][k]]
        assert len(test)==60
        folds.append(test)
    return folds,order


def standardize(Xtr,Xv):
    mean=Xtr.mean(0,keepdim=True)
    std=Xtr.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std,(Xv-mean)/std,mean,std


def supervised_contrastive_loss(z,y,temp=0.1):
    z=F.normalize(z,dim=-1)
    sim=(z @ z.T)/temp
    n=z.shape[0]
    eye=torch.eye(n,dtype=torch.bool,device=z.device)
    sim=sim.masked_fill(eye,-1e9)

    same=(y[:,None]==y[None,:]) & (~eye)
    log_prob=sim - torch.logsumexp(sim,dim=1,keepdim=True)
    pos_count=same.sum(dim=1).clamp_min(1)
    loss=-(log_prob*same).sum(dim=1)/pos_count
    valid=same.sum(dim=1)>0
    return loss[valid].mean()


def train_model(X,y,proj,epochs,lr,wd,temp,lam,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    m=InvarianceRouter(X.shape[1],proj,len(CLASSES)).to(X.device)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    ce_fn=nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        z,logits=m(X)
        ce=ce_fn(logits,y)
        sup=sup_contrastive=supervised_contrastive_loss(z,y,temp=temp)
        loss=ce + lam*sup_contrastive
        loss.backward(); opt.step()
    m.eval(); return m


@torch.no_grad()
def predict_ensemble(models,X):
    probs=[]
    for m in models:
        _,logits=m(X)
        probs.append(torch.softmax(logits,dim=-1))
    return torch.stack(probs).mean(0)


def metrics(y,probs):
    pred=probs.argmax(-1)
    cm=[[0]*len(CLASSES) for _ in CLASSES]
    for g,p in zip(y.tolist(),pred.tolist()):
        cm[g][p]+=1
    correct=sum(cm[i][i] for i in range(len(CLASSES)))
    per={}
    for i,c in enumerate(CLASSES):
        total=sum(cm[i]); hit=cm[i][i]
        per[c]=(hit,total,hit/total if total else 0.0)
    macro=sum(v[2] for v in per.values())/len(per)
    return correct/len(y),correct,macro,cm,per,pred


def write_confusion(path,cm):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["gold"]+CLASSES)
        for i,c in enumerate(CLASSES):
            w.writerow([c]+cm[i])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--projection",type=int,default=64)
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--lr",type=float,default=8e-4)
    p.add_argument("--weight-decay",type=float,default=1e-3)
    p.add_argument("--temperature",type=float,default=0.10)
    p.add_argument("--lambdas",default="0.0,0.1,0.25,0.5,1.0")
    p.add_argument("--seeds",default="41,42,43")
    p.add_argument("--output-dir",default="results/semantic_family_invariance_v122")
    args=p.parse_args()

    lambdas=[float(x) for x in args.lambdas.split(",")]
    seeds=[int(x) for x in args.seeds.split(",")]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok=Tokenizer.load(args.tokenizer)
    base,_=LanguageModel.load_checkpoint(args.model,device=device); base.eval()
    for q in base.parameters(): q.requires_grad_(False)

    prompts=[x[2] for x in DATASET]
    y=torch.tensor([CLASS_TO_ID[x[0]] for x in DATASET],dtype=torch.long,device=device)
    X=torch.stack([encode_hidden(base,tok,p) for p in prompts])

    fresh=[(i,p) for i,p,_,_ in FRESH_V2 if i in CLASS_TO_ID]
    Xfresh=torch.stack([encode_hidden(base,tok,p) for _,p in fresh])
    yfresh=torch.tensor([CLASS_TO_ID[i] for i,_ in fresh],dtype=torch.long,device=device)

    folds,_=family_folds()
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)

    print("="*96)
    print(" LLM_GPU v1.2.2 Semantic Family Invariance Training")
    print("="*96)
    print("Device          :",device)
    print("Base            :",args.model)
    print("Dataset         : 300 prompts, 6 classes x 5 families x 10")
    print("Projection      :",f"256 -> {args.projection}")
    print("Loss            : CE + lambda * supervised contrastive")
    print("Temperature     :",args.temperature)
    print("Lambdas         :",lambdas)
    print("CV              : 5-fold family-held-out")
    print("Seeds/fold      :",seeds)
    print()

    summary=[]
    best=None

    for lam in lambdas:
        oof=torch.zeros((len(DATASET),len(CLASSES)),dtype=torch.float32)
        for fold_idx,test_idx in enumerate(folds):
            test_set=set(test_idx)
            train_idx=[i for i in range(len(DATASET)) if i not in test_set]
            Xtr,Xv,_,_=standardize(X[train_idx],X[test_idx])
            yt=y[train_idx]
            models=[
                train_model(Xtr,yt,args.projection,args.epochs,args.lr,args.weight_decay,
                            args.temperature,lam,s+1000*fold_idx)
                for s in seeds
            ]
            oof[test_idx]=predict_ensemble(models,Xv).cpu()

        cv_acc,cv_correct,macro,cm,per,_=metrics(y.cpu(),oof)

        Xall,Xf,_,_=standardize(X,Xfresh)
        full_models=[
            train_model(Xall,y,args.projection,args.epochs,args.lr,args.weight_decay,
                        args.temperature,lam,s+9000)
            for s in seeds
        ]
        pf=predict_ensemble(full_models,Xf).cpu()
        f_acc,f_correct,fmacro,fcm,fper,_=metrics(yfresh.cpu(),pf)

        print(f"lambda={lam:.2f}")
        print("-"*76)
        print(f"Family-held-out OOF : {cv_correct}/300 ({cv_acc:.1%}) macro={macro:.1%}")
        for c in CLASSES:
            h,n,r=per[c]; print(f"  {c:11s} {h:2d}/{n:2d} ({r:.1%})")
        print(f"Fresh-v2 transfer   : {f_correct}/60 ({f_acc:.1%}) macro={fmacro:.1%}")
        for c in CLASSES:
            h,n,r=fper[c]; print(f"  fresh {c:11s} {h}/{n} ({r:.1%})")
        print()

        tag=str(lam).replace(".","p")
        write_confusion(outdir/f"lambda_{tag}_cv_confusion.csv",cm)
        write_confusion(outdir/f"lambda_{tag}_fresh_v2_confusion.csv",fcm)

        row=[lam,cv_acc,macro,f_acc,fmacro]
        summary.append(row)
        key=(cv_acc,macro,f_acc)
        if best is None or key>best[0]:
            best=(key,lam)

    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["lambda","family_cv_accuracy","family_cv_macro_recall","fresh_v2_accuracy","fresh_v2_macro_recall"])
        w.writerows(summary)

    print("Comparison")
    print("----------")
    for lam,cv_acc,macro,f_acc,fmacro in summary:
        print(f"lambda={lam:4.2f}  family-CV={cv_acc:.1%}  macro={macro:.1%}  Fresh-v2={f_acc:.1%}")
    print()
    print(f"Best lambda by family-held-out CV: {best[1]:.2f}")
    print("Reference v1.2.1 linear family-CV: 53.0%")
    print("Reference v1.2.1 linear Fresh-v2: 81.7%")
    print()
    print("Interpretation:")
    print("A useful invariance objective should raise family-held-out CV without collapsing")
    print("the strong CUDA/Python/Transformer transfer performance. If lambda>0 does not")
    print("improve family-CV, the frozen base representation itself is the main bottleneck.")
    print("Output dir:",outdir)


if __name__=="__main__":
    main()
