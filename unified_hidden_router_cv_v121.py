# unified_hidden_router_cv_v121.py
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from semantic_router_dataset_v121 import DATASET, CLASSES
from fresh_generalization_cases_v114 import CASES as FRESH_V2
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID={c:i for i,c in enumerate(CLASSES)}

class LinearRouter(nn.Module):
    def __init__(self,d,n):
        super().__init__(); self.fc=nn.Linear(d,n)
    def forward(self,x): return self.fc(x)

class MLPRouter(nn.Module):
    def __init__(self,d,h,n):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Dropout(0.10),nn.Linear(h,n))
    def forward(self,x): return self.net(x)

@torch.no_grad()
def encode_hidden(model,tok,prompt):
    device=next(model.parameters()).device
    ids=tok.encode(f"人: {prompt}\nAI: ",add_bos=True)
    x=torch.tensor([ids[-model.context_length:]],dtype=torch.long,device=device)
    return model.forward_hidden(x)[:,-1,:][0].detach()

def standardize(Xtr,Xv):
    mean=Xtr.mean(0,keepdim=True)
    std=Xtr.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std,(Xv-mean)/std,mean,std

def make_model(arch,d,h,n):
    return LinearRouter(d,n) if arch=="linear" else MLPRouter(d,h,n)

def train_model(X,y,arch,hidden,epochs,lr,wd,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    m=make_model(arch,X.shape[1],hidden,len(CLASSES)).to(X.device)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    loss_fn=nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss=loss_fn(m(X),y); loss.backward(); opt.step()
    m.eval(); return m

def family_folds():
    # Five semantic families per class. Fold k holds out family k for every class.
    fams=defaultdict(list)
    for i,(label,family,prompt) in enumerate(DATASET):
        fams[label].append(family)
    order={}
    for label in CLASSES:
        order[label]=list(dict.fromkeys(fams[label]))
        assert len(order[label])==5
    folds=[]
    for k in range(5):
        test=[i for i,(label,family,_) in enumerate(DATASET) if family==order[label][k]]
        assert len(test)==60
        folds.append(test)
    return folds,order

@torch.no_grad()
def predict_ensemble(models,X):
    ps=[torch.softmax(m(X),dim=-1) for m in models]
    return torch.stack(ps).mean(0)

def confusion_metrics(y,probs):
    pred=probs.argmax(-1)
    cm=[[0]*len(CLASSES) for _ in CLASSES]
    for g,p in zip(y.tolist(),pred.tolist()): cm[g][p]+=1
    correct=sum(cm[i][i] for i in range(len(CLASSES)))
    per={}
    for i,c in enumerate(CLASSES):
        total=sum(cm[i]); hit=cm[i][i]
        per[c]=(hit,total,hit/total if total else 0.0)
    return correct/len(y),correct,cm,per,pred

def write_confusion(path,cm):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["gold"]+CLASSES)
        for i,c in enumerate(CLASSES): w.writerow([c]+cm[i])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--lr",type=float,default=8e-4)
    p.add_argument("--weight-decay",type=float,default=1e-3)
    p.add_argument("--hidden",type=int,default=64)
    p.add_argument("--seeds",default="41,42,43")
    p.add_argument("--output-dir",default="results/unified_hidden_router_cv_v121")
    args=p.parse_args()

    seeds=[int(x) for x in args.seeds.split(",")]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    base,_=LanguageModel.load_checkpoint(args.model,device=device); base.eval()
    for q in base.parameters(): q.requires_grad_(False)

    prompts=[x[2] for x in DATASET]
    y=torch.tensor([CLASS_TO_ID[x[0]] for x in DATASET],dtype=torch.long,device=device)
    X=torch.stack([encode_hidden(base,tok,s) for s in prompts])
    folds,family_order=family_folds()

    fresh=[(i,p) for i,p,_,_ in FRESH_V2 if i in CLASS_TO_ID]
    Xfresh=torch.stack([encode_hidden(base,tok,p) for _,p in fresh])
    yfresh=torch.tensor([CLASS_TO_ID[i] for i,_ in fresh],dtype=torch.long,device=device)

    print("="*94)
    print(" LLM_GPU v1.2.1 Expanded Unified Semantic Router - Family-Held-Out CV")
    print("="*94)
    print("Device          :",device)
    print("Expanded data   : 300 prompts (6 classes x 50)")
    print("Families/class  : 5")
    print("CV              : 5 folds, one unseen semantic family/class/fold")
    print("Seeds/fold      :",seeds)
    print("Architectures   : Linear, MLP(256->64->6)")
    print("Fresh-v2 check  : 60 technical prompts (development transfer check)")
    print()

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    summary=[]

    for arch in ("linear","mlp"):
        oof=torch.zeros((len(DATASET),len(CLASSES)),dtype=torch.float32)
        for fold_idx,test_idx in enumerate(folds):
            test_set=set(test_idx)
            train_idx=[i for i in range(len(DATASET)) if i not in test_set]
            Xtr,Xv,mean,std=standardize(X[train_idx],X[test_idx])
            yt=y[train_idx]
            models=[train_model(Xtr,yt,arch,args.hidden,args.epochs,args.lr,args.weight_decay,s+1000*fold_idx) for s in seeds]
            with torch.no_grad():
                probs=predict_ensemble(models,Xv).cpu()
            oof[test_idx]=probs

        cv_acc,cv_correct,cm,per,pred=confusion_metrics(y.cpu(),oof)
        print(arch.upper())
        print("-"*74)
        print(f"Family-held-out OOF: {cv_correct}/300 ({cv_acc:.1%})")
        for c in CLASSES:
            h,n,r=per[c]; print(f"  {c:11s} {h:2d}/{n:2d} ({r:.1%})")

        # Train on all 300 and transfer-check on Fresh-v2 technical 60.
        Xall,Xf,mean,std=standardize(X,Xfresh)
        full_models=[train_model(Xall,y,arch,args.hidden,args.epochs,args.lr,args.weight_decay,s+9000) for s in seeds]
        with torch.no_grad():
            pf=predict_ensemble(full_models,Xf).cpu()
        f_acc,f_correct,fcm,fper,fpred=confusion_metrics(yfresh.cpu(),pf)
        print(f"Fresh-v2 transfer : {f_correct}/60 ({f_acc:.1%})")
        for c in CLASSES:
            h,n,r=fper[c]; print(f"  fresh {c:11s} {h}/{n} ({r:.1%})")
        print()

        write_confusion(outdir/f"{arch}_cv_confusion.csv",cm)
        write_confusion(outdir/f"{arch}_fresh_v2_confusion.csv",fcm)

        rows=[]
        for i,(label,family,prompt) in enumerate(DATASET):
            ranks=torch.argsort(oof[i],descending=True)
            rows.append({
                "id":f"E{i+1:03d}","gold":label,"family":family,"prompt":prompt,
                "pred":CLASSES[int(ranks[0])],
                "correct":int(int(ranks[0])==CLASS_TO_ID[label]),
                "top1":float(oof[i,ranks[0]]),"top2_class":CLASSES[int(ranks[1])],
                "top2":float(oof[i,ranks[1]]),"margin":float(oof[i,ranks[0]]-oof[i,ranks[1]]),
                **{f"p_{c}":float(oof[i,j]) for j,c in enumerate(CLASSES)}
            })
        with (outdir/f"{arch}_cv_predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

        summary.append([arch,cv_acc,f_acc])

    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["architecture","family_cv_accuracy","fresh_v2_transfer_accuracy"]); w.writerows(summary)

    print("Comparison")
    print("----------")
    for arch,cv_acc,f_acc in summary:
        print(f"{arch:6s} family-CV={cv_acc:.1%}  Fresh-v2={f_acc:.1%}")
    print()
    print("Reference: v1.2.0 Fresh-v2 5-fold CV = 63.3% (60-sample dataset).")
    print("Goal: determine whether broader semantic training data improves GPU/CPU separation")
    print("without sacrificing CUDA/Python/Transformer recognition.")
    print("Output dir:",outdir)

if __name__=="__main__":
    main()
