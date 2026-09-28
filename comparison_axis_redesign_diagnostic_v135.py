# comparison_axis_redesign_diagnostic_v135.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from semantic_gpu_cpu_aligned_v134 import DATASET as ALIGNED_DATASET
from semantic_gpu_cpu_comparison_hard_v135 import DATASET as HARD_DATASET
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


def ensemble_eval(Xtr,ytr,Xte,yte,epochs,lr,wd,seeds):
    Xtr,Xte=standardize(Xtr,Xte)
    probs=[]
    for seed in seeds:
        p=train_probe(Xtr,ytr,epochs,lr,wd,seed)
        with torch.no_grad():
            probs.append(torch.softmax(p(Xte),dim=-1).cpu())
    pred=torch.stack(probs).mean(0).argmax(-1)
    yy=yte.cpu()
    acc=float((pred==yy).float().mean().item())
    gpu=float((pred[yy==0]==0).float().mean().item()) if bool((yy==0).any()) else 0.0
    cpu=float((pred[yy==1]==1).float().mean().item()) if bool((yy==1).any()) else 0.0
    return acc,gpu,cpu,pred


def old_comparison_dataset():
    return [(l,a,p) for l,a,p in ALIGNED_DATASET if a=="comparison"]


def hard_dataset():
    return [(l,a,pair,p) for l,a,pair,p in HARD_DATASET]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=250)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--seeds",default="41,42,43")
    ap.add_argument("--output-dir",default="results/comparison_axis_redesign_v135")
    args=ap.parse_args()

    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    old=old_comparison_dataset()
    hard=hard_dataset()

    print("="*108)
    print(" LLM_GPU v1.3.5 Comparison-Axis Redesign / Hard Contrast Diagnostic")
    print("="*108)
    print("Device            :",device)
    print("Base              :",args.model)
    print("Representation    : frozen Block1 final-token hidden")
    print("Probe             : Linear 256 -> 2")
    print("Old comparison    : 20 prompts from v1.3.4")
    print("Hard comparison   : 10 matched GPU/CPU topic pairs")
    print("Seeds             :",seeds)
    print()

    Xold=torch.stack([block1_vector(model,tok,p) for _,_,p in old])
    yold=torch.tensor([0 if l=="gpu" else 1 for l,_,_ in old],dtype=torch.long,device=device)
    Xhard=torch.stack([block1_vector(model,tok,p) for _,_,_,p in hard])
    yhard=torch.tensor([0 if l=="gpu" else 1 for l,_,_,_ in hard],dtype=torch.long,device=device)

    # A. Old comparison within-axis 5-fold (2 GPU + 2 CPU per fold).
    old_gpu=[i for i,(l,_,_) in enumerate(old) if l=="gpu"]
    old_cpu=[i for i,(l,_,_) in enumerate(old) if l=="cpu"]
    old_probs=[]; old_gold=[]
    for k in range(5):
        test=old_gpu[2*k:2*k+2]+old_cpu[2*k:2*k+2]
        train=[i for i in range(len(old)) if i not in set(test)]
        r=ensemble_eval(Xold[train],yold[train],Xold[test],yold[test],args.epochs,args.lr,args.weight_decay,[s+1000*k for s in seeds])
        old_probs.append(r[:3]); old_gold.append((k,*r[:3]))
    old_acc=sum(x[0] for x in old_probs)/5
    old_gpu_rec=sum(x[1] for x in old_probs)/5
    old_cpu_rec=sum(x[2] for x in old_probs)/5

    # B. Hard contrast matched-pair leave-one-pair-out.
    hard_rows=[]
    preds=[]
    golds=[]
    for pair_id in range(1,11):
        test=[i for i,(_,_,pid,_) in enumerate(hard) if pid==pair_id]
        train=[i for i in range(len(hard)) if i not in set(test)]
        acc,gpu,cpu,pred=ensemble_eval(
            Xhard[train],yhard[train],Xhard[test],yhard[test],
            args.epochs,args.lr,args.weight_decay,[s+2000*pair_id for s in seeds]
        )
        hard_rows.append([pair_id,acc,gpu,cpu])
        preds.append(pred); golds.append(yhard[test].cpu())
    all_pred=torch.cat(preds); all_gold=torch.cat(golds)
    hard_acc=float((all_pred==all_gold).float().mean().item())
    hard_gpu=float((all_pred[all_gold==0]==0).float().mean().item())
    hard_cpu=float((all_pred[all_gold==1]==1).float().mean().item())

    # C. Train on non-comparison aligned axes, test on old and hard comparison.
    noncomp=[x for x in ALIGNED_DATASET if x[1]!="comparison"]
    Xnon=torch.stack([block1_vector(model,tok,p) for _,_,p in noncomp])
    ynon=torch.tensor([0 if l=="gpu" else 1 for l,_,_ in noncomp],dtype=torch.long,device=device)

    transfer_old=ensemble_eval(Xnon,ynon,Xold,yold,args.epochs,args.lr,args.weight_decay,[s+30000 for s in seeds])
    transfer_hard=ensemble_eval(Xnon,ynon,Xhard,yhard,args.epochs,args.lr,args.weight_decay,[s+40000 for s in seeds])

    print("OLD COMPARISON: within-axis 5-fold")
    print("----------------------------------")
    print(f"acc={old_acc:.1%} gpu={old_gpu_rec:.1%} cpu={old_cpu_rec:.1%}")
    print()
    print("HARD COMPARISON: matched-pair leave-one-pair-out")
    print("-----------------------------------------------")
    print(f"acc={hard_acc:.1%} gpu={hard_gpu:.1%} cpu={hard_cpu:.1%}")
    print()
    print("TRANSFER FROM NON-COMPARISON AXES")
    print("---------------------------------")
    print(f"to old comparison : acc={transfer_old[0]:.1%} gpu={transfer_old[1]:.1%} cpu={transfer_old[2]:.1%}")
    print(f"to hard comparison: acc={transfer_hard[0]:.1%} gpu={transfer_hard[1]:.1%} cpu={transfer_hard[2]:.1%}")

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["evaluation","accuracy","gpu_recall","cpu_recall"])
        w.writerow(["old_comparison_within_axis",old_acc,old_gpu_rec,old_cpu_rec])
        w.writerow(["hard_comparison_leave_one_pair_out",hard_acc,hard_gpu,hard_cpu])
        w.writerow(["noncomparison_to_old_comparison",*transfer_old[:3]])
        w.writerow(["noncomparison_to_hard_comparison",*transfer_hard[:3]])

    with (outdir/"hard_pair_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["pair_id","accuracy","gpu_recall","cpu_recall"])
        w.writerows(hard_rows)

    print()
    print("Interpretation")
    print("--------------")
    print("If hard matched-pair accuracy is much higher than old comparison accuracy,")
    print("the old comparison wording/ontology was a major source of failure.")
    print("If hard within-pair is good but transfer from non-comparison axes remains low,")
    print("the model still lacks a comparison-invariant GPU/CPU abstraction.")
    print("If both are low, the Block1 representation itself is weak on relational comparison.")
    print("Output dir:",outdir)


if __name__=="__main__":
    main()
