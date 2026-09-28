# gpu_cpu_boundary_training_v130.py
from __future__ import annotations

import argparse
import copy
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_router_dataset_v121 import DATASET, CLASSES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
GPU_ID = CLASS_TO_ID["gpu"]
CPU_ID = CLASS_TO_ID["cpu"]


class BoundaryAdapter(nn.Module):
    def __init__(self, base: LanguageModel, proj_dim: int = 64):
        super().__init__()
        self.block1 = copy.deepcopy(base.blocks[0])
        for p in self.block1.parameters():
            p.requires_grad_(True)
        self.projection = nn.Sequential(
            nn.Linear(base.d_model, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        self.binary_head = nn.Linear(proj_dim, 2)

    def forward(self, x):
        h = self.block1(x)
        last = h[:, -1, :]
        z = self.projection(last)
        logits = self.binary_head(z)
        return last, z, logits


class LinearProbe(nn.Module):
    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_out)
    def forward(self, x):
        return self.fc(x)


def family_order_and_folds():
    fams = defaultdict(list)
    for label,family,_ in DATASET:
        fams[label].append(family)
    order = {c:list(dict.fromkeys(fams[c])) for c in CLASSES}
    for c in CLASSES:
        assert len(order[c]) == 5

    folds=[]
    for k in range(5):
        test=[i for i,(label,family,_) in enumerate(DATASET) if family == order[label][k]]
        assert len(test)==60
        folds.append(test)
    return order,folds


def encode_ids(tok, model, prompt):
    ids = tok.encode(f"人: {prompt}\nAI: ", add_bos=True)
    return ids[-model.context_length:]


@torch.no_grad()
def embedding_input(base, ids):
    device=next(base.parameters()).device
    x=torch.tensor([ids],dtype=torch.long,device=device)
    h=base.embedding(x)
    if base.position_embedding is not None:
        pos=torch.arange(x.shape[1],device=device)
        h=h+base.position_embedding(pos).unsqueeze(0)
    return h[0].detach().cpu()


@torch.no_grad()
def original_block1_last(base, emb):
    device=next(base.parameters()).device
    h=base.blocks[0](emb.unsqueeze(0).to(device))
    return h[0,-1,:].detach().cpu()


def cache_inputs(base,tok):
    embs=[]; refs=[]; lengths=[]
    for _,_,prompt in DATASET:
        ids=encode_ids(tok,base,prompt)
        emb=embedding_input(base,ids)
        embs.append(emb)
        refs.append(original_block1_last(base,emb))
        lengths.append(len(ids))
    return embs,torch.stack(refs),lengths


def batches(indices,lengths,batch_size,seed):
    by_len=defaultdict(list)
    for i in indices:
        by_len[lengths[i]].append(i)
    rng=random.Random(seed)
    result=[]
    for idxs in by_len.values():
        idxs=list(idxs); rng.shuffle(idxs)
        for s in range(0,len(idxs),batch_size):
            result.append(idxs[s:s+batch_size])
    rng.shuffle(result)
    return result


def make_batch(embs,idxs,device):
    return torch.stack([embs[i] for i in idxs]).to(device)


def supcon(z,y,temp):
    z=F.normalize(z,dim=-1)
    sim=(z@z.T)/temp
    n=z.shape[0]
    eye=torch.eye(n,dtype=torch.bool,device=z.device)
    sim=sim.masked_fill(eye,-1e9)
    same=(y[:,None]==y[None,:]) & (~eye)
    logp=sim-torch.logsumexp(sim,dim=1,keepdim=True)
    count=same.sum(1).clamp_min(1)
    valid=same.sum(1)>0
    if not bool(valid.any()):
        return z.new_zeros(())
    return (-(logp*same).sum(1)/count)[valid].mean()


def train_adapter(base,embs,refs,lengths,train_pair_idx,*,
                  proj_dim,epochs,batch_size,block_lr,head_lr,wd,
                  contrastive_weight,preservation_weight,temp,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device=next(base.parameters()).device
    m=BoundaryAdapter(base,proj_dim).to(device)

    opt=torch.optim.AdamW([
        {"params":m.block1.parameters(),"lr":block_lr},
        {"params":list(m.projection.parameters())+list(m.binary_head.parameters()),"lr":head_lr},
    ],weight_decay=wd)
    ce_fn=nn.CrossEntropyLoss()

    for epoch in range(epochs):
        m.train()
        for idxs in batches(train_pair_idx,lengths,batch_size,seed+epoch):
            xb=make_batch(embs,idxs,device)
            yb=torch.tensor(
                [0 if DATASET[i][0]=="gpu" else 1 for i in idxs],
                dtype=torch.long,device=device
            )
            ref=refs[idxs].to(device)
            opt.zero_grad()
            h,z,logits=m(xb)
            ce=ce_fn(logits,yb)
            sc=supcon(z,yb,temp) if len(idxs)>1 else ce.new_zeros(())
            preserve=F.mse_loss(h,ref)
            loss=ce+contrastive_weight*sc+preservation_weight*preserve
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step()
    m.eval()
    return m


@torch.no_grad()
def adapted_features(model,embs,lengths,indices,device):
    out=torch.zeros((len(indices),model.block1.norm1.normalized_shape[0]),dtype=torch.float32)
    pos={idx:j for j,idx in enumerate(indices)}
    by_len=defaultdict(list)
    for i in indices:
        by_len[lengths[i]].append(i)
    for idxs in by_len.values():
        xb=make_batch(embs,idxs,device)
        h=model.block1(xb)[:,-1,:].cpu()
        for r,idx in enumerate(idxs):
            out[pos[idx]]=h[r]
    return out


@torch.no_grad()
def binary_probs(model,embs,lengths,indices,device):
    out=torch.zeros((len(indices),2),dtype=torch.float32)
    pos={idx:j for j,idx in enumerate(indices)}
    by_len=defaultdict(list)
    for i in indices:
        by_len[lengths[i]].append(i)
    for idxs in by_len.values():
        xb=make_batch(embs,idxs,device)
        _,_,logits=model(xb)
        p=torch.softmax(logits,dim=-1).cpu()
        for r,idx in enumerate(idxs):
            out[pos[idx]]=p[r]
    return out


def standardize(Xtr,Xte):
    mean=Xtr.mean(0,keepdim=True)
    std=Xtr.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std,(Xte-mean)/std


def train_probe(X,y,epochs,lr,wd,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    m=LinearProbe(X.shape[1],len(CLASSES)).to(X.device)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    loss_fn=nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss=loss_fn(m(X),y); loss.backward(); opt.step()
    m.eval(); return m


def multiclass_metrics(y,pred):
    cm=[[0]*len(CLASSES) for _ in CLASSES]
    for g,p in zip(y.tolist(),pred.tolist()):
        cm[g][p]+=1
    correct=sum(cm[i][i] for i in range(len(CLASSES)))
    per={}
    for i,c in enumerate(CLASSES):
        total=sum(cm[i]); hit=cm[i][i]
        per[c]=(hit,total,hit/total if total else 0.0)
    macro=sum(v[2] for v in per.values())/len(per)
    return correct/len(y),correct,macro,cm,per


def write_confusion(path,cm):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["gold"]+CLASSES)
        for i,c in enumerate(CLASSES):
            w.writerow([c]+cm[i])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--block-lrs",default="1e-5,3e-5,1e-4")
    p.add_argument("--head-lr",type=float,default=8e-4)
    p.add_argument("--probe-lr",type=float,default=1e-3)
    p.add_argument("--epochs",type=int,default=120)
    p.add_argument("--probe-epochs",type=int,default=250)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--projection",type=int,default=64)
    p.add_argument("--contrastive-weight",type=float,default=0.25)
    p.add_argument("--preservation-weight",type=float,default=0.10)
    p.add_argument("--temperature",type=float,default=0.10)
    p.add_argument("--weight-decay",type=float,default=1e-3)
    p.add_argument("--seeds",default="41,42")
    p.add_argument("--output-dir",default="results/gpu_cpu_boundary_training_v130")
    args=p.parse_args()

    block_lrs=[float(x) for x in args.block_lrs.split(",")]
    seeds=[int(x) for x in args.seeds.split(",")]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok=Tokenizer.load(args.tokenizer)
    base,_=LanguageModel.load_checkpoint(args.model,device=device)
    base.eval()
    for q in base.parameters():
        q.requires_grad_(False)

    family_order,folds=family_order_and_folds()
    y_all=torch.tensor([CLASS_TO_ID[x[0]] for x in DATASET],dtype=torch.long)

    print("="*104)
    print(" LLM_GPU v1.3.0 GPU-CPU Semantic Boundary Training")
    print("="*104)
    print("Device             :",device)
    print("Base               :",args.model)
    print("Adapted component  : Block1")
    print("Boundary target    : GPU vs CPU")
    print("Loss               : binary CE + 0.25*SupCon + 0.10*preservation")
    print("Block1 LR sweep    :",block_lrs)
    print("Head LR            :",args.head_lr)
    print("CV                 : 5-fold semantic-family-held-out")
    print("Seeds/fold         :",seeds)
    print("References         : pairwise GPU-vs-CPU=57.0%, 6-class Block1=60.7%")
    print()

    print("Caching embedding inputs and original Block1 references...")
    embs,refs,lengths=cache_inputs(base,tok)

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    summary=[]

    for block_lr in block_lrs:
        pair_correct=0
        pair_total=0
        six_oof=torch.zeros((len(DATASET),len(CLASSES)),dtype=torch.float32)

        for fold_idx,test_idx in enumerate(folds):
            test_set=set(test_idx)
            train_all=[i for i in range(len(DATASET)) if i not in test_set]
            train_pair=[i for i in train_all if DATASET[i][0] in ("gpu","cpu")]
            test_pair=[i for i in test_idx if DATASET[i][0] in ("gpu","cpu")]

            seed_pair_probs=[]
            seed_six_probs=[]

            for seed in seeds:
                adapter=train_adapter(
                    base,embs,refs,lengths,train_pair,
                    proj_dim=args.projection,epochs=args.epochs,batch_size=args.batch_size,
                    block_lr=block_lr,head_lr=args.head_lr,wd=args.weight_decay,
                    contrastive_weight=args.contrastive_weight,
                    preservation_weight=args.preservation_weight,temp=args.temperature,
                    seed=seed+1000*fold_idx,
                )

                pp=binary_probs(adapter,embs,lengths,test_pair,device)
                seed_pair_probs.append(pp)

                train_feat=adapted_features(adapter,embs,lengths,train_all,device).to(device)
                test_feat=adapted_features(adapter,embs,lengths,test_idx,device).to(device)
                train_feat,test_feat=standardize(train_feat,test_feat)
                ytr=y_all[train_all].to(device)

                probe=train_probe(
                    train_feat,ytr,args.probe_epochs,args.probe_lr,args.weight_decay,
                    seed+5000*fold_idx
                )
                with torch.no_grad():
                    seed_six_probs.append(torch.softmax(probe(test_feat),dim=-1).cpu())

            pair_p=torch.stack(seed_pair_probs).mean(0)
            pair_pred=pair_p.argmax(-1)
            pair_gold=torch.tensor(
                [0 if DATASET[i][0]=="gpu" else 1 for i in test_pair],
                dtype=torch.long
            )
            pair_correct+=int((pair_pred==pair_gold).sum().item())
            pair_total+=len(test_pair)

            six_oof[test_idx]=torch.stack(seed_six_probs).mean(0)

        six_pred=six_oof.argmax(-1)
        six_acc,six_correct,six_macro,cm,per=multiclass_metrics(y_all,six_pred)
        pair_acc=pair_correct/pair_total

        print()
        print(f"block_lr={block_lr:.1e}")
        print("-"*84)
        print(f"GPU-vs-CPU held-out : {pair_correct}/{pair_total} ({pair_acc:.1%})")
        print(f"6-class held-out    : {six_correct}/300 ({six_acc:.1%}) macro={six_macro:.1%}")
        for c in CLASSES:
            h,n,r=per[c]
            print(f"  {c:11s} {h:2d}/{n:2d} ({r:.1%})")

        tag=f"{block_lr:.0e}".replace("-","m")
        write_confusion(outdir/f"lr_{tag}_six_class_confusion.csv",cm)
        summary.append([
            block_lr,pair_acc,six_acc,six_macro,
            *[per[c][2] for c in CLASSES]
        ])

    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow([
            "block1_lr","gpu_cpu_pairwise_accuracy","six_class_accuracy","six_class_macro",
            *[f"recall_{c}" for c in CLASSES]
        ])
        w.writerows(summary)

    print()
    print("Comparison")
    print("----------")
    for row in summary:
        print(
            f"lr={row[0]:.1e} pair={row[1]:.1%} six={row[2]:.1%} "
            f"gpu={row[4]:.1%} cpu={row[5]:.1%} llm={row[6]:.1%} "
            f"trans={row[7]:.1%} cuda={row[8]:.1%} python={row[9]:.1%}"
        )

    best=max(summary,key=lambda r:(r[1],r[2]))
    print()
    print(f"Best pairwise setting: block_lr={best[0]:.1e} pair={best[1]:.1%} six={best[2]:.1%}")
    print("Target: GPU-vs-CPU >= 75-80% while preserving useful 6-class performance.")
    print("Important: this is development CV on the existing 300-prompt dataset.")
    print("An untouched Fresh-v3 set is still required for final validation.")
    print("Output dir:",outdir)


if __name__=="__main__":
    main()
