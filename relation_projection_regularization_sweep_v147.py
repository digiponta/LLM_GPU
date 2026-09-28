# relation_projection_regularization_sweep_v147.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_explicit_role_span_v143 import DATASET as CONCEPT_DATASET, CLASSES
from semantic_relation_paraphrase_v145 import RELATION_SAMPLES, FAMILIES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer
from relation_paraphrase_generalization_v145 import encode_text_stages, concept_cv_features

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
STAGE = "block5"

ARCHES = [
    ("raw", None, None),
    ("linear32", 0, 32),
    ("mlp16_16", 16, 16),
    ("mlp64_32", 64, 32),
]
SUPCON_WEIGHTS = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]


class Projection(nn.Module):
    def __init__(self, d_in, hidden, proj_dim):
        super().__init__()
        if hidden == 0:
            self.net = nn.Linear(d_in, proj_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, proj_dim),
            )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class Head(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.fc = nn.Linear(d_in, 2)

    def forward(self, x):
        return self.fc(x)


def standardize_fit(X):
    mean=X.mean(0,keepdim=True)
    std=X.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
    return mean,std


def standardize_apply(X,mean,std):
    return (X-mean)/std


def supcon_loss(z,y,temp):
    n=z.shape[0]
    sim=(z@z.T)/temp
    eye=torch.eye(n,dtype=torch.bool,device=z.device)
    same=y[:,None].eq(y[None,:]) & ~eye
    logits=sim.masked_fill(eye,float("-inf"))
    log_prob=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    pos=same.sum(1).clamp_min(1)
    return -(log_prob.masked_fill(~same,0.0).sum(1)/pos).mean()


def train_model(Xtr,ytr,arch,hidden,proj_dim,args,seed,sup_w):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if arch=="raw":
        proj=None
        head=Head(Xtr.shape[1]).to(Xtr.device)
        params=head.parameters()
    else:
        proj=Projection(Xtr.shape[1],hidden,proj_dim).to(Xtr.device)
        head=Head(proj_dim).to(Xtr.device)
        params=list(proj.parameters())+list(head.parameters())

    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=args.weight_decay)

    for _ in range(args.epochs):
        opt.zero_grad()
        if proj is None:
            z=Xtr
            logits=head(z)
            loss=F.cross_entropy(logits,ytr)
        else:
            z=proj(Xtr)
            logits=head(z)
            ce=F.cross_entropy(logits,ytr)
            loss=ce + sup_w*supcon_loss(z,ytr,args.temperature)
        loss.backward()
        opt.step()

    if proj is not None:
        proj.eval()
    head.eval()
    return proj,head


@torch.no_grad()
def predict_with_model(proj,head,X):
    z=X if proj is None else proj(X)
    return torch.softmax(head(z),dim=-1), z


def family_cv(X,y,arch,hidden,proj_dim,sup_w,args,seeds):
    pred_all=torch.zeros(len(y),dtype=torch.long)
    family_scores={}
    centroid_cosines=[]
    within_scores=[]

    for fi,fam in enumerate(FAMILIES):
        test=[i for i,s in enumerate(RELATION_SAMPLES) if s["family"]==fam]
        train=[i for i,s in enumerate(RELATION_SAMPLES) if s["family"]!=fam]

        mean,std=standardize_fit(X[train])
        Xtr=standardize_apply(X[train],mean,std)
        Xte=standardize_apply(X[test],mean,std)

        probs=[]
        fold_cent=[]
        fold_within=[]

        for seed in seeds:
            proj,head=train_model(
                Xtr,y[train],arch,hidden,proj_dim,args,
                seed+1000*fi,sup_w
            )
            p,ztr=predict_with_model(proj,head,Xtr)
            pte,_=predict_with_model(proj,head,Xte)
            probs.append(pte.cpu())

            if proj is not None:
                a=ztr[y[train]==0]
                b=ztr[y[train]==1]
                c0=F.normalize(a.mean(0,keepdim=True),dim=-1)
                c1=F.normalize(b.mean(0,keepdim=True),dim=-1)
                fold_cent.append(float((c0*c1).sum()))
                fold_within.append(float(((a@c0.T).mean()+(b@c1.T).mean())/2))

        pred=torch.stack(probs).mean(0).argmax(-1)
        pred_all[test]=pred
        family_scores[fam]=float((pred==y[test].cpu()).float().mean())

        if fold_cent:
            centroid_cosines.extend(fold_cent)
            within_scores.extend(fold_within)

    overall=float((pred_all==y.cpu()).float().mean())
    centroid=sum(centroid_cosines)/len(centroid_cosines) if centroid_cosines else float("nan")
    within=sum(within_scores)/len(within_scores) if within_scores else float("nan")
    return overall,pred_all,family_scores,centroid,within


def reconstruct_target(pred_rel,p1,p2):
    hits=[]
    for i,s in enumerate(CONCEPT_DATASET):
        desired=0 if s["target"]==s["first_class"] else 1
        candidates=[j for j,r in enumerate(RELATION_SAMPLES) if r["target_position"]==desired]
        for j in candidates:
            pos=int(pred_rel[j])
            pred=int(p1[i]) if pos==0 else int(p2[i])
            hits.append(pred==CLASS_TO_ID[s["target"]])
    return sum(hits)/len(hits)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=400)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--temperature",type=float,default=0.1)
    ap.add_argument("--seeds",default="41,42,43")
    ap.add_argument("--output-dir",default="results/relation_projection_regularization_sweep_v147")
    args=ap.parse_args()

    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print("="*118)
    print(" LLM_GPU v1.4.7 Relation Projection Regularization Sweep")
    print("="*118)
    print("Device          :",device)
    print("Base            :",args.model)
    print("Stage           :",STAGE)
    print("Evaluation      : leave-one-relation-family-out")
    print("Architectures   :",", ".join(a[0] for a in ARCHES))
    print("SupCon weights  :",", ".join(str(x) for x in SUPCON_WEIGHTS))
    print("Base model      : completely frozen")
    print()

    cache={s["text"]:encode_text_stages(model,tok,s["text"]) for s in RELATION_SAMPLES}
    X=torch.stack([cache[s["text"]][STAGE] for s in RELATION_SAMPLES]).to(device)
    y=torch.tensor([s["target_position"] for s in RELATION_SAMPLES],dtype=torch.long,device=device)

    print("Preparing concept decoder reference...")
    p1,p2=concept_cv_features(model,tok,STAGE,args,seeds)

    rows=[]
    best=None

    for arch,hidden,proj_dim in ARCHES:
        weights=[0.0] if arch=="raw" else SUPCON_WEIGHTS
        for sup_w in weights:
            rel,pred,fams,centroid,within=family_cv(
                X,y,arch,hidden,proj_dim,sup_w,args,seeds
            )
            target=reconstruct_target(pred,p1,p2)
            row={
                "arch":arch,
                "supcon_weight":sup_w,
                "relation_accuracy":rel,
                "target_accuracy":target,
                "centroid_cosine":centroid,
                "within":within,
                **{f"family_{f}":fams[f] for f in FAMILIES},
            }
            rows.append(row)

            print(
                f"{arch:10s} sup={sup_w:>4.2f} "
                f"relation={rel:>6.1%} target={target:>6.1%} "
                f"centroid={centroid:+.3f} within={within:.3f}"
            )

            if best is None or (rel,target)>(best["relation_accuracy"],best["target_accuracy"]):
                best=row

    outdir=Path(args.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)

    fields=[
        "arch","supcon_weight","relation_accuracy","target_accuracy",
        "centroid_cosine","within",*[f"family_{f}" for f in FAMILIES]
    ]
    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    print()
    print("Best unseen relation generalization")
    print("-----------------------------------")
    print(
        f'{best["arch"]} sup={best["supcon_weight"]:.2f} : '
        f'{best["relation_accuracy"]:.1%}'
    )
    print(f'Reconstructed Target                 : {best["target_accuracy"]:.1%}')
    print()
    print("Reference")
    print("---------")
    print("v1.4.5 best unseen relation accuracy : 60.0%")
    print("v1.4.6 unseen relation accuracy      : 55.0%")
    print("Binary chance                        : 50.0%")
    print()
    print("Interpretation")
    print("--------------")
    print("The sweep tests whether unseen-family transfer improves when projection")
    print("capacity and contrastive pressure are reduced. Generalization accuracy")
    print("is the primary criterion; extreme training-space separation is diagnostic")
    print("only and is not treated as success by itself.")
    print("Output dir:",outdir)


if __name__=="__main__":
    main()
