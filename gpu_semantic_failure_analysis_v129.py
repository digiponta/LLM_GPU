# gpu_semantic_failure_analysis_v129.py
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from semantic_router_dataset_v121 import DATASET, CLASSES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID={c:i for i,c in enumerate(CLASSES)}
STAGES=["block1","block2","block3","block4","block5","block6","final_norm"]
GPU="gpu"


class LinearProbe(nn.Module):
    def __init__(self,d_in,n_out):
        super().__init__()
        self.fc=nn.Linear(d_in,n_out)
    def forward(self,x):
        return self.fc(x)


def family_folds_for_all():
    fams=defaultdict(list)
    for label,family,_ in DATASET:
        fams[label].append(family)
    order={c:list(dict.fromkeys(fams[c])) for c in CLASSES}
    folds=[]
    for k in range(5):
        test=[i for i,(label,family,_) in enumerate(DATASET) if family==order[label][k]]
        assert len(test)==60
        folds.append(test)
    return folds,order


@torch.no_grad()
def extract_stages(model,tok,prompt):
    device=next(model.parameters()).device
    ids=tok.encode(f"人: {prompt}\nAI: ",add_bos=True)
    ids=ids[-model.context_length:]
    x=torch.tensor([ids],dtype=torch.long,device=device)

    h=model.embedding(x)
    if model.position_embedding is not None:
        pos=torch.arange(x.shape[1],device=device)
        h=h+model.position_embedding(pos).unsqueeze(0)

    out={}
    for i,block in enumerate(model.blocks,start=1):
        h=block(h)
        if i<=6:
            out[f"block{i}"]=h[:,-1,:][0].detach().cpu()
    h=model.final_norm(h)
    out["final_norm"]=h[:,-1,:][0].detach().cpu()
    return out


def standardize(Xtr,Xte):
    mean=Xtr.mean(0,keepdim=True)
    std=Xtr.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std,(Xte-mean)/std


def train_probe(X,y,n_out,epochs,lr,wd,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    m=LinearProbe(X.shape[1],n_out).to(X.device)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    loss_fn=nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss=loss_fn(m(X),y)
        loss.backward(); opt.step()
    m.eval()
    return m


@torch.no_grad()
def multiclass_oof(X,y,folds,epochs,lr,wd,seeds):
    n=len(y)
    probs=torch.zeros((n,len(CLASSES)),dtype=torch.float32)
    for fold_idx,test_idx in enumerate(folds):
        test_set=set(test_idx)
        train_idx=[i for i in range(n) if i not in test_set]
        Xtr,Xte=standardize(X[train_idx],X[test_idx])
        ytr=y[train_idx]
        ps=[]
        for seed in seeds:
            with torch.enable_grad():
                m=train_probe(Xtr,ytr,len(CLASSES),epochs,lr,wd,seed+1000*fold_idx)
            ps.append(torch.softmax(m(Xte),dim=-1).cpu())
        probs[test_idx]=torch.stack(ps).mean(0)
    return probs


def pairwise_folds(label_a,label_b,family_order):
    idxs=[i for i,(label,_,_) in enumerate(DATASET) if label in (label_a,label_b)]
    folds=[]
    for k in range(5):
        test=[
            i for i in idxs
            if (DATASET[i][0]==label_a and DATASET[i][1]==family_order[label_a][k])
            or (DATASET[i][0]==label_b and DATASET[i][1]==family_order[label_b][k])
        ]
        assert len(test)==20
        folds.append((idxs,test))
    return folds


@torch.no_grad()
def pairwise_oof(X,label_a,label_b,family_order,epochs,lr,wd,seeds):
    pairs=pairwise_folds(label_a,label_b,family_order)
    subset=sorted(set(i for idxs,_ in pairs for i in idxs))
    pos={idx:j for j,idx in enumerate(subset)}
    ybin=torch.tensor([0 if DATASET[i][0]==label_a else 1 for i in subset],dtype=torch.long,device=X.device)
    probs=torch.zeros((len(subset),2),dtype=torch.float32)

    for fold_idx,(all_idxs,test_idxs) in enumerate(pairs):
        train_idxs=[i for i in all_idxs if i not in set(test_idxs)]
        tr_pos=[pos[i] for i in train_idxs]
        te_pos=[pos[i] for i in test_idxs]

        Xtr,Xte=standardize(X[train_idxs],X[test_idxs])
        ytr=ybin[tr_pos]
        ps=[]
        for seed in seeds:
            with torch.enable_grad():
                m=train_probe(Xtr,ytr,2,epochs,lr,wd,seed+2000*fold_idx)
            ps.append(torch.softmax(m(Xte),dim=-1).cpu())
        probs[te_pos]=torch.stack(ps).mean(0)

    pred=probs.argmax(-1)
    acc=float((pred==ybin.cpu()).float().mean().item())
    return acc,probs,ybin.cpu(),subset


def write_matrix(path,header,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--epochs",type=int,default=250)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--weight-decay",type=float,default=1e-3)
    p.add_argument("--seeds",default="41,42,43")
    p.add_argument("--output-dir",default="results/gpu_semantic_failure_analysis_v129")
    args=p.parse_args()

    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)

    prompts=[x[2] for x in DATASET]
    y=torch.tensor([CLASS_TO_ID[x[0]] for x in DATASET],dtype=torch.long,device=device)

    print("="*102)
    print(" LLM_GPU v1.2.9 GPU Semantic Failure Analysis")
    print("="*102)
    print("Device          :",device)
    print("Base            :",args.model)
    print("Dataset         : 300 prompts, 6 classes x 5 families x 10")
    print("Stages          :",", ".join(STAGES))
    print("Multiclass probe: Linear 256 -> 6")
    print("Pairwise probe  : Linear 256 -> 2")
    print("Evaluation      : 5-fold family-held-out CV")
    print("Seeds/fold      :",seeds)
    print("Base model      : completely frozen")
    print()

    feats={stage:[] for stage in STAGES}
    print("Extracting layer-wise features...")
    for i,prompt in enumerate(prompts,start=1):
        vecs=extract_stages(model,tok,prompt)
        for stage in STAGES:
            feats[stage].append(vecs[stage])
        if i%50==0:
            print(f"  encoded {i}/300")
    feats={k:torch.stack(v).to(device) for k,v in feats.items()}

    folds,family_order=family_folds_for_all()
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)

    gpu_indices=[i for i,(label,_,_) in enumerate(DATASET) if label==GPU]
    family_names=family_order[GPU]

    multiclass_summary=[]
    family_rows=[]

    print()
    print("GPU multiclass failure modes by layer")
    print("-------------------------------------")
    for stage in STAGES:
        probs=multiclass_oof(feats[stage],y,folds,args.epochs,args.lr,args.weight_decay,seeds)
        pred=probs.argmax(-1)

        counts={c:0 for c in CLASSES}
        for i in gpu_indices:
            counts[CLASSES[int(pred[i])]]+=1

        correct=counts[GPU]
        print(
            f"{stage:10s} GPU={correct:2d}/50 ({correct/50:.1%}) "
            + " ".join(f"{c}:{counts[c]:2d}" for c in CLASSES if c!=GPU)
        )
        multiclass_summary.append([
            stage,correct/50,*[counts[c] for c in CLASSES]
        ])

        for fam in family_names:
            idxs=[i for i in gpu_indices if DATASET[i][1]==fam]
            fam_counts={c:0 for c in CLASSES}
            for i in idxs:
                fam_counts[CLASSES[int(pred[i])]]+=1
            family_rows.append([
                stage,fam,len(idxs),fam_counts[GPU],
                *[fam_counts[c] for c in CLASSES]
            ])

    with (outdir/"gpu_multiclass_by_layer.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["stage","gpu_recall",*[f"pred_{c}" for c in CLASSES]])
        w.writerows(multiclass_summary)

    with (outdir/"gpu_family_failure_modes.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["stage","gpu_family","n","gpu_correct",*[f"pred_{c}" for c in CLASSES]])
        w.writerows(family_rows)

    print()
    print("GPU pairwise separability")
    print("--------------------------")
    pair_rows=[]
    others=[c for c in CLASSES if c!=GPU]
    for stage in STAGES:
        vals=[]
        for other in others:
            acc,_,_,_=pairwise_oof(
                feats[stage],GPU,other,family_order,
                args.epochs,args.lr,args.weight_decay,seeds
            )
            vals.append(acc)
            pair_rows.append([stage,other,acc])
        print(
            f"{stage:10s} "
            + " ".join(f"GPU-vs-{o[:5]}={a:.1%}" for o,a in zip(others,vals))
        )

    with (outdir/"gpu_pairwise_separability.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["stage","other_class","accuracy"])
        w.writerows(pair_rows)

    # Best stage per pair
    print()
    print("Best stage per GPU pair")
    print("-----------------------")
    best_rows=[]
    for other in others:
        rows=[r for r in pair_rows if r[1]==other]
        best=max(rows,key=lambda r:r[2])
        print(f"GPU vs {other:11s}: {best[0]:10s} {best[2]:.1%}")
        best_rows.append([other,best[0],best[2]])

    with (outdir/"gpu_pairwise_best_stage.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["other_class","best_stage","best_accuracy"]); w.writerows(best_rows)

    print()
    print("Interpretation guide")
    print("--------------------")
    print("A low pairwise score identifies a true semantic boundary problem.")
    print("A high pairwise score with poor 6-class GPU recall means pairwise information")
    print("exists but is not organized into a stable multiclass GPU region.")
    print("Family rows show which GPU semantic families collapse into which neighboring class.")
    print("Output dir:",outdir)


if __name__=="__main__":
    main()
