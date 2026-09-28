# relation_dual_generalization_v1411.py
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_relation_hard_expanded_v1410 import RELATION_SAMPLES, FAMILIES
from semantic_explicit_role_span_v143 import DATASET as CONCEPT_DATASET, CLASSES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer
from relation_paraphrase_generalization_v145 import encode_text_stages, concept_cv_features

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
STAGE = "block5"
HIDDEN = 16
PROJ_DIM = 16
SUPCON_WEIGHT = 0.25
CONCEPT_SEEDS = [41,42,43]
N_FOLDS = 5


class Projection(nn.Module):
    def __init__(self,d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,HIDDEN),
            nn.GELU(),
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN,PROJ_DIM),
        )

    def forward(self,x):
        return F.normalize(self.net(x),dim=-1)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(PROJ_DIM,2)

    def forward(self,x):
        return self.fc(x)


def supcon(z,y,temp):
    n = z.shape[0]
    sim = (z @ z.T) / temp
    eye = torch.eye(n,dtype=torch.bool,device=z.device)
    same = y[:,None].eq(y[None,:]) & ~eye
    logits = sim.masked_fill(eye,float("-inf"))
    logp = logits - torch.logsumexp(logits,dim=1,keepdim=True)
    pos = same.sum(1).clamp_min(1)
    return -(logp.masked_fill(~same,0).sum(1)/pos).mean()


def standardize(Xtr,Xte):
    m = Xtr.mean(0,keepdim=True)
    s = Xtr.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
    return (Xtr-m)/s,(Xte-m)/s


def train_predict(X,y,train,test,args,seed):
    Xtr,Xte = standardize(X[train],X[test])

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    proj = Projection(X.shape[1]).to(X.device)
    head = Head().to(X.device)
    opt = torch.optim.AdamW(
        list(proj.parameters())+list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for _ in range(args.epochs):
        opt.zero_grad()
        z = proj(Xtr)
        loss = (
            F.cross_entropy(head(z),y[train])
            + SUPCON_WEIGHT * supcon(z,y[train],args.temperature)
        )
        loss.backward()
        opt.step()

    proj.eval()
    head.eval()
    with torch.no_grad():
        return torch.softmax(head(proj(Xte)),dim=-1).cpu()


def reconstruct(pred_rel,p1,p2):
    hits=[]
    for i,s in enumerate(CONCEPT_DATASET):
        desired = 0 if s["target"]==s["first_class"] else 1
        candidates = [
            j for j,r in enumerate(RELATION_SAMPLES)
            if r["target_position"]==desired
        ]
        for j in candidates:
            pos = int(pred_rel[j])
            pred_target = int(p1[i]) if pos==0 else int(p2[i])
            hits.append(pred_target==CLASS_TO_ID[s["target"]])
    return sum(hits)/len(hits)


def family_holdout(X,y,args,seed):
    pred = torch.zeros(len(y),dtype=torch.long)
    fam_scores = {}

    for fi,fam in enumerate(FAMILIES):
        test = [i for i,s in enumerate(RELATION_SAMPLES) if s["family"]==fam]
        train = [i for i,s in enumerate(RELATION_SAMPLES) if s["family"]!=fam]
        probs = train_predict(X,y,train,test,args,seed+1000*fi)
        pr = probs.argmax(-1)
        pred[test] = pr
        fam_scores[fam] = float((pr==y[test].cpu()).float().mean())

    micro = float((pred==y.cpu()).float().mean())
    macro = statistics.mean(fam_scores.values())
    return micro,macro,pred,fam_scores


def build_within_family_folds():
    folds=[[] for _ in range(N_FOLDS)]
    for fam in FAMILIES:
        for label in (0,1):
            idx=[
                i for i,s in enumerate(RELATION_SAMPLES)
                if s["family"]==fam and s["target_position"]==label
            ]
            assert len(idx) % N_FOLDS == 0
            chunk=len(idx)//N_FOLDS
            for k in range(N_FOLDS):
                folds[k].extend(idx[k*chunk:(k+1)*chunk])
    return folds


def within_family_holdout(X,y,args,seed):
    folds=build_within_family_folds()
    pred=torch.zeros(len(y),dtype=torch.long)

    for k,test in enumerate(folds):
        ts=set(test)
        train=[i for i in range(len(y)) if i not in ts]
        probs=train_predict(X,y,train,test,args,seed+10000+1000*k)
        pred[test]=probs.argmax(-1)

    fam_scores={}
    for fam in FAMILIES:
        idx=[i for i,s in enumerate(RELATION_SAMPLES) if s["family"]==fam]
        fam_scores[fam]=float((pred[idx]==y[idx].cpu()).float().mean())

    micro=float((pred==y.cpu()).float().mean())
    macro=statistics.mean(fam_scores.values())
    return micro,macro,pred,fam_scores


def mean_std(xs):
    if len(xs)==1:
        return float(xs[0]),0.0
    return statistics.mean(xs),statistics.stdev(xs)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=400)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--temperature",type=float,default=0.1)
    ap.add_argument("--seeds",default="1-20")
    ap.add_argument("--output-dir",default="results/relation_dual_generalization_v1411")
    args=ap.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        a,b=args.seeds.split("-",1)
        seeds=list(range(int(a),int(b)+1))
    else:
        seeds=[int(x) for x in args.seeds.split(",") if x.strip()]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("="*118)
    print(" LLM_GPU v1.4.11 Dual Generalization Evaluation")
    print("="*118)
    print("Device           :",device)
    print("Base             :",args.model)
    print("Stage            :",STAGE)
    print("Dataset          : 140 phrases")
    print("Fixed setting    : mlp16_16 + SupCon 0.25")
    print("Relation seeds   :",f"{seeds[0]}..{seeds[-1]}")
    print("Evaluation A     : leave-one-family-out")
    print("Evaluation B     : within-family paraphrase holdout (5-fold, 20% held out)")
    print("Metrics          : micro + macro accuracy")
    print("Base model       : completely frozen")
    print()

    print("Encoding relation phrases...")
    cache={s["text"]:encode_text_stages(model,tok,s["text"]) for s in RELATION_SAMPLES}
    X=torch.stack([cache[s["text"]][STAGE] for s in RELATION_SAMPLES]).to(device)
    y=torch.tensor([s["target_position"] for s in RELATION_SAMPLES],dtype=torch.long,device=device)

    print("Preparing fixed concept decoder reference...")
    p1,p2=concept_cv_features(model,tok,STAGE,args,CONCEPT_SEEDS)

    rows=[]
    famA={f:[] for f in FAMILIES}
    famB={f:[] for f in FAMILIES}
    A_micro=[]; A_macro=[]; A_target=[]
    B_micro=[]; B_macro=[]; B_target=[]

    print()
    print("Per-seed results")
    print("-"*118)

    for seed in seeds:
        am,ama,apred,afam=family_holdout(X,y,args,seed)
        bm,bma,bpred,bfam=within_family_holdout(X,y,args,seed)

        atgt=reconstruct(apred,p1,p2)
        btgt=reconstruct(bpred,p1,p2)

        A_micro.append(am); A_macro.append(ama); A_target.append(atgt)
        B_micro.append(bm); B_macro.append(bma); B_target.append(btgt)

        for fam in FAMILIES:
            famA[fam].append(afam[fam])
            famB[fam].append(bfam[fam])

        rows.append({
            "seed":seed,
            "family_micro":am,
            "family_macro":ama,
            "family_target":atgt,
            "paraphrase_micro":bm,
            "paraphrase_macro":bma,
            "paraphrase_target":btgt,
            **{f"A_{f}":afam[f] for f in FAMILIES},
            **{f"B_{f}":bfam[f] for f in FAMILIES},
        })

        print(
            f"seed={seed:>3d}  "
            f"A_family micro={am:>6.1%} macro={ama:>6.1%} target={atgt:>6.1%}  |  "
            f"B_para micro={bm:>6.1%} macro={bma:>6.1%} target={btgt:>6.1%}"
        )

    def summarize(name,vals):
        m,s=mean_std(vals)
        return name,m,s,min(vals),max(vals)

    summary_rows=[
        summarize("family_micro",A_micro),
        summarize("family_macro",A_macro),
        summarize("family_target",A_target),
        summarize("paraphrase_micro",B_micro),
        summarize("paraphrase_macro",B_macro),
        summarize("paraphrase_target",B_target),
    ]

    family_rows=[]
    for fam in FAMILIES:
        am,as_=mean_std(famA[fam])
        bm,bs=mean_std(famB[fam])
        family_rows.append((fam,am,as_,min(famA[fam]),max(famA[fam]),bm,bs,min(famB[fam]),max(famB[fam])))

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    fields=[
        "seed","family_micro","family_macro","family_target",
        "paraphrase_micro","paraphrase_macro","paraphrase_target",
        *[f"A_{f}" for f in FAMILIES],
        *[f"B_{f}" for f in FAMILIES],
    ]
    with (out/"per_seed.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    with (out/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["metric","mean","std","min","max"])
        w.writerows(summary_rows)

    with (out/"family_comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow([
            "family",
            "family_holdout_mean","family_holdout_std","family_holdout_min","family_holdout_max",
            "paraphrase_holdout_mean","paraphrase_holdout_std","paraphrase_holdout_min","paraphrase_holdout_max",
        ])
        w.writerows(family_rows)

    print()
    print("A. Unknown relation-family transfer")
    print("-"*88)
    for metric,vals in [
        ("micro accuracy",A_micro),
        ("macro accuracy",A_macro),
        ("reconstructed Target",A_target),
    ]:
        m,s=mean_std(vals)
        print(f"{metric:28s}: {m:.1%} ± {s:.1%}  [{min(vals):.1%}, {max(vals):.1%}]")

    print()
    print("B. Within-family unseen-paraphrase transfer")
    print("-"*88)
    for metric,vals in [
        ("micro accuracy",B_micro),
        ("macro accuracy",B_macro),
        ("reconstructed Target",B_target),
    ]:
        m,s=mean_std(vals)
        print(f"{metric:28s}: {m:.1%} ± {s:.1%}  [{min(vals):.1%}, {max(vals):.1%}]")

    print()
    print("Per-family comparison")
    print("-"*88)
    for fam,am,as_,amin,amax,bm,bs,bmin,bmax in family_rows:
        print(
            f"{fam:10s} family={am:.1%}±{as_:.1%}  "
            f"paraphrase={bm:.1%}±{bs:.1%}"
        )

    print()
    print("Reference")
    print("---------")
    print("v1.4.9 overall relation (100 phrases): 65.7% ± 2.2%")
    print("v1.4.10 weighted micro (140 phrases)  : 64.5% ± 2.2%")
    print()
    print("Interpretation")
    print("--------------")
    print("Evaluation A measures transfer to a completely unseen relation family.")
    print("Evaluation B measures transfer to unseen wording when the relation family")
    print("itself is represented in training. The gap between A and B separates")
    print("relation-type generalization from paraphrase generalization.")
    print("Macro accuracy is included because family sizes are unequal.")
    print("Output dir:",out)


if __name__=="__main__":
    main()
