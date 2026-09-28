# contrast_relation_decomposition_v1412.py
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_contrast_decomposition_v1412 import CONTRAST_SAMPLES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer
from relation_paraphrase_generalization_v145 import encode_text_stages

STAGE = "block5"
HIDDEN = 16
PROJ_DIM = 16
SUPCON_WEIGHT = 0.25
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
    n=z.shape[0]
    sim=(z@z.T)/temp
    eye=torch.eye(n,dtype=torch.bool,device=z.device)
    same=y[:,None].eq(y[None,:]) & ~eye
    logits=sim.masked_fill(eye,float("-inf"))
    logp=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    pos=same.sum(1).clamp_min(1)
    return -(logp.masked_fill(~same,0).sum(1)/pos).mean()


def standardize_fit(X):
    m=X.mean(0,keepdim=True)
    s=X.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
    return m,s


def apply_std(X,m,s):
    return (X-m)/s


def train_model(X,y,args,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    p=Projection(X.shape[1]).to(X.device)
    h=Head().to(X.device)
    opt=torch.optim.AdamW(
        list(p.parameters())+list(h.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for _ in range(args.epochs):
        opt.zero_grad()
        z=p(X)
        loss=F.cross_entropy(h(z),y)+SUPCON_WEIGHT*supcon(z,y,args.temperature)
        loss.backward()
        opt.step()

    p.eval(); h.eval()
    return p,h


@torch.no_grad()
def probs(p,h,X):
    return torch.softmax(h(p(X)),dim=-1)


def build_folds():
    folds=[[] for _ in range(N_FOLDS)]
    for label in (0,1):
        idx=[i for i,s in enumerate(CONTRAST_SAMPLES) if s["selected_position"]==label]
        assert len(idx)==20
        for k in range(N_FOLDS):
            folds[k].extend(idx[k*4:(k+1)*4])
    return folds


def eval_direct(Xfull,yselected,args,seed):
    pred=torch.zeros(len(yselected),dtype=torch.long)
    folds=build_folds()
    for k,test in enumerate(folds):
        ts=set(test)
        train=[i for i in range(len(yselected)) if i not in ts]
        m,s=standardize_fit(Xfull[train])
        Xtr=apply_std(Xfull[train],m,s)
        Xte=apply_std(Xfull[test],m,s)
        p,h=train_model(Xtr,yselected[train],args,seed+1000*k)
        pred[test]=probs(p,h,Xte).argmax(-1).cpu()
    return float((pred==yselected.cpu()).float().mean()),pred


def eval_decomposed(Xneg,Xsel,yneg,ysel,args,seed):
    pred_neg=torch.zeros(len(yneg),dtype=torch.long)
    pred_sel=torch.zeros(len(ysel),dtype=torch.long)
    pred_bound=torch.zeros(len(ysel),dtype=torch.long)
    consistency=[]

    folds=build_folds()
    for k,test in enumerate(folds):
        ts=set(test)
        train=[i for i in range(len(ysel)) if i not in ts]

        mn,sn=standardize_fit(Xneg[train])
        ms,ss=standardize_fit(Xsel[train])
        Xntr=apply_std(Xneg[train],mn,sn)
        Xnte=apply_std(Xneg[test],mn,sn)
        Xstr=apply_std(Xsel[train],ms,ss)
        Xste=apply_std(Xsel[test],ms,ss)

        pn,hn=train_model(Xntr,yneg[train],args,seed+10000+2000*k)
        ps,hs=train_model(Xstr,ysel[train],args,seed+11000+2000*k)

        pneg=probs(pn,hn,Xnte).cpu()
        psel=probs(ps,hs,Xste).cpu()

        n_hat=pneg.argmax(-1)
        s_hat=psel.argmax(-1)
        pred_neg[test]=n_hat
        pred_sel[test]=s_hat

        # Deterministic semantic binder:
        # selected side should be opposite the negated side.
        # Fuse both pieces of evidence in probability space.
        fused = psel.clone()
        fused[:,0] += pneg[:,1]
        fused[:,1] += pneg[:,0]
        bound=fused.argmax(-1)
        pred_bound[test]=bound

        consistency.extend((s_hat == (1-n_hat)).tolist())

    neg_acc=float((pred_neg==yneg.cpu()).float().mean())
    sel_acc=float((pred_sel==ysel.cpu()).float().mean())
    bound_acc=float((pred_bound==ysel.cpu()).float().mean())
    consistency_rate=sum(consistency)/len(consistency)
    return neg_acc,sel_acc,bound_acc,consistency_rate,pred_bound


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
    ap.add_argument("--output-dir",default="results/contrast_relation_decomposition_v1412")
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
    print(" LLM_GPU v1.4.12 Contrast Relation Decomposition")
    print("="*118)
    print("Device           :",device)
    print("Base             :",args.model)
    print("Stage            :",STAGE)
    print("Contrast samples : 40 = 20 FIRST-selected + 20 SECOND-selected")
    print("Direct baseline  : full phrase -> selected position")
    print("Decomposition    : negation span + selection span -> deterministic binder")
    print("Evaluation       : 5-fold unseen-paraphrase holdout")
    print("Relation seeds   :",f"{seeds[0]}..{seeds[-1]}")
    print("Base model       : completely frozen")
    print()

    print("Encoding full phrases and semantic spans...")
    full_cache={}
    neg_cache={}
    sel_cache={}
    for s in CONTRAST_SAMPLES:
        if s["text"] not in full_cache:
            full_cache[s["text"]]=encode_text_stages(model,tok,s["text"])[STAGE]
        if s["negated_text"] not in neg_cache:
            neg_cache[s["negated_text"]]=encode_text_stages(model,tok,s["negated_text"])[STAGE]
        if s["selected_text"] not in sel_cache:
            sel_cache[s["selected_text"]]=encode_text_stages(model,tok,s["selected_text"])[STAGE]

    Xfull=torch.stack([full_cache[s["text"]] for s in CONTRAST_SAMPLES]).to(device)
    Xneg=torch.stack([neg_cache[s["negated_text"]] for s in CONTRAST_SAMPLES]).to(device)
    Xsel=torch.stack([sel_cache[s["selected_text"]] for s in CONTRAST_SAMPLES]).to(device)
    ysel=torch.tensor([s["selected_position"] for s in CONTRAST_SAMPLES],dtype=torch.long,device=device)
    yneg=torch.tensor([s["negated_position"] for s in CONTRAST_SAMPLES],dtype=torch.long,device=device)

    rows=[]
    direct_scores=[]
    neg_scores=[]
    sel_scores=[]
    bound_scores=[]
    consistency_scores=[]

    print()
    print("Per-seed results")
    print("-"*118)

    for seed in seeds:
        direct,_=eval_direct(Xfull,ysel,args,seed)
        neg,sel,bound,consistency,_=eval_decomposed(
            Xneg,Xsel,yneg,ysel,args,seed
        )

        direct_scores.append(direct)
        neg_scores.append(neg)
        sel_scores.append(sel)
        bound_scores.append(bound)
        consistency_scores.append(consistency)

        rows.append({
            "seed":seed,
            "direct_accuracy":direct,
            "negated_position_accuracy":neg,
            "selected_span_accuracy":sel,
            "bound_target_accuracy":bound,
            "span_consistency":consistency,
            "delta_bound_vs_direct":bound-direct,
        })

        print(
            f"seed={seed:>3d}  direct={direct:>6.1%}  "
            f"neg={neg:>6.1%}  select={sel:>6.1%}  "
            f"bound={bound:>6.1%}  consistency={consistency:>6.1%}  "
            f"delta={bound-direct:+.1%}"
        )

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    fields=[
        "seed","direct_accuracy","negated_position_accuracy",
        "selected_span_accuracy","bound_target_accuracy",
        "span_consistency","delta_bound_vs_direct",
    ]
    with (out/"per_seed.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    metrics=[
        ("direct_accuracy",direct_scores),
        ("negated_position_accuracy",neg_scores),
        ("selected_span_accuracy",sel_scores),
        ("bound_target_accuracy",bound_scores),
        ("span_consistency",consistency_scores),
        ("delta_bound_vs_direct",[b-d for b,d in zip(bound_scores,direct_scores)]),
    ]
    with (out/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["metric","mean","std","min","max"])
        for name,vals in metrics:
            m,s=mean_std(vals)
            w.writerow([name,m,s,min(vals),max(vals)])

    print()
    print("Multi-seed summary")
    print("-"*88)
    for label,vals in [
        ("Direct full-phrase baseline",direct_scores),
        ("Negated-position span",neg_scores),
        ("Selected-position span",sel_scores),
        ("Decomposed + binder",bound_scores),
        ("Span consistency",consistency_scores),
    ]:
        m,s=mean_std(vals)
        print(f"{label:30s}: {m:.1%} ± {s:.1%}  [{min(vals):.1%}, {max(vals):.1%}]")

    delta=[b-d for b,d in zip(bound_scores,direct_scores)]
    dm,ds=mean_std(delta)
    print(f"{'Binder gain vs direct':30s}: {dm:+.1%} ± {ds:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.11 contrast family holdout      : 58.4% ± 3.3%")
    print("v1.4.11 contrast paraphrase holdout  : 58.9% ± 4.6%")
    print()
    print("Interpretation")
    print("--------------")
    print("The direct baseline asks one representation to solve the entire contrast")
    print("relation at once. The decomposed route separately decodes the negated and")
    print("selected positions from explicit spans, then combines them with a fixed")
    print("semantic rule: selected = opposite(negated).")
    print("A consistent gain would indicate that contrast difficulty comes from")
    print("composition/binding rather than absence of positional semantics.")
    print("Output dir:",out)


if __name__=="__main__":
    main()
