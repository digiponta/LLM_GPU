# relation_hard_family_expansion_v1410.py
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_relation_hard_expanded_v1410 import RELATION_SAMPLES, FAMILIES, EXPECTED
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


def eval_seed(X,y,args,seed):
    pred = torch.zeros(len(y),dtype=torch.long)
    family_scores = {}

    for fi,fam in enumerate(FAMILIES):
        test = [i for i,s in enumerate(RELATION_SAMPLES) if s["family"]==fam]
        train = [i for i,s in enumerate(RELATION_SAMPLES) if s["family"]!=fam]

        probs = train_predict(X,y,train,test,args,seed+1000*fi)
        pr = probs.argmax(-1)
        pred[test] = pr
        family_scores[fam] = float((pr==y[test].cpu()).float().mean())

    return float((pred==y.cpu()).float().mean()), pred, family_scores


def reconstruct(pred_rel,p1,p2):
    hits = []
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


def mean_std(xs):
    if len(xs)==1:
        return float(xs[0]),0.0
    return statistics.mean(xs),statistics.stdev(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=400)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--temperature",type=float,default=0.1)
    ap.add_argument("--seeds",default="1-20")
    ap.add_argument(
        "--output-dir",
        default="results/relation_hard_family_expansion_v1410",
    )
    args = ap.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        a,b = args.seeds.split("-",1)
        seeds = list(range(int(a),int(b)+1))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    model,_ = LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("="*118)
    print(" LLM_GPU v1.4.10 Hard-Family Expansion")
    print("="*118)
    print("Device           :",device)
    print("Base             :",args.model)
    print("Stage            :",STAGE)
    print("Dataset          : 140 phrases")
    print("Family sizes     : direct=20, select=20, ordinal=40, contrast=40, referent=20")
    print("Fixed setting    : mlp16_16 + SupCon 0.25")
    print("Relation seeds   :",f"{seeds[0]}..{seeds[-1]}")
    print("Evaluation       : leave-one-relation-family-out")
    print("Base model       : completely frozen")
    print()

    print("Encoding 140 relation phrases...")
    cache = {
        s["text"]:encode_text_stages(model,tok,s["text"])
        for s in RELATION_SAMPLES
    }
    X = torch.stack(
        [cache[s["text"]][STAGE] for s in RELATION_SAMPLES]
    ).to(device)
    y = torch.tensor(
        [s["target_position"] for s in RELATION_SAMPLES],
        dtype=torch.long,device=device,
    )

    print("Preparing fixed concept decoder reference...")
    p1,p2 = concept_cv_features(model,tok,STAGE,args,CONCEPT_SEEDS)

    rows = []
    rels = []
    tgts = []
    per_family = {f:[] for f in FAMILIES}

    print()
    print("Per-seed results")
    print("-"*118)

    for seed in seeds:
        rel,pred,fams = eval_seed(X,y,args,seed)
        tgt = reconstruct(pred,p1,p2)

        rels.append(rel)
        tgts.append(tgt)
        for fam in FAMILIES:
            per_family[fam].append(fams[fam])

        rows.append({
            "seed":seed,
            "relation_accuracy":rel,
            "target_accuracy":tgt,
            **{f"family_{fam}":fams[fam] for fam in FAMILIES},
        })

        famtxt = " ".join(
            f"{fam[:3]}={fams[fam]:.0%}" for fam in FAMILIES
        )
        print(
            f"seed={seed:>3d} relation={rel:>6.1%} "
            f"target={tgt:>6.1%} {famtxt}"
        )

    rel_mean,rel_std = mean_std(rels)
    tgt_mean,tgt_std = mean_std(tgts)

    family_rows = []
    for fam in FAMILIES:
        m,s = mean_std(per_family[fam])
        family_rows.append((
            fam,
            len([x for x in RELATION_SAMPLES if x["family"]==fam]),
            m,s,min(per_family[fam]),max(per_family[fam])
        ))

    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    with (out/"per_seed.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields = [
            "seed","relation_accuracy","target_accuracy",
            *[f"family_{x}" for x in FAMILIES]
        ]
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with (out/"family_stability.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["family","n_test","mean","std","min","max"])
        w.writerows(family_rows)

    with (out/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset_size",
            "relation_mean","relation_std","relation_min","relation_max",
            "target_mean","target_std","target_min","target_max",
        ])
        w.writerow([
            len(RELATION_SAMPLES),
            rel_mean,rel_std,min(rels),max(rels),
            tgt_mean,tgt_std,min(tgts),max(tgts),
        ])

    print()
    print("Multi-seed summary")
    print("-"*88)
    print(f"unseen relation mean ± std           : {rel_mean:.1%} ± {rel_std:.1%}")
    print(f"unseen relation min / max            : {min(rels):.1%} / {max(rels):.1%}")
    print(f"reconstructed Target mean ± std      : {tgt_mean:.1%} ± {tgt_std:.1%}")
    print(f"reconstructed Target min / max       : {min(tgts):.1%} / {max(tgts):.1%}")

    print()
    print("Per-family stability")
    print("-"*88)
    for fam,n,m,s,mn,mx in family_rows:
        print(
            f"{fam:10s} n={n:>2d} mean={m:.1%} std={s:.1%} "
            f"min={mn:.1%} max={mx:.1%}"
        )

    print()
    print("Reference")
    print("---------")
    print("v1.4.9 overall relation              : 65.7% ± 2.2%")
    print("v1.4.9 ordinal                       : 57.0% ± 7.1%")
    print("v1.4.9 contrast                      : 54.0% ± 6.2%")
    print("v1.4.9 direct / select / referent    : 72.5% / 71.2% / 74.0%")
    print()
    print("Target")
    print("------")
    print("overall relation mean > 70%")
    print("ordinal > 65%")
    print("contrast > 65%")
    print("overall std < 3%")
    print()
    print("Interpretation")
    print("--------------")
    print("Only the hard ordinal and contrast families were expanded.")
    print("Improvement in those held-out families therefore tests whether targeted")
    print("semantic paraphrase augmentation can reduce family-specific weakness")
    print("without changing the frozen base model or projection architecture.")
    print("Output dir:",out)


if __name__=="__main__":
    main()
