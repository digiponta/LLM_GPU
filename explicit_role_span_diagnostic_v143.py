# explicit_role_span_diagnostic_v143.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from semantic_explicit_role_span_v143 import DATASET, CLASSES, PAIR_KEYS
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
STAGES = ["block1","block2","block3","block4","block5","block6","final_norm"]
FEATURES = [
    "span1",
    "span2",
    "relation",
    "span12",
    "span1_relation",
    "span2_relation",
    "span12_relation",
]

class LinearHead(nn.Module):
    def __init__(self,d_in,n_out):
        super().__init__()
        self.fc = nn.Linear(d_in,n_out)
    def forward(self,x):
        return self.fc(x)

@torch.no_grad()
def encode_text_stages(model,tok,text):
    device = next(model.parameters()).device
    ids = tok.encode(f"人: {text}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    tid = torch.tensor([ids],dtype=torch.long,device=device)

    x = model.embedding(tid)
    if model.position_embedding is not None:
        pos = torch.arange(tid.shape[1],device=device)
        x = x + model.position_embedding(pos).unsqueeze(0)

    out = {}
    for i,block in enumerate(model.blocks,start=1):
        x = block(x)
        out[f"block{i}"] = x[0].mean(0).detach().cpu()

    x = model.final_norm(x)
    out["final_norm"] = x[0].mean(0).detach().cpu()
    return out

def standardize(Xtr,Xte):
    mean = Xtr.mean(0,keepdim=True)
    std = Xtr.std(0,keepdim=True,unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std,(Xte-mean)/std

def train_head(X,y,n_out,args,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    m = LinearHead(X.shape[1],n_out).to(X.device)
    opt = torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        m.train(); opt.zero_grad()
        loss = loss_fn(m(X),y); loss.backward(); opt.step()
    m.eval()
    return m

def predict_split(X,y,n_out,train,test,args,seeds,offset):
    Xtr,Xte = standardize(X[train],X[test])
    probs = []
    for seed in seeds:
        m = train_head(Xtr,y[train],n_out,args,seed+offset)
        with torch.no_grad():
            probs.append(torch.softmax(m(Xte),dim=-1).cpu())
    return torch.stack(probs).mean(0).argmax(-1)

def build_folds():
    folds=[[] for _ in range(5)]
    for c in CLASSES:
        for order in (0,1):
            idx=[i for i,s in enumerate(DATASET) if s["target"]==c and s["surface_order"]==order]
            assert len(idx)==10
            for k in range(5):
                folds[k].extend(idx[2*k:2*k+2])
    return folds

def cv_predict(X,y,n_out,folds,args,seeds,offset):
    pred = torch.zeros(len(y),dtype=torch.long)
    for k,test in enumerate(folds):
        ts=set(test)
        train=[i for i in range(len(y)) if i not in ts]
        pred[test]=predict_split(X,y,n_out,train,test,args,seeds,offset+1000*k)
    return pred

def target_from_pair_role(pair_id,role):
    a,b = PAIR_KEYS[int(pair_id)]
    return CLASS_TO_ID[a if int(role)==0 else b]

def feature_vector(stage,feat,cache1,cache2,cacher):
    a=cache1[stage]; b=cache2[stage]; r=cacher[stage]
    if feat=="span1": return a
    if feat=="span2": return b
    if feat=="relation": return r
    if feat=="span12": return torch.cat([a,b],0)
    if feat=="span1_relation": return torch.cat([a,r],0)
    if feat=="span2_relation": return torch.cat([b,r],0)
    if feat=="span12_relation": return torch.cat([a,b,r],0)
    raise ValueError(feat)

def cross_order(X,y_role,y_pair,y_target,train_order,test_order,args,seeds,offset):
    train=[i for i,s in enumerate(DATASET) if s["surface_order"]==train_order]
    test=[i for i,s in enumerate(DATASET) if s["surface_order"]==test_order]

    pr=predict_split(X,y_role,2,train,test,args,seeds,offset)
    pp=predict_split(X,y_pair,15,train,test,args,seeds,offset+5000)

    gr=y_role[test].cpu(); gp=y_pair[test].cpu(); gt=y_target[test].cpu()
    recon=torch.tensor([target_from_pair_role(p,r) for p,r in zip(pp.tolist(),pr.tolist())])

    return (
        float((pr==gr).float().mean()),
        float((pp==gp).float().mean()),
        float((recon==gt).float().mean()),
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=250)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--seeds",default="41,42,43")
    ap.add_argument("--output-dir",default="results/explicit_role_span_v143")
    args=ap.parse_args()

    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print("="*118)
    print(" LLM_GPU v1.4.3 Explicit Role Marker / Span Diagnostic")
    print("="*118)
    print("Device       :",device)
    print("Base         :",args.model)
    print("Dataset      : 120 structured samples")
    print("Inputs       : concept span1, concept span2, natural-language relation phrase")
    print("Stages       :",", ".join(STAGES))
    print("Features     :",", ".join(FEATURES))
    print("Tasks        : Role 2-way, Pair 15-way, reconstructed Target")
    print("Base model   : completely frozen")
    print()

    # Cache repeated text encodings.
    text_cache={}
    def enc(text):
        if text not in text_cache:
            text_cache[text]=encode_text_stages(model,tok,text)
        return text_cache[text]

    encoded=[]
    print("Encoding explicit spans and relation markers...")
    for i,s in enumerate(DATASET,start=1):
        encoded.append((enc(s["first_desc"]),enc(s["second_desc"]),enc(s["relation"])))
        if i%20==0:
            print(f"  prepared {i}/120")

    y_role=torch.tensor([s["role_direction"] for s in DATASET],dtype=torch.long,device=device)
    y_pair=torch.tensor([s["pair_id"] for s in DATASET],dtype=torch.long,device=device)
    y_target=torch.tensor([CLASS_TO_ID[s["target"]] for s in DATASET],dtype=torch.long)

    folds=build_folds()
    rows=[]

    for stage in STAGES:
        for feat in FEATURES:
            X=torch.stack([
                feature_vector(stage,feat,a,b,r)
                for a,b,r in encoded
            ]).to(device)

            baseoff=10000*STAGES.index(stage)+500*FEATURES.index(feat)
            pr=cv_predict(X,y_role,2,folds,args,seeds,baseoff+10)
            pp=cv_predict(X,y_pair,15,folds,args,seeds,baseoff+20)

            role_cv=float((pr==y_role.cpu()).float().mean())
            pair_cv=float((pp==y_pair.cpu()).float().mean())

            recon=torch.tensor([target_from_pair_role(p,r) for p,r in zip(pp.tolist(),pr.tolist())])
            target_cv=float((recon==y_target).float().mean())

            t2c=cross_order(X,y_role,y_pair,y_target,0,1,args,seeds,baseoff+100)
            c2t=cross_order(X,y_role,y_pair,y_target,1,0,args,seeds,baseoff+200)
            cross_role=(t2c[0]+c2t[0])/2
            cross_pair=(t2c[1]+c2t[1])/2
            cross_target=(t2c[2]+c2t[2])/2

            print(
                f"{stage:10s} {feat:18s} "
                f"roleCV={role_cv:6.1%} pairCV={pair_cv:6.1%} targetCV={target_cv:6.1%} "
                f"crossRole={cross_role:6.1%} crossPair={cross_pair:6.1%} crossTarget={cross_target:6.1%}"
            )

            rows.append([
                stage,feat,X.shape[1],
                role_cv,pair_cv,target_cv,
                t2c[0],c2t[0],cross_role,
                t2c[1],c2t[1],cross_pair,
                t2c[2],c2t[2],cross_target,
            ])

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow([
            "stage","feature","feature_dim",
            "role_cv_accuracy","pair_cv_accuracy","target_cv_accuracy",
            "role_targetfirst_to_contrastfirst","role_contrastfirst_to_targetfirst","role_cross_order_mean",
            "pair_targetfirst_to_contrastfirst","pair_contrastfirst_to_targetfirst","pair_cross_order_mean",
            "target_targetfirst_to_contrastfirst","target_contrastfirst_to_targetfirst","target_cross_order_mean",
        ])
        w.writerows(rows)

    best_role=max(rows,key=lambda r:r[3])
    best_cross_role=max(rows,key=lambda r:r[8])
    best_target=max(rows,key=lambda r:r[5])

    print()
    print("Best mixed-order Role CV")
    print("------------------------")
    print(f"{best_role[0]} / {best_role[1]} : {best_role[3]:.1%}")

    print()
    print("Best cross-position Role")
    print("------------------------")
    print(f"{best_cross_role[0]} / {best_cross_role[1]} : {best_cross_role[8]:.1%}")

    print()
    print("Best reconstructed Target CV")
    print("----------------------------")
    print(f"{best_target[0]} / {best_target[1]} : {best_target[5]:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.2 best mixed Role CV       : 21.7%")
    print("v1.4.2 best cross-position Role : 39.2%")
    print("Binary chance                   : 50.0%")
    print()
    print("Interpretation")
    print("--------------")
    print("If span12_relation strongly exceeds span12 and relation-only baselines, explicit")
    print("Concept + Relation factorization recovers role binding that a single prompt vector lost.")
    print("If relation-only is already perfect, the experiment mainly confirms that the relation")
    print("phrase itself carries the role rule; reconstructed Target still requires concept-pair identity.")
    print("Cross-position performance tests whether the factorization generalizes across word order.")
    print("Output dir:",outdir)

if __name__=="__main__":
    main()
