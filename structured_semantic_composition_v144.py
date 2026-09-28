# structured_semantic_composition_v144.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from semantic_explicit_role_span_v143 import DATASET, CLASSES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
STAGES = ["block1","block2","block3","block4","block5","block6","final_norm"]

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
    probs=[]
    for seed in seeds:
        m=train_head(Xtr,y[train],n_out,args,seed+offset)
        with torch.no_grad():
            probs.append(torch.softmax(m(Xte),dim=-1).cpu())
    return torch.stack(probs).mean(0).argmax(-1)

def build_folds():
    folds=[[] for _ in range(5)]
    for c in CLASSES:
        for order in (0,1):
            idx=[i for i,s in enumerate(DATASET)
                 if s["target"]==c and s["surface_order"]==order]
            assert len(idx)==10
            for k in range(5):
                folds[k].extend(idx[2*k:2*k+2])
    return folds

def cv_predict(X,y,n_out,folds,args,seeds,offset):
    pred=torch.zeros(len(y),dtype=torch.long)
    for k,test in enumerate(folds):
        ts=set(test)
        train=[i for i in range(len(y)) if i not in ts]
        pred[test]=predict_split(X,y,n_out,train,test,args,seeds,offset+1000*k)
    return pred

def reconstruct_target(pred_c1,pred_c2,pred_pos):
    out=[]
    for c1,c2,pos in zip(pred_c1.tolist(),pred_c2.tolist(),pred_pos.tolist()):
        out.append(c1 if pos==0 else c2)
    return torch.tensor(out,dtype=torch.long)

def eval_cross_order(X1,X2,XR,y1,y2,ypos,ytarget,train_order,test_order,args,seeds,offset):
    train=[i for i,s in enumerate(DATASET) if s["surface_order"]==train_order]
    test=[i for i,s in enumerate(DATASET) if s["surface_order"]==test_order]

    p1=predict_split(X1,y1,6,train,test,args,seeds,offset)
    p2=predict_split(X2,y2,6,train,test,args,seeds,offset+5000)
    pp=predict_split(XR,ypos,2,train,test,args,seeds,offset+10000)

    g1=y1[test].cpu(); g2=y2[test].cpu(); gp=ypos[test].cpu(); gt=ytarget[test].cpu()
    rt=reconstruct_target(p1,p2,pp)

    return {
        "c1": float((p1==g1).float().mean()),
        "c2": float((p2==g2).float().mean()),
        "pos": float((pp==gp).float().mean()),
        "target": float((rt==gt).float().mean()),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=250)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--seeds",default="41,42,43")
    ap.add_argument("--output-dir",default="results/structured_semantic_composition_v144")
    args=ap.parse_args()

    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print("="*118)
    print(" LLM_GPU v1.4.4 Structured Semantic Composition")
    print("="*118)
    print("Device          :",device)
    print("Base            :",args.model)
    print("Dataset         : 120 structured samples")
    print("Concept1 head   : 6-way from span1")
    print("Concept2 head   : 6-way from span2")
    print("Target-position : 2-way from relation phrase")
    print("Binding         : deterministic first/second selector")
    print("Stages          :",", ".join(STAGES))
    print("Base model      : completely frozen")
    print()

    cache={}
    def enc(text):
        if text not in cache:
            cache[text]=encode_text_stages(model,tok,text)
        return cache[text]

    encoded=[]
    print("Encoding concept spans and relation phrases...")
    for i,s in enumerate(DATASET,start=1):
        encoded.append((enc(s["first_desc"]),enc(s["second_desc"]),enc(s["relation"])))
        if i%20==0:
            print(f"  prepared {i}/120")

    y1=torch.tensor([CLASS_TO_ID[s["first_class"]] for s in DATASET],dtype=torch.long,device=device)
    y2=torch.tensor([CLASS_TO_ID[s["second_class"]] for s in DATASET],dtype=torch.long,device=device)
    ypos=torch.tensor([0 if s["target"]==s["first_class"] else 1 for s in DATASET],dtype=torch.long,device=device)
    ytarget=torch.tensor([CLASS_TO_ID[s["target"]] for s in DATASET],dtype=torch.long)

    folds=build_folds()
    rows=[]

    for stage in STAGES:
        X1=torch.stack([a[stage] for a,_,_ in encoded]).to(device)
        X2=torch.stack([b[stage] for _,b,_ in encoded]).to(device)
        XR=torch.stack([r[stage] for _,_,r in encoded]).to(device)

        baseoff=10000*STAGES.index(stage)
        p1=cv_predict(X1,y1,6,folds,args,seeds,baseoff+10)
        p2=cv_predict(X2,y2,6,folds,args,seeds,baseoff+20)
        pp=cv_predict(XR,ypos,2,folds,args,seeds,baseoff+30)

        c1_acc=float((p1==y1.cpu()).float().mean())
        c2_acc=float((p2==y2.cpu()).float().mean())
        pos_acc=float((pp==ypos.cpu()).float().mean())

        rt=reconstruct_target(p1,p2,pp)
        target_acc=float((rt==ytarget).float().mean())

        oracle_concepts=reconstruct_target(y1.cpu(),y2.cpu(),pp)
        oracle_pos=reconstruct_target(p1,p2,ypos.cpu())
        oracle_concepts_acc=float((oracle_concepts==ytarget).float().mean())
        oracle_pos_acc=float((oracle_pos==ytarget).float().mean())

        t2c=eval_cross_order(X1,X2,XR,y1,y2,ypos,ytarget,0,1,args,seeds,baseoff+100)
        c2t=eval_cross_order(X1,X2,XR,y1,y2,ypos,ytarget,1,0,args,seeds,baseoff+200)

        cross_c1=(t2c["c1"]+c2t["c1"])/2
        cross_c2=(t2c["c2"]+c2t["c2"])/2
        cross_pos=(t2c["pos"]+c2t["pos"])/2
        cross_target=(t2c["target"]+c2t["target"])/2

        print()
        print(stage.upper())
        print("-"*88)
        print(f"concept1 accuracy                    : {c1_acc:.1%}")
        print(f"concept2 accuracy                    : {c2_acc:.1%}")
        print(f"target-position accuracy             : {pos_acc:.1%}")
        print(f"reconstructed target                 : {target_acc:.1%}")
        print(f"gold concepts + predicted position   : {oracle_concepts_acc:.1%}")
        print(f"predicted concepts + gold position   : {oracle_pos_acc:.1%}")
        print(f"cross-position concept1              : {cross_c1:.1%}")
        print(f"cross-position concept2              : {cross_c2:.1%}")
        print(f"cross-position target-position       : {cross_pos:.1%}")
        print(f"cross-position reconstructed target  : {cross_target:.1%}")

        rows.append([
            stage,c1_acc,c2_acc,pos_acc,target_acc,
            oracle_concepts_acc,oracle_pos_acc,
            t2c["c1"],c2t["c1"],cross_c1,
            t2c["c2"],c2t["c2"],cross_c2,
            t2c["pos"],c2t["pos"],cross_pos,
            t2c["target"],c2t["target"],cross_target,
        ])

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow([
            "stage",
            "concept1_accuracy","concept2_accuracy","target_position_accuracy","target_accuracy",
            "gold_concepts_pred_position_target_accuracy",
            "pred_concepts_gold_position_target_accuracy",
            "concept1_t2c","concept1_c2t","concept1_cross_mean",
            "concept2_t2c","concept2_c2t","concept2_cross_mean",
            "position_t2c","position_c2t","position_cross_mean",
            "target_t2c","target_c2t","target_cross_mean",
        ])
        w.writerows(rows)

    best_target=max(rows,key=lambda r:r[4])
    best_cross=max(rows,key=lambda r:r[18])

    print()
    print("Best reconstructed Target")
    print("-------------------------")
    print(f"{best_target[0]} : {best_target[4]:.1%}")

    print()
    print("Best cross-position Target")
    print("--------------------------")
    print(f"{best_cross[0]} : {best_cross[18]:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.3 best reconstructed Target CV : 16.7%")
    print("6-class chance                      : 16.7%")
    print()
    print("Interpretation")
    print("--------------")
    print("If structured composition sharply improves Target accuracy, the missing operation was")
    print("explicit role binding rather than concept recognition.")
    print("Gold-concepts and gold-position oracle columns show whether concept classification")
    print("or target-position prediction is the remaining bottleneck.")
    print("Cross-position results test whether the deterministic binding survives word-order reversal.")
    print("Output dir:",outdir)

if __name__=="__main__":
    main()
