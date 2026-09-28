# train_llm_semantic_detector_v1007.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from train_llm_semantic_detector_v1006 import (
    POSITIVE, NEGATIVE, LLMDetector, encode_hidden,
)
from evaluate_post_entity_boundary_binding_v01115 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

# Hard negatives are deliberately distinct from the original 30-case benchmark
# and from the v1.0.1 held-out five LLM prompts.
HARD_NEGATIVE_TRAIN = [
    # capital / geography / factual QA
    "フランスの首都はどこですか。",
    "国の行政機関が集まる中心都市を何と呼びますか。",
    "東京都は日本のどのような都市ですか。",
    "イギリスの首都を一つ答えてください。",
    "政府機関が置かれる中心都市について説明してください。",
    # Python / programming language
    "初心者が学びやすい高水準プログラミング言語を挙げてください。",
    "コードの可読性を重視した汎用言語は何ですか。",
    "データ処理でよく利用されるスクリプト言語を答えてください。",
    "インデントを文法に利用するプログラミング言語を説明してください。",
    # model-shaped non-LLM negatives
    "Attentionを中心に使うモデル構造を説明してください。",
    "自己注意機構を持つニューラルネットの構造は何ですか。",
    "画像認識で使うニューラルネットモデルの例を挙げてください。",
    "分類モデルの精度を比較するには何を揃えますか。",
    "モデルAとモデルBの性能差を公平に評価する方法は何ですか。",
]

HARD_NEGATIVE_GUARD = [
    "ドイツの首都はどこですか。",
    "行政の中心となる都市を一つ答えてください。",
    "読みやすさを重視するプログラミング言語は何ですか。",
    "データ分析向けの高水準言語を一つ挙げてください。",
    "Attentionを使う代表的なモデル構造は何ですか。",
    "二つのモデルを同じ評価指標で比較するにはどうしますか。",
]


def choose_threshold_guard(pos_probs, neg_probs):
    candidates = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.92,0.94,0.96]
    best = None
    for th in candidates:
        tp = int((pos_probs >= th).sum().item())
        fn = int((pos_probs < th).sum().item())
        fp = int((neg_probs >= th).sum().item())
        tn = int((neg_probs < th).sum().item())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        # Safety first: zero hard-negative false positives; then maximize recall.
        key = (fp == 0, recall, precision, tn, -th)
        if best is None or key > best[0]:
            best = (key, th, tp, fp, fn, tn, precision, recall)
    return best[1:]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--output",default="model/model-gpu-v1.0.7-llm-detector-hard-negative.pt")
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--lr",type=float,default=7e-4)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for q in model.parameters(): q.requires_grad_(False)

    pos=list(POSITIVE)
    neg=list(NEGATIVE)+list(HARD_NEGATIVE_TRAIN)
    random.shuffle(pos); random.shuffle(neg)

    # Keep a balanced ordinary validation split, while reserving a dedicated
    # hard-negative guard that is never used for gradient updates.
    val_pos=pos[:6]
    val_neg=neg[:6]
    train=[(x,1) for x in pos[6:]]+[(x,0) for x in neg[6:]]
    random.shuffle(train)

    Xtr=torch.stack([encode_hidden(model,tok,t) for t,_ in train])
    ytr=torch.tensor([y for _,y in train],dtype=torch.float32,device=device)
    Xvp=torch.stack([encode_hidden(model,tok,t) for t in val_pos])
    Xvn=torch.stack([encode_hidden(model,tok,t) for t in val_neg])
    Xhg=torch.stack([encode_hidden(model,tok,t) for t in HARD_NEGATIVE_GUARD])

    det=LLMDetector(Xtr.shape[1],32).to(device)
    opt=torch.optim.Adam(det.parameters(),lr=args.lr,weight_decay=2e-4)
    pos_weight=torch.tensor([(ytr.numel()-ytr.sum()).item()/max(1.0,ytr.sum().item())],device=device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state=None; best_loss=float("inf")
    yv=torch.cat([
        torch.ones(len(val_pos),device=device),
        torch.zeros(len(val_neg),device=device),
    ])
    Xv=torch.cat([Xvp,Xvn],dim=0)

    for ep in range(1,args.epochs+1):
        det.train(); opt.zero_grad()
        loss=loss_fn(det(Xtr),ytr); loss.backward(); opt.step()
        det.eval()
        with torch.no_grad():
            vl=float(loss_fn(det(Xv),yv).item())
        if vl<best_loss:
            best_loss=vl
            best_state={k:v.detach().cpu().clone() for k,v in det.state_dict().items()}
        if ep in (1,50,100,150,200,250,args.epochs):
            print(f"epoch={ep:3d} train={float(loss.item()):.6f} val={vl:.6f}")

    det.load_state_dict(best_state); det.eval()
    with torch.no_grad():
        pos_probs=torch.sigmoid(det(Xvp))
        # Threshold guard combines ordinary validation negatives and unseen
        # hard negatives, so false-positive safety is explicitly optimized.
        neg_probs=torch.cat([torch.sigmoid(det(Xvn)),torch.sigmoid(det(Xhg))],dim=0)

    th,tp,fp,fn,tn,prec,rec=choose_threshold_guard(pos_probs,neg_probs)

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "state_dict":det.state_dict(),
        "d_model":int(Xtr.shape[1]),
        "hidden":32,
        "threshold":float(th),
        "seed":args.seed,
        "train_size":len(train),
        "val_positive_size":len(val_pos),
        "val_negative_size":len(val_neg),
        "hard_negative_guard_size":len(HARD_NEGATIVE_GUARD),
        "hard_negative_train":HARD_NEGATIVE_TRAIN,
        "hard_negative_guard":HARD_NEGATIVE_GUARD,
        "val_best_loss":best_loss,
    },out)

    print()
    print("LLM detector hard-negative training")
    print("-----------------------------------")
    print("Device:",device)
    print("Train:",len(train))
    print("Validation positives:",len(val_pos))
    print("Validation negatives:",len(val_neg))
    print("Hard-negative guard:",len(HARD_NEGATIVE_GUARD))
    print(f"Best val loss: {best_loss:.6f}")
    print(f"Selected threshold: {th:.2f}")
    print(f"Guard precision: {prec:.1%} TP={tp} FP={fp}")
    print(f"Positive recall: {rec:.1%} TP={tp} FN={fn}")
    print("Checkpoint:",out)

if __name__=="__main__":
    main()
