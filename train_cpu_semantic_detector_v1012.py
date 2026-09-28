# train_cpu_semantic_detector_v1012.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from train_llm_semantic_detector_v1006 import LLMDetector as BinaryDetector, encode_hidden
from evaluate_post_entity_boundary_binding_v01115 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

POSITIVE = [
    "多種類の命令を順番に処理するのが得意なプロセッサは何ですか。",
    "一般用途のプログラム実行を担当する中央処理装置は何ですか。",
    "分岐や複雑な命令処理に向く演算装置を説明してください。",
    "OSの制御や汎用処理を主に担う装置は何ですか。",
    "コンピュータ全体の制御役となるプロセッサを答えてください。",
    "逐次実行や条件分岐に強い汎用プロセッサは何ですか。",
    "中央処理装置として命令実行と制御を行うものは何ですか。",
    "幅広い命令セットを扱う演算装置を説明してください。",
    "一般的なソフトウェアを実行する中心的なプロセッサは何ですか。",
    "複雑な制御フローを処理しやすい演算装置は何ですか。",
    "メモリや入出力を制御しながら命令を処理する装置は何ですか。",
    "PCで汎用的な計算と制御を担当するプロセッサは何ですか。",
    "分岐予測を使って命令を実行する中央演算装置は何ですか。",
    "OSカーネルの処理を主に実行するプロセッサを説明してください。",
    "命令解釈と制御を中心に行う汎用演算装置は何ですか。",
    "さまざまな種類の命令を順次実行する装置は何ですか。",
    "コンピュータの主要な制御機能を担うプロセッサは何ですか。",
    "一般用途の処理を柔軟に実行する中央演算装置は何ですか。",
    "条件分岐を含む複雑な処理に向くプロセッサは何ですか。",
    "一つずつ異なる命令を実行するのが得意な装置を説明してください。",
    "システム全体の制御と汎用計算を担う演算装置は何ですか。",
    "命令列を順番に解釈して実行するプロセッサは何ですか。",
    "制御主体の処理に適した中央処理装置を一つ挙げてください。",
    "一般的なアプリケーション実行を担当するプロセッサは何ですか。",
    "複雑な命令を柔軟に処理する汎用演算装置を答えてください。",
    "多様な処理を切り替えながら実行するプロセッサは何ですか。",
    "分岐と制御を多く含む処理を得意とする装置は何ですか。",
    "システム制御と命令実行の中心となる演算装置は何ですか。",
    "汎用性の高い中央処理装置を説明してください。",
    "逐次的な制御処理を担うプロセッサを一つ挙げてください。",
]

NEGATIVE = [
    # GPU
    "大量の並列演算を高速に処理する装置は何ですか。",
    "行列計算をまとめて実行するのが得意なプロセッサは何ですか。",
    "画像処理で多数の画素を同時に計算する装置を説明してください。",
    "深層学習の大量演算を並列化するハードウェアは何ですか。",
    "多数のスレッドを並行実行するプロセッサは何ですか。",
    "同じ種類の演算を大量に処理する装置は何ですか。",
    # CUDA
    "NVIDIA GPUで汎用計算を行う技術は何ですか。",
    "GPU計算用のNVIDIAプラットフォームを説明してください。",
    "GPGPUプログラミングで使うNVIDIA技術は何ですか。",
    # LLM
    "大量の文章を学習して生成するモデルは何ですか。",
    "自然言語の続きを予測する大規模モデルは何ですか。",
    "会話文を作る大規模言語モデルを説明してください。",
    # Transformer
    "Attentionを中心に使うモデル構造は何ですか。",
    "Self-Attentionを使う代表的アーキテクチャを答えてください。",
    # Python
    "読みやすい文法で知られるプログラミング言語は何ですか。",
    "データ分析でよく使われる高水準言語は何ですか。",
    "インデントでブロックを表す言語を説明してください。",
    # generic processor / hardware hard negatives
    "プロセッサとは何ですか。",
    "演算装置の役割を説明してください。",
    "コンピュータのハードウェア構成を説明してください。",
    "高速計算に使う装置には何がありますか。",
    "CPUとGPUの違いを説明してください。",
    "プロセッサの性能を比較する方法は何ですか。",
    # general
    "日本の首都はどこですか。",
    "一言で答えてください。",
    "別の話題に移りましょう。",
    "同じ内容をもう一度説明してください。",
    "今日はここまでにします。",
]

HARD_NEGATIVE_GUARD = [
    "GPUはどのような計算を得意としますか。",
    "CPUとGPUを比較してください。",
    "プロセッサという言葉の意味を説明してください。",
    "NVIDIAの並列計算環境は何ですか。",
    "Attentionを使うモデル構造は何ですか。",
    "高水準プログラミング言語とは何ですか。",
]


def choose_threshold(pos_probs, neg_probs):
    candidates=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.92,0.94]
    best=None
    for th in candidates:
        tp=int((pos_probs>=th).sum().item())
        fn=int((pos_probs<th).sum().item())
        fp=int((neg_probs>=th).sum().item())
        precision=tp/(tp+fp) if tp+fp else 1.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        key=(fp==0, recall, precision, -th)
        if best is None or key>best[0]:
            best=(key,th,tp,fp,fn,precision,recall)
    return best[1:]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--output",default="model/model-gpu-v1.0.12-cpu-detector.pt")
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--lr",type=float,default=8e-4)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for q in model.parameters(): q.requires_grad_(False)

    pos=list(POSITIVE); neg=list(NEGATIVE)
    random.shuffle(pos); random.shuffle(neg)
    val_pos=pos[:6]; val_neg=neg[:6]
    train=[(x,1) for x in pos[6:]]+[(x,0) for x in neg[6:]]
    random.shuffle(train)

    Xtr=torch.stack([encode_hidden(model,tok,t) for t,_ in train])
    ytr=torch.tensor([y for _,y in train],dtype=torch.float32,device=device)
    Xvp=torch.stack([encode_hidden(model,tok,t) for t in val_pos])
    Xvn=torch.stack([encode_hidden(model,tok,t) for t in val_neg])
    Xhg=torch.stack([encode_hidden(model,tok,t) for t in HARD_NEGATIVE_GUARD])

    det=BinaryDetector(Xtr.shape[1],32).to(device)
    opt=torch.optim.Adam(det.parameters(),lr=args.lr,weight_decay=2e-4)
    pos_weight=torch.tensor([(ytr.numel()-ytr.sum()).item()/max(1.0,ytr.sum().item())],device=device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xv=torch.cat([Xvp,Xvn],dim=0)
    yv=torch.cat([torch.ones(len(val_pos),device=device),torch.zeros(len(val_neg),device=device)])
    best_state=None; best_loss=float("inf")
    for ep in range(1,args.epochs+1):
        det.train(); opt.zero_grad()
        loss=loss_fn(det(Xtr),ytr); loss.backward(); opt.step()
        det.eval()
        with torch.no_grad(): vl=float(loss_fn(det(Xv),yv).item())
        if vl<best_loss:
            best_loss=vl
            best_state={k:v.detach().cpu().clone() for k,v in det.state_dict().items()}
        if ep in (1,50,100,150,200,250,args.epochs):
            print(f"epoch={ep:3d} train={float(loss.item()):.6f} val={vl:.6f}")

    det.load_state_dict(best_state); det.eval()
    with torch.no_grad():
        pos_probs=torch.sigmoid(det(Xvp))
        neg_probs=torch.cat([torch.sigmoid(det(Xvn)),torch.sigmoid(det(Xhg))],dim=0)
    th,tp,fp,fn,prec,rec=choose_threshold(pos_probs,neg_probs)

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "state_dict":det.state_dict(),
        "d_model":int(Xtr.shape[1]),
        "hidden":32,
        "threshold":float(th),
        "seed":args.seed,
        "best_val_loss":best_loss,
        "train_size":len(train),
        "val_positive_size":len(val_pos),
        "val_negative_size":len(val_neg),
        "hard_negative_guard_size":len(HARD_NEGATIVE_GUARD),
    },out)

    print()
    print("CPU semantic detector training")
    print("------------------------------")
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
