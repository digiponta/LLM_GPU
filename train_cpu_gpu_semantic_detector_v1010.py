# train_cpu_gpu_semantic_detector_v1010.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from train_llm_semantic_detector_v1006 import encode_hidden
from evaluate_post_entity_boundary_binding_v01115 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CPU_PROMPTS = [
    "複雑な命令を順番に処理するのが得意な演算装置は何ですか。",
    "OSの制御や一般的なプログラム実行を担当する装置は何ですか。",
    "分岐の多い処理に向く中央演算装置を説明してください。",
    "多種類の命令を柔軟に実行するプロセッサは何ですか。",
    "コンピュータ全体の制御を担う演算装置を一つ挙げてください。",
    "逐次的な命令実行を得意とする汎用プロセッサは何ですか。",
    "アプリケーションやOSの一般処理を担当する装置は何ですか。",
    "中央処理装置として制御と演算を行うものは何ですか。",
    "複雑な条件分岐を処理しやすいプロセッサを説明してください。",
    "幅広い種類の命令を扱う汎用演算装置は何ですか。",
    "メモリや入出力装置を制御しながら処理する装置は何ですか。",
    "一つ一つ異なる命令を処理するのに適した演算装置は何ですか。",
    "PCで一般的なソフトウェア実行を担うプロセッサは何ですか。",
    "命令解釈と制御を中心に行う中央の演算装置は何ですか。",
    "汎用計算とシステム制御を担当するプロセッサを答えてください。",
    "順次実行や分岐処理に強いプロセッサは何ですか。",
    "コンピュータの主要な制御役となる演算装置を説明してください。",
    "さまざまな命令セットを使う処理に向く装置は何ですか。",
    "一般用途のプログラムを実行する中心的プロセッサは何ですか。",
    "複雑な制御フローを扱うのが得意な演算装置は何ですか。",
    "OSカーネルの制御処理を主に実行する装置は何ですか。",
    "デスクトップPCの汎用処理を担当するプロセッサは何ですか。",
    "分岐予測を使いながら命令を実行するプロセッサは何ですか。",
    "制御中心の処理に適した汎用的な演算装置を一つ挙げてください。",
]

GPU_PROMPTS = [
    "多数の同じ演算を同時に実行するのが得意な装置は何ですか。",
    "大量の行列計算を並列に処理するプロセッサは何ですか。",
    "画像処理で多数の画素を同時に計算する装置を説明してください。",
    "ニューラルネットの大量演算を高速化する演算装置は何ですか。",
    "同型の数値計算をまとめて処理するのに向くプロセッサは何ですか。",
    "多数の演算器で並列処理する装置を一つ挙げてください。",
    "画像生成AIの行列演算を高速化するハードウェアは何ですか。",
    "ベクトルや行列の大量計算を得意とする装置は何ですか。",
    "ディープラーニングで並列演算に使われるプロセッサは何ですか。",
    "大量のデータ要素に同じ処理を適用する装置を説明してください。",
    "グラフィックス処理向けに多数のコアを持つ演算装置は何ですか。",
    "機械学習のテンソル計算を高速に処理する装置は何ですか。",
    "並列性の高い科学計算に向くプロセッサを答えてください。",
    "多数の積和演算を同時に行うのに適した装置は何ですか。",
    "映像レンダリングで大量の計算を並列処理する装置は何ですか。",
    "同一種類の演算を大量に実行するのが得意なプロセッサは何ですか。",
    "行列積を高速化するためによく使われるハードウェアは何ですか。",
    "深層学習の訓練で大量の並列計算を担当する装置は何ですか。",
    "グラフィックスと汎用並列計算に使われるプロセッサは何ですか。",
    "多数のスレッドを並行実行するのに向いた演算装置を説明してください。",
    "画像の各画素を同時に処理するのが得意な装置は何ですか。",
    "大量の単純な数値演算を並列化するプロセッサは何ですか。",
    "AIモデルの学習で高速な行列演算を担う装置は何ですか。",
    "高い並列度を活かして計算を高速化するプロセッサは何ですか。",
]

OTHER_PROMPTS = [
    # CUDA
    "NVIDIA GPUで汎用計算を行うための技術は何ですか。",
    "GPU向けのNVIDIA並列計算環境を説明してください。",
    "NVIDIAのGPGPUプログラミング基盤は何ですか。",
    "GPU上でカーネルを実行する技術を答えてください。",
    # Python
    "読みやすい文法で知られるプログラミング言語は何ですか。",
    "データ分析で広く使われる高水準言語を答えてください。",
    "インデントでブロックを表す言語は何ですか。",
    "機械学習でよく使われる汎用プログラミング言語は何ですか。",
    # LLM
    "大量の文章を学習して文章を生成するモデルは何ですか。",
    "自然言語の続きを予測する大規模モデルは何ですか。",
    "会話文を生成する大規模言語モデルを説明してください。",
    "テキストを学習する生成モデルを一つ挙げてください。",
    # Transformer
    "Attentionを中心に使うモデル構造は何ですか。",
    "Self-Attentionを使う代表的アーキテクチャは何ですか。",
    "系列処理で注意機構を使うモデル構造を説明してください。",
    "RNNではなくAttention主体の構造を答えてください。",
    # General / conversation
    "日本の首都はどこですか。",
    "一言で答えてください。",
    "別の話題に移りましょう。",
    "同じ内容をもう一度説明してください。",
    "二つの実験結果を比較する方法は何ですか。",
    "今日はここまでにします。",
    "エラーメッセージの原因を調べる方法を教えてください。",
    "天気について話してください。",
]

LABELS = ["cpu", "gpu", "other"]


class HardwareDetector(nn.Module):
    def __init__(self, d_model: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)


def split_class(items, n_val=5):
    items=list(items)
    random.shuffle(items)
    return items[n_val:], items[:n_val]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--output",default="model/model-gpu-v1.0.10-cpu-gpu-detector.pt")
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

    cpu_tr,cpu_v=split_class(CPU_PROMPTS)
    gpu_tr,gpu_v=split_class(GPU_PROMPTS)
    oth_tr,oth_v=split_class(OTHER_PROMPTS)

    train=[(x,0) for x in cpu_tr]+[(x,1) for x in gpu_tr]+[(x,2) for x in oth_tr]
    val=[(x,0) for x in cpu_v]+[(x,1) for x in gpu_v]+[(x,2) for x in oth_v]
    random.shuffle(train); random.shuffle(val)

    Xtr=torch.stack([encode_hidden(model,tok,t) for t,_ in train])
    ytr=torch.tensor([y for _,y in train],dtype=torch.long,device=device)
    Xv=torch.stack([encode_hidden(model,tok,t) for t,_ in val])
    yv=torch.tensor([y for _,y in val],dtype=torch.long,device=device)

    det=HardwareDetector(Xtr.shape[1],48).to(device)
    opt=torch.optim.Adam(det.parameters(),lr=args.lr,weight_decay=2e-4)
    loss_fn=nn.CrossEntropyLoss()

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
        probs=torch.softmax(det(Xv),dim=-1)
        pred=torch.argmax(probs,dim=-1)
        conf=torch.max(probs,dim=-1).values

    # Select a conservative confidence threshold: zero wrong CPU/GPU overrides
    # on validation if possible, then maximize correct CPU/GPU activations.
    candidates=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]
    best=None
    for th in candidates:
        active=(pred!=2)&(conf>=th)
        wrong=int(((pred!=yv)&active).sum().item())
        helpful=int(((pred==yv)&active&(yv!=2)).sum().item())
        other_fp=int((active&(yv==2)).sum().item())
        key=(wrong==0 and other_fp==0, helpful, -wrong, -other_fp, th)
        if best is None or key>best[0]:
            best=(key,th,wrong,other_fp,helpful)
    _,threshold,wrong,other_fp,helpful=best

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "state_dict":det.state_dict(),
        "d_model":int(Xtr.shape[1]),
        "hidden":48,
        "labels":LABELS,
        "threshold":float(threshold),
        "seed":args.seed,
        "train_size":len(train),
        "val_size":len(val),
        "best_val_loss":best_loss,
    },out)

    print()
    print("CPU/GPU semantic detector training")
    print("----------------------------------")
    print("Device:",device)
    print("Train:",len(train),"Validation:",len(val))
    print(f"Best val loss: {best_loss:.6f}")
    print(f"Selected threshold: {threshold:.2f}")
    print(f"Validation helpful CPU/GPU: {helpful}")
    print(f"Validation wrong active   : {wrong}")
    print(f"Validation OTHER false pos: {other_fp}")
    print("Checkpoint:",out)

if __name__=="__main__":
    main()
