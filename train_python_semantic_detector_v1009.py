# train_python_semantic_detector_v1009.py
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
    "読みやすい文法で知られる高水準プログラミング言語は何ですか。",
    "データ分析でよく使われる汎用プログラミング言語を説明してください。",
    "インデントでブロックを表す代表的な言語は何ですか。",
    "機械学習でも広く使われるスクリプト言語を一つ挙げてください。",
    "可読性を重視して設計されたプログラミング言語は何ですか。",
    "初心者にも学びやすいことで知られる汎用言語は何ですか。",
    "AI開発やデータ処理でよく使う高水準言語は何ですか。",
    "シンプルな文法を持つ人気のプログラミング言語を答えてください。",
    "インタプリタで実行される代表的な汎用言語を説明してください。",
    "科学計算や自動化で広く使われる言語は何ですか。",
    "コードの読みやすさを重視する高水準言語は何ですか。",
    "データサイエンス分野で定番のプログラミング言語を一つ挙げてください。",
    "インデントを構文として利用する代表的な言語を答えてください。",
    "機械学習ライブラリが豊富な汎用プログラミング言語は何ですか。",
    "可読性の高いコードを書きやすい言語を説明してください。",
    "スクリプト用途にも汎用開発にも使われる言語は何ですか。",
    "初心者向け教材でもよく使われるプログラミング言語は何ですか。",
    "データ分析とWeb開発の両方で使われる言語を一つ挙げてください。",
    "簡潔な記述がしやすい高水準言語は何ですか。",
    "AI研究で頻繁に利用される汎用言語を説明してください。",
    "NumPyやPyTorchと組み合わせて使う代表的な言語は何ですか。",
    "インデント主体の見やすい構文を持つ言語は何ですか。",
    "読みやすさで人気のあるプログラミング言語を答えてください。",
    "自動化スクリプトにも使いやすい汎用言語は何ですか。",
    "教育用途でもよく採用される高水準言語は何ですか。",
    "データ処理でコードを短く書きやすい言語を説明してください。",
    "WebからAIまで幅広く使われるプログラミング言語は何ですか。",
    "簡潔で理解しやすい文法を持つ言語を一つ挙げてください。",
    "科学技術計算で広く使われる高水準言語は何ですか。",
    "豊富なライブラリを持つ読みやすい汎用言語を説明してください。",
]

NEGATIVE = [
    # LLM
    "大量の文章を学習して自然な文を生成するモデルは何ですか。",
    "自然言語の続きを予測する大規模モデルを説明してください。",
    "会話文を生成する大規模言語モデルは何ですか。",
    "テキストから言語パターンを学ぶモデルを一つ挙げてください。",
    # CUDA
    "NVIDIA GPUで汎用計算を行う技術は何ですか。",
    "GPU計算用のNVIDIAプラットフォームを説明してください。",
    "NVIDIAのGPGPUプログラミング環境は何ですか。",
    "GPU上で計算カーネルを動かす技術を答えてください。",
    # CPU/GPU
    "汎用命令を実行して制御を行う装置は何ですか。",
    "多数の並列演算を高速に処理する装置は何ですか。",
    "OSや一般処理を担当する中央処理装置を説明してください。",
    "画像処理で大量の計算を同時実行する装置は何ですか。",
    # Transformer
    "Attentionを中心に使うモデル構造は何ですか。",
    "Self-Attentionを利用する代表的アーキテクチャを説明してください。",
    "LLMで使われるAttention型の構造は何ですか。",
    "RNNではなく注意機構を使う系列モデル構造を答えてください。",
    # generic language / code hard negatives
    "C言語の特徴を説明してください。",
    "Javaはどのようなプログラミング言語ですか。",
    "JavaScriptは主にどこで使われますか。",
    "コンパイル言語とインタプリタ言語の違いを説明してください。",
    "関数型プログラミング言語の例を挙げてください。",
    "プログラミング言語の型システムとは何ですか。",
    "高水準言語とは何ですか。",
    "コードの可読性を高める方法を説明してください。",
    # conversation / general QA
    "一言で答えてください。",
    "別の話題に移りたいです。",
    "同じ内容を簡潔に説明してください。",
    "日本の首都はどこですか。",
    "二つのモデルを公平に比較するにはどうしますか。",
    "今日はここまでにします。",
]

HARD_NEGATIVE_GUARD = [
    "Javaの特徴を一文で説明してください。",
    "C++はどのような言語ですか。",
    "高水準言語の意味を説明してください。",
    "プログラムを読みやすくする方法は何ですか。",
    "NVIDIA GPU向けの計算技術は何ですか。",
    "文章を生成する大規模モデルは何ですか。",
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
    p.add_argument("--output",default="model/model-gpu-v1.0.9-python-detector.pt")
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
        "train_size":len(train),
        "val_positive_size":len(val_pos),
        "val_negative_size":len(val_neg),
        "hard_negative_guard_size":len(HARD_NEGATIVE_GUARD),
        "best_val_loss":best_loss,
    },out)

    print()
    print("Python semantic detector training")
    print("---------------------------------")
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
