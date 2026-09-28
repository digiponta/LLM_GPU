# train_llm_semantic_detector_v1006.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from evaluate_post_entity_boundary_binding_v01115 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

POSITIVE = [
    "大量の文章を学習して自然な文を生成するモデルは何ですか。",
    "大規模なテキストから言語の規則性を学ぶモデルを説明してください。",
    "文章生成に使う大規模言語モデルとは何ですか。",
    "自然言語の続きを予測して文章を作るモデルは何ですか。",
    "会話や文章生成を行う言語モデルを一つ挙げてください。",
    "テキストを学習して質問に文章で答えるモデルは何ですか。",
    "大量の文書で学習する言語生成モデルを何と呼びますか。",
    "単語列のパターンを学び文章を生成するモデルを説明してください。",
    "自然言語を入力して文章を出力する大規模モデルは何ですか。",
    "チャットボットの文章生成で使う言語モデルは何ですか。",
    "文脈から次の語を予測する大規模なモデルを説明してください。",
    "大量のテキストデータから言語能力を獲得するモデルは何ですか。",
    "文章の要約や生成に使われる大規模モデルは何ですか。",
    "人間の言葉を扱う生成モデルを一つ答えてください。",
    "言語パターンを学んで新しい文章を作るAIモデルは何ですか。",
    "テキスト生成を目的とする大規模ニューラルモデルを説明してください。",
    "質問文を理解して文章で応答するモデルは何ですか。",
    "大規模コーパスから自然言語を学ぶモデルを何と呼びますか。",
    "文章の意味と文脈を使って応答を生成するモデルは何ですか。",
    "生成AIで文章を作る中心的な言語モデルを説明してください。",
    "大量の言語データで事前学習されるモデルは何ですか。",
    "文を読んで次の文を生成する大規模モデルは何ですか。",
    "自然言語処理で文章生成を担うモデルを答えてください。",
    "会話内容に応じて返答文を生成するモデルは何ですか。",
    "言語を扱う生成系の大規模モデルを説明してください。",
    "テキストから学習してテキストを生成するAIは何ですか。",
    "文章の続きを生成できる大規模な言語AIを何と呼びますか。",
    "言葉の並びを学習する大規模モデルを一つ挙げてください。",
    "自然な文章を生成するための大規模モデルは何ですか。",
    "言語知識を学習し文章で回答するモデルを説明してください。",
]

NEGATIVE = [
    "大量の並列演算を高速に行う装置は何ですか。",
    "画像処理で多数の計算を同時に行うプロセッサは何ですか。",
    "行列計算を並列化するのが得意な演算装置を説明してください。",
    "機械学習の数値計算を高速化するハードウェアは何ですか。",
    "多数の演算器で同時処理する装置は何ですか。",
    "汎用命令を実行しコンピュータを制御する装置は何ですか。",
    "OSや一般処理を担当する中央演算装置を説明してください。",
    "分岐や多様な命令処理に向くプロセッサは何ですか。",
    "コンピュータ全体の制御を担う装置は何ですか。",
    "一般的なプログラムを実行する中央処理装置は何ですか。",
    "Attentionを中心にしたモデル構造は何ですか。",
    "Self-Attentionを使う代表的アーキテクチャを説明してください。",
    "系列処理で注意機構を使うニューラルネット構造は何ですか。",
    "RNNの代わりにAttentionを使う構造を答えてください。",
    "LLMで広く使われるAttention型のモデル構造は何ですか。",
    "NVIDIA GPUで汎用計算を行うための技術は何ですか。",
    "GPU向けのNVIDIA並列計算環境を説明してください。",
    "NVIDIAのGPGPUプログラミング基盤は何ですか。",
    "GPU上で計算カーネルを実行するNVIDIA技術は何ですか。",
    "NVIDIA GPU用の汎用計算プラットフォームを答えてください。",
    "読みやすい文法で知られるプログラミング言語は何ですか。",
    "データ分析でよく使われる汎用言語を一つ挙げてください。",
    "インデントでブロックを表す言語は何ですか。",
    "機械学習でも広く使われる高水準言語は何ですか。",
    "可読性を重視した汎用プログラミング言語を説明してください。",
    "一言で答えてください。",
    "別の話題に移りたいです。",
    "もう一度分かりやすく説明してください。",
    "二つの実験を同じ条件で比較したいです。",
    "今日はここまでにします。",
]


class LLMDetector(nn.Module):
    def __init__(self, d_model: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def encode_hidden(model, tok, text):
    device = next(model.parameters()).device
    ids = tok.encode(f"人: {text}\nAI: ", add_bos=True)
    x = torch.tensor([ids[-model.context_length:]], dtype=torch.long, device=device)
    h = model.forward_hidden(x)[:, -1, :]
    return h[0].detach()


def choose_threshold(probs, labels):
    candidates = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]
    best = None
    for th in candidates:
        pred = probs >= th
        tp = int(((pred == 1) & (labels == 1)).sum().item())
        fp = int(((pred == 1) & (labels == 0)).sum().item())
        fn = int(((pred == 0) & (labels == 1)).sum().item())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        key = (fp == 0, recall, precision, -th)
        if best is None or key > best[0]:
            best = (key, th, tp, fp, fn, precision, recall)
    return best[1:]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--output",default="model/model-gpu-v1.0.6-llm-detector.pt")
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for q in model.parameters(): q.requires_grad_(False)

    samples=[(x,1) for x in POSITIVE]+[(x,0) for x in NEGATIVE]
    random.shuffle(samples)
    n_val=max(12,int(len(samples)*0.2))
    val=samples[:n_val]; train=samples[n_val:]

    Xtr=torch.stack([encode_hidden(model,tok,t) for t,_ in train])
    ytr=torch.tensor([y for _,y in train],dtype=torch.float32,device=device)
    Xv=torch.stack([encode_hidden(model,tok,t) for t,_ in val])
    yv=torch.tensor([y for _,y in val],dtype=torch.float32,device=device)

    det=LLMDetector(Xtr.shape[1],32).to(device)
    opt=torch.optim.Adam(det.parameters(),lr=args.lr,weight_decay=1e-4)
    pos_weight=torch.tensor([(ytr.numel()-ytr.sum()).item()/max(1.0,ytr.sum().item())],device=device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state=None; best_loss=float("inf")
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
        pv=torch.sigmoid(det(Xv))
    th,tp,fp,fn,prec,rec=choose_threshold(pv,yv)

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "state_dict":det.state_dict(),
        "d_model":int(Xtr.shape[1]),
        "hidden":32,
        "threshold":float(th),
        "seed":args.seed,
        "train_size":len(train),
        "val_size":len(val),
        "positive_prompts":POSITIVE,
        "negative_prompts":NEGATIVE,
        "val_best_loss":best_loss,
    },out)
    print()
    print("LLM detector training")
    print("---------------------")
    print("Device:",device)
    print("Train:",len(train),"Validation:",len(val))
    print(f"Best val loss: {best_loss:.6f}")
    print(f"Selected threshold: {th:.2f}")
    print(f"Validation precision: {prec:.1%} TP={tp} FP={fp}")
    print(f"Validation recall   : {rec:.1%} TP={tp} FN={fn}")
    print("Checkpoint:",out)

if __name__=="__main__":
    main()
