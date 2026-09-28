# train_transformer_semantic_detector_v111.py
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
    "Self-Attentionを中心に系列を処理するモデル構造は何ですか。",
    "Attentionだけを主に使う代表的なニューラルネット構造を説明してください。",
    "系列データを並列に処理しやすいAttention型アーキテクチャは何ですか。",
    "EncoderとDecoderでAttentionを使うモデル構造は何ですか。",
    "RNNを使わず自己注意で文脈を扱う構造を答えてください。",
    "大規模言語モデルでよく使われるAttention中心の構造は何ですか。",
    "トークン間の関係をSelf-Attentionで計算するモデル構造は何ですか。",
    "注意機構を積み重ねて系列を表現するアーキテクチャを説明してください。",
    "並列計算しやすい系列モデルとして知られる構造は何ですか。",
    "Multi-Head Attentionを主要部品に持つモデル構造は何ですか。",
    "自己注意とFeed Forward層を繰り返す構造を答えてください。",
    "位置情報とAttentionを組み合わせる系列モデルは何ですか。",
    "文中の離れた単語同士の関係をAttentionで扱う構造は何ですか。",
    "Encoderブロックを積み重ねるAttention型モデルを説明してください。",
    "Decoderブロックで自己注意を行う系列モデル構造は何ですか。",
    "再帰処理を使わず系列を扱う代表的な構造は何ですか。",
    "Attentionヘッドを複数持つニューラルネット構造を一つ挙げてください。",
    "長距離依存を自己注意で処理するアーキテクチャは何ですか。",
    "系列全体を同時に見ながら表現を更新するモデル構造は何ですか。",
    "Embeddingの後にAttention層を重ねる構造を説明してください。",
    "自然言語処理で標準的なAttentionベース構造は何ですか。",
    "BERTやGPTの基礎になっているモデル構造は何ですか。",
    "AttentionとFFNからなるブロックを重ねた構造を答えてください。",
    "トークン列を自己注意で変換する代表的なアーキテクチャは何ですか。",
    "位置埋め込みとMulti-Head Attentionを用いるモデル構造は何ですか。",
    "文脈情報をAttention重みで集約するニューラルネット構造は何ですか。",
    "系列モデリングで自己注意を主役にした構造を説明してください。",
    "RNNより並列化しやすいAttention型モデル構造は何ですか。",
    "言語モデルの基盤として使われる自己注意型アーキテクチャは何ですか。",
    "Encoder-Decoder型のAttentionモデル構造を一つ挙げてください。",
]

NEGATIVE = [
    # LLM hard negatives
    "大量の文章を学習して自然な文章を生成するモデルは何ですか。",
    "次のトークンを予測して文章を作る大規模モデルは何ですか。",
    "自然言語で対話する生成モデルを説明してください。",
    "大量のテキストから言語能力を学ぶモデルを答えてください。",
    "質問に文章で回答する大規模言語モデルは何ですか。",
    "文章を要約したり生成したりする言語モデルは何ですか。",
    # GPU / CPU / CUDA
    "大量の並列計算を高速に処理する装置は何ですか。",
    "複雑な命令や制御を担当する中央処理装置は何ですか。",
    "NVIDIA GPUで汎用計算を行う技術は何ですか。",
    "行列計算を並列処理するハードウェアを説明してください。",
    "OSの汎用処理を行うプロセッサは何ですか。",
    # Python
    "読みやすい文法で知られる高水準言語は何ですか。",
    "データ分析でよく使われるプログラミング言語を答えてください。",
    "インデントでブロックを表す言語は何ですか。",
    # generic NN/model questions
    "ニューラルネットワークとは何ですか。",
    "畳み込みニューラルネットの特徴を説明してください。",
    "画像分類モデルの仕組みを教えてください。",
    "モデルの精度を比較する方法は何ですか。",
    "学習済みモデルの推論速度を測るにはどうしますか。",
    "AIモデルの重みとは何ですか。",
    # general / conversation
    "一言で答えてください。",
    "別の話題に移りましょう。",
    "同じ内容をもう一度説明してください。",
    "日本の首都はどこですか。",
    "今日はここまでにします。",
]

HARD_NEGATIVE_GUARD = [
    "LLMはどのように文章を生成しますか。",
    "大規模言語モデルとは何ですか。",
    "Pythonは何に使われる言語ですか。",
    "GPUの並列処理を説明してください。",
    "CPUとGPUの違いを説明してください。",
    "ニューラルネットの一般的な構造を説明してください。",
    "画像認識モデルにはどんな種類がありますか。",
    "NVIDIAのGPU計算技術は何ですか。",
]


def choose_threshold(pos_probs, neg_probs):
    candidates=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.92,0.94,0.96]
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
    p.add_argument("--output",default="model/model-gpu-v1.1.1-transformer-detector.pt")
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--lr",type=float,default=7e-4)
    p.add_argument("--seed",type=int,default=44)
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
    print("Transformer semantic detector training")
    print("--------------------------------------")
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
