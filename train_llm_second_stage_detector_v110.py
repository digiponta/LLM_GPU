# train_llm_second_stage_detector_v110.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from train_llm_semantic_detector_v1006 import LLMDetector, encode_hidden
from evaluate_post_entity_boundary_binding_v01115 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

# Positive paraphrases emphasize indirect descriptions of LLMs.
# The v1.0.1 held-out LLM prompts are deliberately not copied here.
POSITIVE = [
    "文章の文脈から次に続く語を予測して文を作るモデルは何ですか。",
    "大量のテキストから言葉の使われ方を学ぶ生成モデルを説明してください。",
    "自然言語のパターンを学習して文章を出力するモデルは何ですか。",
    "質問に対して文章で回答を生成する言語系のモデルは何ですか。",
    "テキストデータを使って次のトークンを予測するモデルを答えてください。",
    "文章の続きを確率的に生成するニューラルネットモデルは何ですか。",
    "人間の言葉を扱い、文を生成する大規模なモデルは何ですか。",
    "大量の文書から言語表現を学ぶ生成AIの中核モデルは何ですか。",
    "自然言語で対話できる文章生成モデルを一つ挙げてください。",
    "単語列の関係を学び文章を生成するモデルを説明してください。",
    "文章理解と文章生成の両方に使われる大規模モデルは何ですか。",
    "テキストを学習データとして使う生成モデルを答えてください。",
    "文脈に応じて自然な応答文を生成するモデルは何ですか。",
    "言語データから統計的な規則を学んだ大規模モデルを説明してください。",
    "入力された文章に続く内容を生成するモデルは何ですか。",
    "大量の文字列から言語能力を獲得するモデルを一つ挙げてください。",
    "会話や要約や文章生成に利用される言語モデルは何ですか。",
    "文脈を利用して次の語を推定する生成モデルを説明してください。",
    "自然言語を入力し自然言語を出力する大規模モデルは何ですか。",
    "文章コーパスを学習してテキストを生成するモデルは何ですか。",
    "質問文を受け取り文章の回答を作るモデルを答えてください。",
    "言葉の並び方を学習し新しい文章を生成するモデルは何ですか。",
    "テキスト生成を主目的とする大規模ニューラルモデルを説明してください。",
    "文脈から適切な言葉を選び続ける生成モデルは何ですか。",
    "自然言語処理で文章生成を行う大規模モデルを一つ挙げてください。",
    "大量の文章をもとに応答を作るモデルは何ですか。",
    "文章を入力すると文章で応答する生成AIモデルを説明してください。",
    "言語の確率分布を学んでテキストを生成するモデルは何ですか。",
    "チャット形式の応答生成に使われる大規模モデルは何ですか。",
    "自然言語の知識を大量のテキストから獲得するモデルを答えてください。",
]

NEGATIVE = [
    # Transformer / architecture hard negatives
    "Attentionを中心に使うモデル構造は何ですか。",
    "Self-Attentionを使うニューラルネットの構造を説明してください。",
    "系列処理で注意機構を使うアーキテクチャは何ですか。",
    "RNNの代わりにAttentionを使うモデル構造を答えてください。",
    "ニューラルネットワークのモデル構造とは何ですか。",
    # Python / programming language
    "読みやすい文法で知られる高水準言語は何ですか。",
    "データ分析でよく使われるプログラミング言語を答えてください。",
    "機械学習ライブラリが豊富な汎用言語は何ですか。",
    "インデントでブロックを表す言語を説明してください。",
    # CPU / GPU / CUDA
    "複雑な命令と制御を担当する中央処理装置は何ですか。",
    "大量の並列演算を高速に処理する装置は何ですか。",
    "GPUで多数の行列演算を同時に行う装置を説明してください。",
    "NVIDIA GPUで汎用計算を行う技術は何ですか。",
    "GPU向けのNVIDIA並列計算基盤を答えてください。",
    # model-shaped non-LLM negatives
    "画像分類モデルはどのように学習しますか。",
    "二つのAIモデルを公平に比較する方法は何ですか。",
    "モデルの精度を評価する指標を説明してください。",
    "ニューラルネットの重みとは何ですか。",
    "学習済みモデルをGPUで実行する方法は何ですか。",
    "小さな分類モデルを作るには何が必要ですか。",
    # conversational / general
    "一言で答えてください。",
    "別の話題に移りましょう。",
    "同じ説明をもう一度してください。",
    "今日はここまでにします。",
    "日本の首都はどこですか。",
]

HARD_NEGATIVE_GUARD = [
    "Attentionを使うモデル構造の名前を答えてください。",
    "Pythonはどのような言語ですか。",
    "CPUとGPUの違いを説明してください。",
    "NVIDIAのGPU計算技術は何ですか。",
    "画像認識モデルについて説明してください。",
    "AIモデルの性能比較方法を教えてください。",
    "ニューラルネットの構造とは何ですか。",
    "学習済みモデルの推論速度を測る方法は何ですか。",
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
    p.add_argument("--output",default="model/model-gpu-v1.1.0-llm-second-stage.pt")
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--lr",type=float,default=7e-4)
    p.add_argument("--seed",type=int,default=43)
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

    det=LLMDetector(Xtr.shape[1],32).to(device)
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
    print("LLM second-stage detector training")
    print("----------------------------------")
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
