# train_semantic_encoder_adapter_v062.py
#
# Semantic Encoder Adapter v0.6.2: G05-Neighborhood Binding
#
# Continues from v0.6.1.
# Exact G05 is excluded.
# Adapter and concept/attribute heads are frozen.
# Only the 7-axis hierarchy head is trained.

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List,Tuple

import torch
import torch.nn.functional as F

from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v06 import HIERARCHY_LABELS
from semantic_encoder_adapter_v061 import load_semantic_adapter_v061_checkpoint
from semantic_encoder_adapter_v062 import save_semantic_adapter_v062_checkpoint
from tokenizer_bpe import Tokenizer
from train_semantic_encoder_adapter_v03 import build_rows
from train_semantic_encoder_adapter_v01 import AdapterDataset

DEFAULT_TOKENIZER="model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL="model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INIT="model/model-gpu-v0.9.1-semantic-adapter-v061.pt"
DEFAULT_OUTPUT="model/model-gpu-v0.9.1-semantic-adapter-v062.pt"
SEED=42

# +1 => heterogeneous > homogeneous, -1 => homogeneous > heterogeneous.
NEIGHBORHOOD_ROWS:List[Tuple[str,int]]=[
    ("コンピュータの中心となる装置は、幅広い種類の命令を処理します。",+1),
    ("コンピュータの中核でさまざまな命令を実行するのはCPUです。",+1),
    ("中心的な処理装置は異なる種類の命令を柔軟に扱います。",+1),
    ("多様な命令を処理する装置は、同じ演算だけを繰り返す装置とは異なります。",+1),
    ("幅広い命令を扱う汎用プロセッサはCPUです。",+1),
    ("さまざまな命令を順番に実行する中心的なプロセッサはCPUです。",+1),
    ("異なる命令を処理する中核装置は、分岐やメモリ操作も扱います。",+1),
    ("多種類の命令を処理するCPUは、演算だけでなく制御命令も実行します。",+1),
    ("コンピュータの中心で幅広い命令を実行する装置は汎用CPUです。",+1),
    ("中心となるCPUは多様な命令を処理し、プログラム全体を制御します。",+1),
    ("多様な命令には計算以外に分岐、ロード、ストア、制御も含まれます。",+1),
    ("幅広い命令という表現は、同種計算の大量反復ではなく異種命令処理を示します。",+1),

    ("同じ種類の演算を大量に並列実行する装置はGPUです。",-1),
    ("多数の同型計算を繰り返す高スループット処理はGPU向きです。",-1),
    ("似た演算を大量のデータに適用する処理はhomogeneous computationです。",-1),
    ("同種計算の大量並列はGPU側の特徴です。",-1),
]


def parse_args():
    p=argparse.ArgumentParser(description="Train v0.6.2 G05-neighborhood binding.")
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--init-adapter",default=DEFAULT_INIT)
    p.add_argument("--output",default=DEFAULT_OUTPUT)
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--patience",type=int,default=10)
    p.add_argument("--neighborhood-weight",type=float,default=2.0)
    p.add_argument("--neighborhood-margin",type=float,default=0.20)
    p.add_argument("--preservation-weight",type=float,default=1.0)
    return p.parse_args()


@torch.no_grad()
def encode_prompt_hidden(model,tokenizer,prompt):
    device=next(model.parameters()).device
    text=f"人: {prompt}\nAI: "
    ids=tokenizer.encode(text,add_bos=True)[-model.context_length:]
    x=torch.tensor([ids],dtype=torch.long,device=device)
    return model.forward_hidden(x)[:,-1,:][0].detach()


def exact_overlap():
    dev={str(case["prompt"]) for case in CASES}
    return sorted(prompt for prompt,_ in NEIGHBORHOOD_ROWS if prompt in dev)


def direct_margin_loss(logits,styles,margin):
    probs=torch.sigmoid(logits)
    hi=HIERARCHY_LABELS.index("heterogeneous_instruction")
    ho=HIERARCHY_LABELS.index("homogeneous_computation")
    h=probs[:,hi]; m=probs[:,ho]
    cpu=styles>0; gpu=styles<0
    losses=[]
    if cpu.any():
        losses.append(F.relu(margin+m[cpu]-h[cpu]).mean())
    if gpu.any():
        losses.append(F.relu(margin+h[gpu]-m[gpu]).mean())
    return torch.stack(losses).mean()


def main():
    args=parse_args()
    torch.manual_seed(SEED); random.seed(SEED)
    for filename in (args.tokenizer,args.model,args.init_adapter):
        if not Path(filename).exists(): raise FileNotFoundError(filename)

    overlaps=exact_overlap()
    if overlaps:
        raise RuntimeError("Exact overlap with fixed DEV prompts: "+repr(overlaps))

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=Tokenizer.load(args.tokenizer)
    base_model,base_checkpoint=LanguageModel.load_checkpoint(args.model,device=device)
    base_model.eval()
    for p in base_model.parameters(): p.requires_grad_(False)

    adapter,heads,hierarchy_head,init_checkpoint=load_semantic_adapter_v061_checkpoint(
        args.init_adapter,device
    )
    adapter.eval(); heads.eval()
    for p in adapter.parameters(): p.requires_grad_(False)
    for p in heads.parameters(): p.requires_grad_(False)

    neighborhood_hidden=torch.stack([
        encode_prompt_hidden(base_model,tokenizer,p)
        for p,_s in NEIGHBORHOOD_ROWS
    ],dim=0).to(device)
    styles=torch.tensor([s for _p,s in NEIGHBORHOOD_ROWS],dtype=torch.long,device=device)

    base_rows=build_rows()
    base_set=AdapterDataset(base_rows,base_model,tokenizer)
    preserve_hidden=torch.stack([x[0] for x in base_set.items],dim=0).to(device)

    with torch.no_grad():
        start_preserve_probs=torch.sigmoid(
            hierarchy_head(adapter(preserve_hidden))
        ).detach()

    params=list(hierarchy_head.parameters())
    optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.6.2 Training")
    print("====================================================")
    print("Device                     :",device)
    if device.type=="cuda": print("GPU                        :",torch.cuda.get_device_name(0))
    print("Base checkpoint loss       :",base_checkpoint.get("loss"))
    print("Initialization             :",args.init_adapter)
    print("Initial adapter loss       :",init_checkpoint.get("loss"))
    print("Base encoder               : frozen")
    print("Adapter                    : frozen")
    print("Concept/attribute heads    : frozen")
    print("Hierarchy head             : trainable")
    print("Neighborhood rows          :",len(NEIGHBORHOOD_ROWS))
    print("Exact DEV overlap          :",len(overlaps))
    print("Learning rate              :",args.lr)
    print("Neighborhood margin        :",args.neighborhood_margin)
    print("Neighborhood weight        :",args.neighborhood_weight)
    print("Preservation weight        :",args.preservation_weight)
    print()

    best=float("inf"); best_epoch=0; best_head=None; bad=0
    for epoch in range(1,args.epochs+1):
        hierarchy_head.train()
        optimizer.zero_grad(set_to_none=True)

        adapted=adapter(neighborhood_hidden)
        logits=hierarchy_head(adapted)
        direct=direct_margin_loss(logits,styles,args.neighborhood_margin)

        preserve_probs=torch.sigmoid(hierarchy_head(adapter(preserve_hidden)))
        preservation=F.mse_loss(preserve_probs,start_preserve_probs)

        total=args.neighborhood_weight*direct+args.preservation_weight*preservation
        total.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0)
        optimizer.step()

        value=float(total.item())
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| loss={value:.4f} direct={direct.item():.4f} "
            f"pres={preservation.item():.5f}"
        )

        if value<best-1e-5:
            best=value; best_epoch=epoch; bad=0
            best_head={k:v.detach().cpu().clone() for k,v in hierarchy_head.state_dict().items()}
        else:
            bad+=1
            if bad>=args.patience:
                print("Early stopping.")
                break

    hierarchy_head.load_state_dict(best_head)
    save_semantic_adapter_v062_checkpoint(
        args.output,adapter,heads,hierarchy_head,
        epoch=best_epoch,loss=best,base_model=args.model,init_adapter=args.init_adapter,
        learning_rate=args.lr,neighborhood_weight=args.neighborhood_weight,
        neighborhood_margin=args.neighborhood_margin,
        preservation_weight=args.preservation_weight,
    )
    print()
    print("Semantic Encoder Adapter v0.6.2 training completed.")
    print("Best epoch       :",best_epoch)
    print("Best loss        :",f"{best:.6f}")
    print("Checkpoint saved :",args.output)


if __name__=="__main__":
    main()
