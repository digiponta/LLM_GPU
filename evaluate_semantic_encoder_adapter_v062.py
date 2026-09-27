# evaluate_semantic_encoder_adapter_v062.py
#
# Evaluate v0.6.2 G05-Neighborhood Binding.

from __future__ import annotations
import argparse
from pathlib import Path
import torch

from evaluate_semantic_encoder_adapter_v05 import (
    TARGET_CASES,build_centroids,encode_raw,heldout_technical_accuracy,print_case
)
from evaluate_semantic_encoder_adapter_v06 import hierarchy_values,style_scores,INSTRUCTION_PROBES
from evaluate_semantic_encoder_adapter_v061 import EXTRA_BINDING_PROBES
from model import LanguageModel
from semantic_encoder_adapter_v062 import load_semantic_adapter_v062_checkpoint
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER="model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL="model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_ADAPTER="model/model-gpu-v0.9.1-semantic-adapter-v062.pt"

NEIGHBORHOOD_PROBES=[
    ("Central diverse","コンピュータの中心で幅広い命令を処理する装置は何ですか。","CPU-like"),
    ("Core varied","中核となる装置がさまざまな命令を実行します。CPUとGPUのどちらですか。","CPU-like"),
    ("Different instructions","異なる種類の命令を処理する装置は何ですか。","CPU-like"),
    ("Broad instruction set","幅広い命令を柔軟に扱う汎用プロセッサは何ですか。","CPU-like"),
]


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--adapter",default=DEFAULT_ADAPTER)
    return p.parse_args()


def main():
    args=parse_args()
    for f in (args.tokenizer,args.model,args.adapter):
        if not Path(f).exists(): raise FileNotFoundError(f)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=Tokenizer.load(args.tokenizer)
    model,base_checkpoint=LanguageModel.load_checkpoint(args.model,device=device)
    adapter,heads,hierarchy_head,checkpoint=load_semantic_adapter_v062_checkpoint(args.adapter,device)
    model.eval(); adapter.eval(); heads.eval(); hierarchy_head.eval()

    raw_centroids=build_centroids(model,tokenizer,None)
    adapted_centroids=build_centroids(model,tokenizer,adapter)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.6.2 Evaluation")
    print("====================================================")
    print("Device              :",device)
    if device.type=="cuda": print("GPU                 :",torch.cuda.get_device_name(0))
    print("Base checkpoint loss:",base_checkpoint.get("loss"))
    print("Adapter loss        :",checkpoint.get("loss"))
    print("Init adapter        :",checkpoint.get("init_adapter"))
    print()

    g05=print_case("G05",TARGET_CASES["G05"],"tech_cpu",model,tokenizer,adapter,heads,raw_centroids,adapted_centroids)
    print()
    g08=print_case("G08",TARGET_CASES["G08"],"tech_transformer",model,tokenizer,adapter,heads,raw_centroids,adapted_centroids)
    print()

    raw_correct,total,_=heldout_technical_accuracy(model,tokenizer,raw_centroids,None)
    adapted_correct,_t,_rows=heldout_technical_accuracy(model,tokenizer,adapted_centroids,adapter)
    print("Technical held-out centroid accuracy")
    print("------------------------------------")
    print(f"Raw     : {raw_correct}/{total} ({raw_correct/total:.1%})")
    print(f"Adapted : {adapted_correct}/{total} ({adapted_correct/total:.1%})")
    print()

    probes=list(INSTRUCTION_PROBES)+list(EXTRA_BINDING_PROBES)+list(NEIGHBORHOOD_PROBES)
    results={}
    print("Instruction / neighborhood probes")
    print("---------------------------------")
    for name,prompt,expected in probes:
        adapted=adapter(encode_raw(model,tokenizer,prompt).unsqueeze(0))[0]
        values=hierarchy_values(hierarchy_head,adapted)
        cpu,gpu=style_scores(values)
        pred="CPU-like" if cpu>gpu else "GPU-like"
        delta=values["heterogeneous_instruction"]-values["homogeneous_computation"]
        results[name]=(expected,pred,delta,cpu,gpu,values)
        print(f"{name}: {prompt}")
        print(f"  expected style             : {expected}")
        print(f"  predicted style            : {pred}")
        print(f"  heterogeneous_instruction : {values['heterogeneous_instruction']:.6f}")
        print(f"  homogeneous_computation   : {values['homogeneous_computation']:.6f}")
        print(f"  direct hetero-minus-homo   : {delta:.6f}")
        print(f"  CPU-style score            : {cpu:.6f}")
        print(f"  GPU-style score            : {gpu:.6f}")
        print()

    checks={
        "G05 centroid -> CPU":g05["adapted_top"]=="tech_cpu",
        "G08 centroid -> Transformer":g08["adapted_top"]=="tech_transformer",
        "Held-out technical non-regression":adapted_correct>=raw_correct,
        "G05 style -> CPU-like":results["G05"][1]=="CPU-like",
        "G05 hetero > homo":results["G05"][2]>0.0,
        "CPU non-arithmetic -> CPU-like":results["CPU non-arithmetic"][1]=="CPU-like",
        "CPU heterogeneous -> CPU-like":results["CPU heterogeneous"][1]=="CPU-like",
        "GPU homogeneous -> GPU-like":results["GPU homogeneous"][1]=="GPU-like",
        "Repeated homogeneous math -> GPU-like":results["Repeated homogeneous math"][1]=="GPU-like",
        "Central diverse -> CPU-like":results["Central diverse"][1]=="CPU-like",
        "Core varied -> CPU-like":results["Core varied"][1]=="CPU-like",
        "Different instructions -> CPU-like":results["Different instructions"][1]=="CPU-like",
        "Broad instruction set -> CPU-like":results["Broad instruction set"][1]=="CPU-like",
    }

    print("G05-Neighborhood Binding Gate")
    print("-----------------------------")
    for name,ok in checks.items():
        print(f"{name:48s}: {'PASS' if ok else 'MISS'}")
    passed=sum(int(v) for v in checks.values())
    print()
    print(f"Gate score: {passed}/{len(checks)}")
    if passed==len(checks):
        print("Result: v0.6.2 neighborhood binding gate passed.")
    else:
        print("Result: v0.6.2 neighborhood binding gate is incomplete.")


if __name__=="__main__":
    main()
