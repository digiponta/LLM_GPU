# evaluate_cpu_name_binding_v0102.py

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v08 import load_semantic_adapter_v08_checkpoint
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_BINDING = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"


CPU_ANCHORS = [
    "CPU",
    "Central Processing Unit",
    "中央処理装置",
    "中央演算処理装置",
]
GPU_ANCHORS = [
    "GPU",
    "Graphics Processing Unit",
    "GPUは並列計算を得意とする装置です。",
]

PROBES = [
    ("English full name", "Central Processing Unit"),
    ("Japanese standard", "中央処理装置"),
    ("Japanese alternate", "中央演算処理装置"),
    ("Central mapping", "Centralは中央です。"),
    ("Processing mapping", "Processingは処理です。"),
    ("Unit mapping", "Unitは装置です。"),
    ("Composed Japanese", "中央で演算や命令処理を行う装置"),
    ("Center processing device", "コンピュータ中央で処理を担う装置"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--binding", default=DEFAULT_BINDING)
    return p.parse_args()


@torch.no_grad()
def encode(model, adapter, projection, tokenizer, text):
    device = next(model.parameters()).device
    ids = tokenizer.encode(f"人: {text}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = model.forward_hidden(x)[:, -1, :]
    adapted = adapter(hidden)
    return projection(adapted)[0]


def centroid(vectors):
    return F.normalize(torch.stack(vectors).mean(dim=0), dim=0)


def sim(a, b):
    return float(torch.dot(a, b).item())


def main():
    args = parse_args()
    for f in (args.tokenizer, args.model, args.semantic_adapter, args.binding):
        if not Path(f).exists():
            raise FileNotFoundError(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    adapter, heads, hierarchy_head, semantic_checkpoint = (
        load_semantic_adapter_v08_checkpoint(args.semantic_adapter, device)
    )
    projection, binding_checkpoint = load_cpu_name_binding_checkpoint(
        args.binding, device
    )

    for module in (model, adapter, heads, hierarchy_head, projection):
        module.eval()

    cpu_vecs = [encode(model, adapter, projection, tokenizer, x) for x in CPU_ANCHORS]
    gpu_vecs = [encode(model, adapter, projection, tokenizer, x) for x in GPU_ANCHORS]
    cpu_c = centroid(cpu_vecs)
    gpu_c = centroid(gpu_vecs)

    print()
    print("====================================================")
    print(" CPU Name-Meaning Binding v0.10.2 Evaluation")
    print("====================================================")
    print("Device               :", device)
    print("Base checkpoint loss :", base_checkpoint.get("loss"))
    print("Semantic adapter loss:", semantic_checkpoint.get("loss"))
    print("Binding loss         :", binding_checkpoint.get("loss"))
    print()

    print("Direct name equivalence")
    print("-----------------------")
    english = cpu_vecs[1]
    jp1 = cpu_vecs[2]
    jp2 = cpu_vecs[3]
    print(f"Central Processing Unit <-> 中央処理装置     : {sim(english, jp1):.6f}")
    print(f"Central Processing Unit <-> 中央演算処理装置 : {sim(english, jp2):.6f}")
    print(f"中央処理装置 <-> 中央演算処理装置            : {sim(jp1, jp2):.6f}")
    print()

    results = {}
    print("Lexical/semantic probes")
    print("-----------------------")
    for name, prompt in PROBES:
        z = encode(model, adapter, projection, tokenizer, prompt)
        cpu_s = sim(z, cpu_c)
        gpu_s = sim(z, gpu_c)
        results[name] = (cpu_s, gpu_s)
        print(
            f"{name:25s} CPU={cpu_s:.6f} GPU={gpu_s:.6f} "
            f"margin={cpu_s-gpu_s:+.6f}"
        )
    print()

    g05_prompt = str(CASES[4]["prompt"])
    g05 = encode(model, adapter, projection, tokenizer, g05_prompt)
    g05_cpu = sim(g05, cpu_c)
    g05_gpu = sim(g05, gpu_c)

    print("G05 holdout")
    print("-----------")
    print("Prompt:", g05_prompt)
    print(f"CPU name centroid : {g05_cpu:.6f}")
    print(f"GPU name centroid : {g05_gpu:.6f}")
    print(f"CPU-GPU margin    : {g05_cpu-g05_gpu:+.6f}")
    print()

    checks = {
        "English full name -> CPU": results["English full name"][0] > results["English full name"][1],
        "Japanese standard -> CPU": results["Japanese standard"][0] > results["Japanese standard"][1],
        "Japanese alternate -> CPU": results["Japanese alternate"][0] > results["Japanese alternate"][1],
        "Central mapping -> CPU": results["Central mapping"][0] > results["Central mapping"][1],
        "Processing mapping -> CPU": results["Processing mapping"][0] > results["Processing mapping"][1],
        "Unit mapping -> CPU": results["Unit mapping"][0] > results["Unit mapping"][1],
        "Composed Japanese -> CPU": results["Composed Japanese"][0] > results["Composed Japanese"][1],
        "Center processing device -> CPU": results["Center processing device"][0] > results["Center processing device"][1],
        "G05 holdout -> CPU": g05_cpu > g05_gpu,
        "English/Japanese similarity high": min(sim(english,jp1), sim(english,jp2)) >= 0.80,
    }

    print("CPU Name-Meaning Binding Gate")
    print("-----------------------------")
    for name, ok in checks.items():
        print(f"{name:45s}: {'PASS' if ok else 'MISS'}")
    passed = sum(int(v) for v in checks.values())
    print()
    print(f"Gate score: {passed}/{len(checks)}")
    if passed == len(checks):
        print("Result: CPU English/Japanese name binding passed.")
    else:
        print("Result: CPU name binding still needs refinement.")


if __name__ == "__main__":
    main()
