# evaluate_cpu_definition_projection_v091.py
#
# Deterministic direct CPU/GPU definition probes for the corrected v0.9.1
# soft-intent projection checkpoint.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chat_v09 import build_prompt, generate_reply
from intent_conditioning_v09 import load_intent_head, load_projection
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INTENT_HEAD = "model/model-gpu-v0.8-intent-head.pt"
DEFAULT_PROJECTION = "model/model-gpu-v0.9.1-soft-intent-cpu-definition.pt"

CPU_PROBES = (
    "CPUとは",
    "CPUとは何ですか。",
    "CPUって何ですか。",
    "CPUを簡単に説明して。",
    "CPUの定義を教えて。",
    "Central Processing Unitとは何ですか。",
)
GPU_PROBES = (
    "GPUとは",
    "GPUとは何ですか。",
    "GPUって何ですか。",
    "GPUを簡単に説明して。",
)

CPU_GOOD = ("CPU", "汎用", "命令", "制御", "中央処理", "Central Processing Unit")
CPU_BAD = ("多数の計算を並列", "大量の並列", "並列計算を得意")
GPU_GOOD = ("GPU", "並列", "同時", "大量")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT_HEAD)
    p.add_argument("--projection", default=DEFAULT_PROJECTION)
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def cpu_ok(reply):
    return any(x in reply for x in CPU_GOOD) and not any(x in reply for x in CPU_BAD)


def gpu_ok(reply):
    return any(x in reply for x in GPU_GOOD)


def main():
    args = parse_args()
    for filename in (
        args.tokenizer, args.model, args.intent_head, args.projection
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ckpt = LanguageModel.load_checkpoint(args.model, device=device)
    intent_head, _head_ckpt, labels = load_intent_head(
        args.intent_head, model, device
    )
    projection, projection_ckpt = load_projection(
        args.projection, model, labels, device
    )

    print()
    print("===============================================")
    print(" CPU Definition Projection v0.9.1 Evaluation")
    print("===============================================")
    print("Device          :", device)
    print("Base loss       :", base_ckpt.get("loss"))
    print("Projection loss :", projection_ckpt.get("loss"))
    print("Alpha           :", projection.alpha)
    print("Generation      : greedy / temperature=0")
    print()

    cpu_pass = 0
    for prompt_text in CPU_PROBES:
        prompt = build_prompt([], prompt_text, 0)
        reply, _n, _p = generate_reply(
            model, tokenizer, intent_head, projection, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=20,
            repetition_penalty=1.05,
        )
        ok = cpu_ok(reply)
        cpu_pass += int(ok)
        print(f"CPU {'PASS' if ok else 'MISS'} | {prompt_text}")
        print("  ->", reply)

    print()
    gpu_pass = 0
    for prompt_text in GPU_PROBES:
        prompt = build_prompt([], prompt_text, 0)
        reply, _n, _p = generate_reply(
            model, tokenizer, intent_head, projection, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=20,
            repetition_penalty=1.05,
        )
        ok = gpu_ok(reply)
        gpu_pass += int(ok)
        print(f"GPU {'PASS' if ok else 'MISS'} | {prompt_text}")
        print("  ->", reply)

    print()
    print("Summary")
    print("-------")
    print(f"CPU direct definitions : {cpu_pass}/{len(CPU_PROBES)}")
    print(f"GPU direct definitions : {gpu_pass}/{len(GPU_PROBES)}")


if __name__ == "__main__":
    main()
