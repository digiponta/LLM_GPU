# evaluate_g05_after_cpu_definition_v091.py
#
# Re-evaluate G05 after CPU/GPU definition correction using deterministic
# single-turn v0.9 soft-intent generation.
#
# No additional training is performed here.

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

PROBES = (
    ("CPU direct short", "CPUとは"),
    ("CPU direct", "CPUとは何ですか"),
    ("GPU direct short", "GPUとは"),
    ("CPU/GPU contrast", "CPUとGPUの違いは"),
    ("Original G05", "コンピュータの中心で多様な命令を処理する装置は何ですか。"),
    ("Name-request G05", "コンピュータの中心で多様な命令を処理する装置の名前は何ですか。"),
)

CPU_SIGNALS = ("CPU", "汎用", "命令", "制御", "中央処理", "Central Processing Unit")
GPU_SIGNALS = ("GPU", "並列", "多数の計算", "大量の計算")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT_HEAD)
    p.add_argument("--projection", default=DEFAULT_PROJECTION)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


def classify(reply):
    cpu = any(x in reply for x in CPU_SIGNALS)
    gpu = any(x in reply for x in GPU_SIGNALS)
    if cpu and not gpu:
        return "CPU-like"
    if gpu and not cpu:
        return "GPU-like"
    if cpu and gpu:
        return "mixed"
    return "other"


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
    intent_head, head_ckpt, labels = load_intent_head(
        args.intent_head, model, device
    )
    projection, projection_ckpt = load_projection(
        args.projection, model, labels, device
    )

    print()
    print("================================================")
    print(" G05 Re-evaluation after CPU Definition v0.9.1")
    print("================================================")
    print("Device          :", device)
    print("Base loss       :", base_ckpt.get("loss"))
    print("Intent-head loss:", head_ckpt.get("loss"))
    print("Projection loss :", projection_ckpt.get("loss"))
    print("Alpha           :", projection.alpha)
    print("Generation      : greedy / temperature=0 / history=0")
    print("Training        : none")
    print()

    for name, prompt_text in PROBES:
        prompt = build_prompt([], prompt_text, 0)
        reply, count, intent_prob = generate_reply(
            model,
            tokenizer,
            intent_head,
            projection,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=20,
            repetition_penalty=1.05,
        )
        top_intents = sorted(
            zip(labels, intent_prob),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        print(name)
        print("-" * len(name))
        print("Prompt :", prompt_text)
        print("Reply  :", reply)
        print("Class  :", classify(reply))
        print("Tokens :", count)
        print(
            "Top intents:",
            ", ".join(f"{label}={prob:.4f}" for label, prob in top_intents)
        )
        print()


if __name__ == "__main__":
    main()
