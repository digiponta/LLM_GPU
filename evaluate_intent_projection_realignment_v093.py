# evaluate_intent_projection_realignment_v093.py
#
# LLM_GPU v0.9.3
# Evaluate old-vs-realigned projection under the corrected v0.9.2 intent head.
#
# No training is performed in this file.

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
DEFAULT_HEAD = "model/model-gpu-v0.9.2-intent-head-implicit-cpu.pt"
DEFAULT_OLD_PROJECTION = "model/model-gpu-v0.9.1-soft-intent-cpu-definition.pt"
DEFAULT_NEW_PROJECTION = "model/model-gpu-v0.9.3-soft-intent-realigned.pt"

PROBES = (
    ("CPU direct", "CPUとは何ですか。"),
    ("GPU direct", "GPUとは何ですか。"),
    ("CPU/GPU contrast", "CPUとGPUの違いは何ですか。"),
    ("Original G05", "コンピュータの中心で多様な命令を処理する装置は何ですか。"),
    ("Name-request G05", "コンピュータの中心で多様な命令を処理する装置の名前は何ですか。"),
    ("CPU paraphrase", "中央でさまざまな命令を処理する装置の名称は何ですか。"),
    ("GPU paraphrase", "大量の同種計算を並列処理する装置の名称は何ですか。"),
    ("Error replay", "プログラム実行に失敗した原因を調べたいです。"),
    ("Transformer replay", "Attentionを中心に使う代表的な構造は何ですか。"),
    ("Python replay", "読みやすさで知られる汎用言語を一つ挙げてください。"),
)

CPU_SIGNALS = ("CPU", "汎用", "命令", "制御", "中央処理")
GPU_SIGNALS = ("GPU", "並列", "大量", "多数の計算")
ERROR_SIGNALS = ("エラー", "コード", "原因")
TRANSFORMER_SIGNALS = ("Transformer", "Attention")
PYTHON_SIGNALS = ("Python", "プログラミング言語")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_HEAD)
    p.add_argument("--old-projection", default=DEFAULT_OLD_PROJECTION)
    p.add_argument("--new-projection", default=DEFAULT_NEW_PROJECTION)
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
    if any(x in reply for x in ERROR_SIGNALS):
        return "error-like"
    if any(x in reply for x in TRANSFORMER_SIGNALS):
        return "transformer-like"
    if any(x in reply for x in PYTHON_SIGNALS):
        return "python-like"
    return "other"


@torch.no_grad()
def run_one(model, tokenizer, head, projection, text, max_new_tokens):
    prompt = build_prompt([], text, 0)
    reply, count, probs = generate_reply(
        model,
        tokenizer,
        head,
        projection,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=20,
        repetition_penalty=1.05,
    )
    return reply, count, probs


def main():
    args = parse_args()
    for filename in (
        args.tokenizer,
        args.model,
        args.intent_head,
        args.old_projection,
        args.new_projection,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ckpt = LanguageModel.load_checkpoint(args.model, device=device)
    head, head_ckpt, labels = load_intent_head(args.intent_head, model, device)
    old_projection, old_ckpt = load_projection(
        args.old_projection, model, labels, device
    )
    new_projection, new_ckpt = load_projection(
        args.new_projection, model, labels, device
    )

    print()
    print("==============================================")
    print(" v0.9.3 Intent-Projection Realignment Eval")
    print("==============================================")
    print("Device              :", device)
    print("Base loss           :", base_ckpt.get("loss"))
    print("Intent-head loss    :", head_ckpt.get("loss"))
    print("Old projection loss :", old_ckpt.get("loss"))
    print("New projection loss :", new_ckpt.get("loss"))
    print("Head                : corrected v0.9.2")
    print("Generation          : greedy / temperature=0 / history=0")
    print()

    for name, text in PROBES:
        old_reply, old_count, probs = run_one(
            model, tokenizer, head, old_projection, text, args.max_new_tokens
        )
        new_reply, new_count, _ = run_one(
            model, tokenizer, head, new_projection, text, args.max_new_tokens
        )

        ranked = sorted(
            zip(labels, probs),
            key=lambda x: x[1],
            reverse=True,
        )[:6]

        print(name)
        print("-" * len(name))
        print("Prompt:", text)
        print(
            "Intent:",
            ", ".join(f"{label}={prob:.4f}" for label, prob in ranked)
        )
        print(
            f"OLD [{classify(old_reply):16s}] ({old_count:2d} tok): {old_reply}"
        )
        print(
            f"NEW [{classify(new_reply):16s}] ({new_count:2d} tok): {new_reply}"
        )
        print()


if __name__ == "__main__":
    main()
