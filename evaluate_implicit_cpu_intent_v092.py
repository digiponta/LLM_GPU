# evaluate_implicit_cpu_intent_v092.py
#
# Compare original vs corrected intent head on direct CPU/GPU prompts and G05,
# then test generation through the frozen corrected v0.9.1 projection.

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
DEFAULT_OLD_HEAD = "model/model-gpu-v0.8-intent-head.pt"
DEFAULT_NEW_HEAD = "model/model-gpu-v0.9.2-intent-head-implicit-cpu.pt"
DEFAULT_PROJECTION = "model/model-gpu-v0.9.1-soft-intent-cpu-definition.pt"

PROBES = (
    ("CPU direct", "CPUとは何ですか。"),
    ("GPU direct", "GPUとは何ですか。"),
    ("Original G05", "コンピュータの中心で多様な命令を処理する装置は何ですか。"),
    ("Name-request G05", "コンピュータの中心で多様な命令を処理する装置の名前は何ですか。"),
    ("CPU paraphrase", "中央でさまざまな命令を処理する装置の名称は何ですか。"),
    ("GPU paraphrase", "大量の同種計算を並列処理する装置の名称は何ですか。"),
    ("Error replay", "プログラム実行に失敗した原因を調べたいです。"),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--old-head", default=DEFAULT_OLD_HEAD)
    p.add_argument("--new-head", default=DEFAULT_NEW_HEAD)
    p.add_argument("--projection", default=DEFAULT_PROJECTION)
    return p.parse_args()


@torch.no_grad()
def intent_probs(model, tokenizer, head, text):
    ids = tokenizer.encode(f"人: {text}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    hidden = model.forward_hidden(x)[:, -1, :]
    return torch.sigmoid(head(hidden))[0]


def top(labels, probs, k=6):
    pairs = sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True)
    return pairs[:k]


def main():
    args = parse_args()
    for filename in (
        args.tokenizer, args.model, args.old_head, args.new_head, args.projection
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ckpt = LanguageModel.load_checkpoint(args.model, device=device)
    old_head, old_ckpt, labels = load_intent_head(args.old_head, model, device)
    new_head, new_ckpt, new_labels = load_intent_head(args.new_head, model, device)
    if new_labels != labels:
        raise RuntimeError("Intent label order changed.")
    projection, projection_ckpt = load_projection(
        args.projection, model, labels, device
    )

    cpu_i = labels.index("tech_cpu")
    gpu_i = labels.index("tech_gpu")
    err_i = labels.index("debug_error")

    print()
    print("=============================================")
    print(" v0.9.2 Implicit CPU Intent Evaluation")
    print("=============================================")
    print("Device              :", device)
    print("Base loss           :", base_ckpt.get("loss"))
    print("Old head loss       :", old_ckpt.get("loss"))
    print("New head loss       :", new_ckpt.get("loss"))
    print("Projection loss     :", projection_ckpt.get("loss"))
    print("Projection           : frozen v0.9.1 CPU-definition")
    print("Generation           : greedy / temperature=0 / history=0")
    print()

    for name, text in PROBES:
        old = intent_probs(model, tokenizer, old_head, text)
        new = intent_probs(model, tokenizer, new_head, text)

        prompt = build_prompt([], text, 0)
        reply, _n, _p = generate_reply(
            model,
            tokenizer,
            new_head,
            projection,
            prompt,
            max_new_tokens=96,
            temperature=0.0,
            top_k=20,
            repetition_penalty=1.05,
        )

        print(name)
        print("-" * len(name))
        print("Prompt:", text)
        print(
            f"OLD cpu={old[cpu_i]:.4f} gpu={old[gpu_i]:.4f} err={old[err_i]:.4f} "
            f"cpu-gpu={old[cpu_i]-old[gpu_i]:+.4f}"
        )
        print(
            f"NEW cpu={new[cpu_i]:.4f} gpu={new[gpu_i]:.4f} err={new[err_i]:.4f} "
            f"cpu-gpu={new[cpu_i]-new[gpu_i]:+.4f}"
        )
        print(
            "Top NEW:",
            ", ".join(f"{label}={prob:.4f}" for label, prob in top(labels, new))
        )
        print("Reply:", reply)
        print()


if __name__ == "__main__":
    main()
