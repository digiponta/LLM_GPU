# evaluate_output_alignment_v097.py
#
# Direct evaluation of v0.9.7 output-aligned adapted LM.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from evaluate_partial_intent_v09 import CASES, dimension_match, semantic_match
from output_alignment_v097 import load_checkpoint
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.9.7-output-aligned.pt"
USER_PREFIX = "人: "
AI_PREFIX = "AI: "

FOCUSED = (
    ("CPU direct", "CPUとは何ですか。"),
    ("GPU direct", "GPUとは何ですか。"),
    ("CPU/GPU contrast", "CPUとGPUの違いは何ですか。"),
    ("Original G05", "コンピュータの中心で多様な命令を処理する装置は何ですか。"),
    ("G05 no-question", "コンピュータの中心で多様な命令を処理する装置とは"),
    ("Name-request G05", "コンピュータの中心で多様な命令を処理する装置の名前は何ですか。"),
    ("CPU paraphrase", "中央でさまざまな命令を処理する装置の名称は何ですか。"),
    ("GPU paraphrase", "大量の同種計算を並列処理する装置の名称は何ですか。"),
    ("Transformer", "Attentionを中心に使う代表的な構造は何ですか。"),
    ("Python", "読みやすさで知られる汎用言語を一つ挙げてください。"),
    ("Error", "プログラム実行に失敗した原因を調べたいです。"),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


@torch.no_grad()
def semantic_probs(model, semantic_head, tokenizer, text):
    ids = tokenizer.encode(
        f"{USER_PREFIX}{text}\n{AI_PREFIX}", add_bos=True
    )
    ids = ids[-model.context_length:]
    x = torch.tensor(
        [ids], dtype=torch.long, device=next(model.parameters()).device
    )
    hidden = model.forward_hidden(x)
    return torch.sigmoid(semantic_head(hidden[:, -1, :]))[0]


@torch.no_grad()
def generate(model, tokenizer, text, max_new_tokens=96):
    prompt = f"{USER_PREFIX}{text}\n{AI_PREFIX}"
    generated = list(tokenizer.encode(prompt, add_bos=True))
    response_ids: List[int] = []
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        context = generated[-model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :].clone()

        for token_id in set(response_ids):
            if logits[token_id] >= 0:
                logits[token_id] /= 1.05
            else:
                logits[token_id] *= 1.05

        next_id = int(torch.argmax(logits).item())
        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)
        response_ids.append(next_id)
        decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
        if "\n" in decoded:
            break

    reply = tokenizer.decode(response_ids, skip_special_tokens=True)
    for marker in ("\n人:", "\nAI:", "\n"):
        if marker in reply:
            reply = reply.split(marker, 1)[0]
    return reply.strip()


def top(labels, probs, k=6):
    return sorted(
        zip(labels, probs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:k]


def main():
    args = parse_args()
    for filename in (args.tokenizer, args.model):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, semantic_head, labels, ckpt = load_checkpoint(args.model, device)

    print()
    print("====================================")
    print(" v0.9.7 Output Alignment Eval")
    print("====================================")
    print("Device          :", device)
    print("Checkpoint loss :", ckpt.get("loss"))
    print("Source          :", ckpt.get("source_checkpoint"))
    print("LM-head LR      :", ckpt.get("lm_head_learning_rate"))
    print("Generation      : direct LM / greedy / history=0")
    print()

    semantic_pass = strict_pass = fluent_pass = legacy_pass = 0
    rows = []

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        reply = generate(model, tokenizer, prompt_text, args.max_new_tokens)
        legacy_ok, _m, _c = semantic_match(
            reply, case["required_all"], case["forbidden"]
        )
        dims = dimension_match(
            reply,
            prompt_text,
            str(case["intent"]),
            case["required_all"],
            case["forbidden"],
        )
        semantic_pass += int(dims["semantic_ok"])
        strict_pass += int(dims["strict_ok"])
        fluent_pass += int(dims["fluent_ok"])
        legacy_pass += int(legacy_ok)
        rows.append((idx, str(case["intent"]), reply, dims["strict_ok"]))

    print("Fixed 30-case benchmark")
    print("-----------------------")
    print(f"semantic : {semantic_pass}/30")
    print(f"strict   : {strict_pass}/30")
    print(f"fluency  : {fluent_pass}/30")
    print(f"legacy   : {legacy_pass}/30")
    print()

    for idx in (5, 8, 9, 10, 27, 28):
        row = rows[idx-1]
        print(
            f"G{idx:02d} {row[1]:14s} "
            f"{'PASS' if row[3] else 'MISS'} | {row[2]}"
        )

    print()
    print("Focused probes")
    print("--------------")
    for name, text in FOCUSED:
        probs = semantic_probs(model, semantic_head, tokenizer, text)
        reply = generate(model, tokenizer, text, args.max_new_tokens)
        print(name)
        print("  Prompt :", text)
        print(
            "  Sem    :",
            ", ".join(f"{k}={v:.4f}" for k, v in top(labels, probs))
        )
        print("  Reply  :", reply)


if __name__ == "__main__":
    main()
