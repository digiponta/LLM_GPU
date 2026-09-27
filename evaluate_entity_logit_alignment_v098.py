# evaluate_entity_logit_alignment_v098.py
#
# Evaluate semantic probabilities, entity first-token logits and generation.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from entity_logit_alignment_v098 import (
    ENTITY_BY_TAG,
    entity_token_ids,
    load_checkpoint,
)
from evaluate_partial_intent_v09 import CASES, dimension_match, semantic_match
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.9.8-entity-logit-aligned.pt"
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
def probe_state(model, semantic_head, tokenizer, entity_ids, text):
    ids = tokenizer.encode(f"{USER_PREFIX}{text}\n{AI_PREFIX}", add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    hidden = model.forward_hidden(x)
    h = hidden[:, -1, :]
    sem = torch.sigmoid(semantic_head(h))[0]
    logits = model.lm_head(h)[0]
    entity = {
        tag: float(logits[tid].item())
        for tag, tid in entity_ids.items()
    }
    top_value, top_id = torch.max(logits, dim=0)
    return sem, entity, int(top_id.item()), float(top_value.item())


@torch.no_grad()
def generate(model, tokenizer, text, max_new_tokens):
    generated = tokenizer.encode(
        f"{USER_PREFIX}{text}\n{AI_PREFIX}", add_bos=True
    )
    response_ids: List[int] = []
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        x = torch.tensor(
            [generated[-model.context_length:]],
            dtype=torch.long,
            device=device,
        )
        logits = model(x)[0, -1, :].clone()
        for tid in set(response_ids):
            if logits[tid] >= 0:
                logits[tid] /= 1.05
            else:
                logits[tid] *= 1.05
        next_id = int(torch.argmax(logits).item())
        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)
        response_ids.append(next_id)
        if "\n" in tokenizer.decode(response_ids, skip_special_tokens=True):
            break

    reply = tokenizer.decode(response_ids, skip_special_tokens=True)
    return reply.split("\n", 1)[0].strip()


def top(labels, probs, k=6):
    return sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True)[:k]


def main():
    args = parse_args()
    for filename in (args.tokenizer, args.model):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, semantic_head, labels, ckpt = load_checkpoint(args.model, device)
    ids = entity_token_ids(tokenizer)

    print()
    print("==========================================")
    print(" v0.9.8 Semantic Entity Logit Eval")
    print("==========================================")
    print("Device          :", device)
    print("Checkpoint loss :", ckpt.get("loss"))
    print("Source          :", ckpt.get("source_checkpoint"))
    print("Generation      : direct LM / greedy / history=0")
    print()

    semantic_pass = strict_pass = fluent_pass = legacy_pass = 0
    for case in CASES:
        prompt = str(case["prompt"])
        reply = generate(model, tokenizer, prompt, args.max_new_tokens)
        legacy_ok, _m, _c = semantic_match(
            reply, case["required_all"], case["forbidden"]
        )
        dims = dimension_match(
            reply, prompt, str(case["intent"]),
            case["required_all"], case["forbidden"],
        )
        semantic_pass += int(dims["semantic_ok"])
        strict_pass += int(dims["strict_ok"])
        fluent_pass += int(dims["fluent_ok"])
        legacy_pass += int(legacy_ok)

    print("Fixed 30-case benchmark")
    print("-----------------------")
    print(f"semantic : {semantic_pass}/30")
    print(f"strict   : {strict_pass}/30")
    print(f"fluency  : {fluent_pass}/30")
    print(f"legacy   : {legacy_pass}/30")
    print()

    print("Focused causal probes")
    print("---------------------")
    for name, prompt in FOCUSED:
        sem, entity, top_id, top_value = probe_state(
            model, semantic_head, tokenizer, ids, prompt
        )
        reply = generate(model, tokenizer, prompt, args.max_new_tokens)
        ranked_entity = sorted(entity.items(), key=lambda x: x[1], reverse=True)
        print(name)
        print("  Prompt :", prompt)
        print("  Sem    :", ", ".join(f"{k}={v:.4f}" for k, v in top(labels, sem)))
        print(
            "  Entity :",
            ", ".join(
                f"{tag.replace('tech_','')}={value:.3f}"
                for tag, value in ranked_entity
            ),
        )
        print(
            "  Top-1  :",
            f"id={top_id} token={tokenizer.decode([top_id])!r} logit={top_value:.3f}",
        )
        print("  Reply  :", reply)


if __name__ == "__main__":
    main()
