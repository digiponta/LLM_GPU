# evaluate_strong_intent_coupling_v094.py
#
# Evaluate v0.9.4 FiLM coupling on fixed 30-case benchmark plus focused probes.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from evaluate_partial_intent_v09 import CASES, dimension_match, semantic_match
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from strong_intent_coupling_v094 import (
    forward_strong_intent,
    load_checkpoint,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_HEAD = "model/model-gpu-v0.9.2-intent-head-implicit-cpu.pt"
DEFAULT_COUPLING = "model/model-gpu-v0.9.4-strong-intent-film.pt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "

FOCUSED = (
    ("CPU direct", "CPUとは何ですか。"),
    ("GPU direct", "GPUとは何ですか。"),
    ("CPU/GPU contrast", "CPUとGPUの違いは何ですか。"),
    ("Original G05", "コンピュータの中心で多様な命令を処理する装置は何ですか。"),
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
    p.add_argument("--intent-head", default=DEFAULT_HEAD)
    p.add_argument("--coupling", default=DEFAULT_COUPLING)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


@torch.no_grad()
def prompt_intent(model, tokenizer, head, text):
    ids = tokenizer.encode(f"{USER_PREFIX}{text}\n{AI_PREFIX}", add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    hidden = model.forward_hidden(x)
    probs = torch.sigmoid(head(hidden[:, -1, :]))
    return probs[0]


@torch.no_grad()
def generate(model, tokenizer, head, coupling, text, max_new_tokens=96):
    prompt = f"{USER_PREFIX}{text}\n{AI_PREFIX}"
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    generated = list(prompt_ids)
    response_ids: List[int] = []
    device = next(model.parameters()).device

    probs = prompt_intent(model, tokenizer, head, text).unsqueeze(0)
    scale, shift = coupling(probs)

    for _ in range(max_new_tokens):
        context = generated[-model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        hidden = forward_strong_intent(
            model,
            x,
            scale,
            shift,
            coupling.inject_after,
        )
        logits = model.lm_head(hidden[:, -1, :])[0].clone()

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
    return reply.strip(), probs[0]


def main():
    args = parse_args()
    for filename in (
        args.tokenizer, args.model, args.intent_head, args.coupling
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ckpt = LanguageModel.load_checkpoint(args.model, device=device)
    head, head_ckpt, labels = load_intent_head(args.intent_head, model, device)
    coupling, coupling_ckpt = load_checkpoint(
        args.coupling, model, labels, device
    )

    print()
    print("==========================================")
    print(" v0.9.4 Strong Intent Coupling Evaluation")
    print("==========================================")
    print("Device           :", device)
    print("Base loss        :", base_ckpt.get("loss"))
    print("Intent-head loss :", head_ckpt.get("loss"))
    print("Coupling loss    :", coupling_ckpt.get("loss"))
    print("Inject after     :", coupling.inject_after)
    print("Scale limit      :", coupling.scale_limit)
    print("Shift limit      :", coupling.shift_limit)
    print("Generation       : greedy / history=0")
    print()

    semantic_pass = strict_pass = fluent_pass = legacy_pass = 0
    rows = []

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        reply, _ = generate(
            model, tokenizer, head, coupling, prompt_text, args.max_new_tokens
        )
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
        reply, probs = generate(
            model, tokenizer, head, coupling, text, args.max_new_tokens
        )
        ranked = sorted(
            zip(labels, probs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:6]
        print(name)
        print("  Prompt :", text)
        print("  Intent :", ", ".join(f"{k}={v:.4f}" for k, v in ranked))
        print("  Reply  :", reply)


if __name__ == "__main__":
    main()
