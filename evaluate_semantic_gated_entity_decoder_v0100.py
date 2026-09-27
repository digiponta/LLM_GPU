# evaluate_semantic_gated_entity_decoder_v0100.py
#
# Compare normal v0.9.9 generation with v0.10.0 semantic-gated first-token
# entity decoding.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from entity_logit_alignment_v098 import ENTITY_BY_TAG
from evaluate_partial_intent_v09 import CASES, dimension_match, semantic_match
from semantic_gated_entity_decoder_v0100 import (
    ENTITY_TAGS,
    entity_token_ids,
    load_checkpoint,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_DECODER = "model/model-gpu-v0.10.0-semantic-gated-entity-decoder.pt"
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
    p.add_argument("--decoder", default=DEFAULT_DECODER)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--gate-threshold", type=float, default=None)
    return p.parse_args()


@torch.no_grad()
def prompt_state(model, semantic_head, decoder, tokenizer, text):
    prompt = f"{USER_PREFIX}{text}\n{AI_PREFIX}"
    ids = tokenizer.encode(prompt, add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    hidden = model.forward_hidden(x)
    h = hidden[:, -1, :]
    sem = torch.sigmoid(semantic_head(h))[0]
    gate_logit, entity_logits = decoder(h)
    gate_prob = torch.sigmoid(gate_logit)[0]
    entity_prob = torch.softmax(entity_logits, dim=-1)[0]
    return sem, gate_prob, entity_prob


@torch.no_grad()
def generate(
    model,
    semantic_head,
    decoder,
    tokenizer,
    text,
    threshold,
    max_new_tokens=96,
):
    sem, gate_prob, entity_prob = prompt_state(
        model, semantic_head, decoder, tokenizer, text
    )

    prompt = f"{USER_PREFIX}{text}\n{AI_PREFIX}"
    generated = list(tokenizer.encode(prompt, add_bos=True))
    response_ids: List[int] = []
    device = next(model.parameters()).device
    forced_tag = None

    if float(gate_prob.item()) >= threshold:
        entity_index = int(torch.argmax(entity_prob).item())
        token_ids = entity_token_ids(tokenizer)
        next_id = token_ids[entity_index]
        generated.append(next_id)
        response_ids.append(next_id)
        forced_tag = ENTITY_TAGS[entity_index]

    while len(response_ids) < max_new_tokens:
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
    return reply.strip(), sem, gate_prob, entity_prob, forced_tag


def top(labels, probs, k=6):
    return sorted(
        zip(labels, probs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:k]


def main():
    args = parse_args()
    for filename in (args.tokenizer, args.decoder):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, semantic_head, labels, decoder, ckpt, source_ckpt = load_checkpoint(
        args.decoder, device
    )
    threshold = (
        float(args.gate_threshold)
        if args.gate_threshold is not None
        else float(ckpt.get("gate_threshold", 0.50))
    )

    print()
    print("======================================")
    print(" v0.10.0 Semantic-Gated Entity Eval")
    print("======================================")
    print("Device          :", device)
    print("Decoder loss    :", ckpt.get("loss"))
    print("Source loss     :", source_ckpt.get("loss"))
    print("Gate threshold  :", threshold)
    print("LM/Semantic     : frozen v0.9.9")
    print("Entity routing  : first token only")
    print()

    semantic_pass = strict_pass = fluent_pass = legacy_pass = 0
    rows = []

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        reply, _sem, gate, entity_prob, forced = generate(
            model, semantic_head, decoder, tokenizer,
            prompt_text, threshold, args.max_new_tokens
        )
        legacy_ok, _m, _c = semantic_match(
            reply, case["required_all"], case["forbidden"]
        )
        dims = dimension_match(
            reply, prompt_text, str(case["intent"]),
            case["required_all"], case["forbidden"]
        )
        semantic_pass += int(dims["semantic_ok"])
        strict_pass += int(dims["strict_ok"])
        fluent_pass += int(dims["fluent_ok"])
        legacy_pass += int(legacy_ok)
        rows.append((
            idx, str(case["intent"]), reply, dims["strict_ok"],
            float(gate.item()), forced,
        ))

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
            f"{'PASS' if row[3] else 'MISS'} "
            f"gate={row[4]:.3f} forced={row[5]} | {row[2]}"
        )

    print()
    print("Focused routing probes")
    print("----------------------")
    for name, text in FOCUSED:
        reply, sem, gate, entity_prob, forced = generate(
            model, semantic_head, decoder, tokenizer,
            text, threshold, args.max_new_tokens
        )
        print(name)
        print("  Prompt :", text)
        print(
            "  Sem    :",
            ", ".join(f"{k}={v:.4f}" for k, v in top(labels, sem))
        )
        print("  Gate   :", f"{float(gate.item()):.4f}")
        print(
            "  Entity :",
            ", ".join(
                f"{ENTITY_BY_TAG[tag]}={prob:.4f}"
                for tag, prob in sorted(
                    zip(ENTITY_TAGS, entity_prob.tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        )
        print("  Forced :", forced)
        print("  Reply  :", reply)


if __name__ == "__main__":
    main()
