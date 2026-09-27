# evaluate_chat.py
#
# Lightweight deterministic evaluation for LLM_GPU v0.5 conversational behavior.
# This is not a general intelligence benchmark. It checks whether the small
# model learned basic reply formatting, short-answer behavior, and a few
# held-out conversational intents.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from chat import AI_PREFIX, USER_PREFIX, generate_reply
from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_MODEL = "model/model-gpu-v0.5-chat.pt"

TEST_CASES: List[Dict[str, object]] = [
    {"prompt": "こんにちは、元気ですか。", "keywords": ["こんにちは", "元気"]},
    {"prompt": "日本の首都を教えてください。", "keywords": ["東京"]},
    {"prompt": "GPUは何をするものですか。", "keywords": ["計算", "並列", "GPU"]},
    {"prompt": "わからないので、もう一度説明して。", "keywords": ["説明", "確認"]},
    {"prompt": "今日は疲れました。", "keywords": ["休", "お疲れ"]},
    {"prompt": "プログラムでエラーが出ました。", "keywords": ["エラー", "確認"]},
    {"prompt": "研究結果を比べたいです。", "keywords": ["比較", "条件", "指標"]},
    {"prompt": "短く答えてください。", "keywords": ["はい", "短"]},
    {"prompt": "話題を変えましょう。", "keywords": ["話題", "どうぞ"]},
    {"prompt": "今日はここまでにします。", "keywords": ["お疲れ", "また"]},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate basic LLM_GPU v0.5 chat behavior."
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    return parser.parse_args()


def repetition_ratio(text: str) -> float:
    if len(text) < 4:
        return 0.0
    bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
    return 1.0 - len(set(bigrams)) / max(1, len(bigrams))


def main() -> None:
    args = parse_args()

    if not Path(args.tokenizer).exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )

    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    print()
    print("====================================")
    print(" LLM_GPU v0.5 Chat Evaluation")
    print("====================================")
    print("Device          :", device)
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Cases           :", len(TEST_CASES))
    print()

    keyword_hits = 0
    nonempty = 0
    sane_repetition = 0
    total_chars = 0

    # Greedy decoding (temperature=0) makes runs reproducible.
    for index, case in enumerate(TEST_CASES, start=1):
        user_text = str(case["prompt"])
        keywords = [str(k) for k in case["keywords"]]
        prompt = f"{USER_PREFIX}{user_text}\n{AI_PREFIX}"

        reply, _ = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.12,
        )

        hit = any(keyword in reply for keyword in keywords)
        rep = repetition_ratio(reply)
        is_nonempty = bool(reply.strip())
        rep_ok = rep < 0.60

        keyword_hits += int(hit)
        nonempty += int(is_nonempty)
        sane_repetition += int(rep_ok)
        total_chars += len(reply)

        print(f"[{index:02d}] 人: {user_text}")
        print(f"     AI: {reply}")
        print(
            "     keyword=" + ("PASS" if hit else "MISS")
            + f"  repetition={rep:.3f}"
        )

    count = len(TEST_CASES)
    print()
    print("Summary")
    print("-------")
    print(f"Keyword hit rate : {keyword_hits}/{count} ({keyword_hits / count:.1%})")
    print(f"Non-empty replies: {nonempty}/{count} ({nonempty / count:.1%})")
    print(
        f"Repetition sanity: {sane_repetition}/{count} "
        f"({sane_repetition / count:.1%})"
    )
    print(f"Mean reply length: {total_chars / count:.1f} characters")
    print()
    print(
        "Interpretation: keyword hit rate is only a small regression metric. "
        "Read the generated replies as well; this tiny model is not expected "
        "to provide broad factual or reasoning coverage."
    )


if __name__ == "__main__":
    main()
