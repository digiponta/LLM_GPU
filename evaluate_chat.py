# evaluate_chat.py
#
# Deterministic semantic-aware regression evaluation for LLM_GPU v0.8.
# PASS requires all required concept groups and rejects forbidden/conflicting
# concepts. This avoids false positives such as a GPU answer that actually
# describes a CPU.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from chat import AI_PREFIX, USER_PREFIX, generate_reply
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat.pt"

# required_all is a list of synonym groups.
# At least one phrase from EVERY group must appear.
# forbidden fails the case even when required terms are present.
TEST_CASES: List[Dict[str, object]] = [
    {
        "prompt": "こんにちは、元気ですか。",
        "required_all": [["こんにちは", "元気", "話しましょう"]],
        "forbidden": [],
    },
    {
        "prompt": "日本の首都を教えてください。",
        "required_all": [["東京"]],
        "forbidden": [],
    },
    {
        "prompt": "GPUは何をするものですか。",
        "required_all": [["GPU"], ["並列", "多数の計算", "大量の計算"]],
        "forbidden": ["汎用的な処理を担当", "命令実行"],
    },
    {
        "prompt": "わからないので、もう一度説明して。",
        "required_all": [["説明", "分かりにく", "もちろん", "もう一度"]],
        "forbidden": [],
    },
    {
        "prompt": "今日は疲れました。",
        "required_all": [["休", "お疲れ"]],
        "forbidden": [],
    },
    {
        "prompt": "プログラムでエラーが出ました。",
        "required_all": [["エラー", "原因", "確認"]],
        "forbidden": [],
    },
    {
        "prompt": "研究結果を比べたいです。",
        "required_all": [["比較", "比べ"], ["条件", "指標"]],
        "forbidden": [],
    },
    {
        "prompt": "短く答えてください。",
        "required_all": [["短", "簡潔", "要点"]],
        "forbidden": [],
    },
    {
        "prompt": "話題を変えましょう。",
        "required_all": [["話題", "新しい", "どうぞ"]],
        "forbidden": ["休憩してください"],
    },
    {
        "prompt": "今日はここまでにします。",
        "required_all": [["お疲れ", "また", "続き"]],
        "forbidden": [],
    },
]

TECHNICAL_CONTRAST_CASES: List[Dict[str, object]] = [
    {
        "prompt": "GPUの役割は何ですか。",
        "required_all": [["GPU"], ["並列", "多数の計算", "大量の計算"]],
        "forbidden": ["汎用的な処理を担当", "命令実行", "言語モデル"],
    },
    {
        "prompt": "CPUはどんな装置ですか。",
        "required_all": [["CPU", "演算装置"], ["汎用", "命令", "コンピュータ全体"]],
        "forbidden": ["並列計算を得意", "言語モデル"],
    },
    {
        "prompt": "LLMは何をするモデルですか。",
        "required_all": [["LLM", "言語モデル"], ["言語", "文章"]],
        "forbidden": ["演算装置", "GPUを汎用計算"],
    },
    {
        "prompt": "Transformerの特徴は何ですか。",
        "required_all": [["Transformer", "Attention"], ["Attention"]],
        "forbidden": ["プログラミング言語"],
    },
    {
        "prompt": "CUDAは何のために使いますか。",
        "required_all": [["CUDA", "NVIDIA"], ["GPU"], ["計算", "汎用計算"]],
        "forbidden": ["プログラミング言語です"],
    },
    {
        "prompt": "Pythonとは何ですか。",
        "required_all": [["Python", "プログラミング"], ["言語"]],
        "forbidden": ["演算装置", "NVIDIA GPU"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic-aware evaluation for LLM_GPU v0.8 chat."
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def repetition_ratio(text: str) -> float:
    if len(text) < 4:
        return 0.0
    bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
    return 1.0 - len(set(bigrams)) / max(1, len(bigrams))


def semantic_match(
    text: str,
    required_all: Sequence[Sequence[str]],
    forbidden: Sequence[str],
) -> Tuple[bool, List[str], List[str]]:
    missing_groups = []
    for group in required_all:
        if not any(term in text for term in group):
            missing_groups.append("/".join(group))

    forbidden_hits = [term for term in forbidden if term in text]
    passed = not missing_groups and not forbidden_hits
    return passed, missing_groups, forbidden_hits


def print_match_details(
    passed: bool,
    missing: Sequence[str],
    forbidden_hits: Sequence[str],
    indent: str,
) -> None:
    print(indent + "semantic=" + ("PASS" if passed else "MISS"))
    if missing:
        print(indent + "missing : " + ", ".join(missing))
    if forbidden_hits:
        print(indent + "conflict: " + ", ".join(forbidden_hits))


def generate_case(model, tokenizer, user_text, max_new_tokens):
    prompt = f"{USER_PREFIX}{user_text}\n{AI_PREFIX}"
    reply, _ = generate_reply(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=1,
        repetition_penalty=1.05,
    )
    return reply


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
    print(" LLM_GPU v0.8 Semantic Evaluation")
    print("====================================")
    print("Device          :", device)
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Cases           :", len(TEST_CASES))
    print()

    semantic_hits = 0
    nonempty = 0
    sane_repetition = 0
    total_chars = 0

    for index, case in enumerate(TEST_CASES, start=1):
        user_text = str(case["prompt"])
        required_all = case["required_all"]
        forbidden = case["forbidden"]

        reply = generate_case(
            model, tokenizer, user_text, args.max_new_tokens
        )
        passed, missing, forbidden_hits = semantic_match(
            reply, required_all, forbidden
        )

        rep = repetition_ratio(reply)
        is_nonempty = bool(reply.strip())
        rep_ok = rep < 0.60

        semantic_hits += int(passed)
        nonempty += int(is_nonempty)
        sane_repetition += int(rep_ok)
        total_chars += len(reply)

        print(f"[{index:02d}] 人: {user_text}")
        print(f"     AI: {reply}")
        print_match_details(
            passed, missing, forbidden_hits, indent="     "
        )
        print(f"     repetition={rep:.3f}")

    count = len(TEST_CASES)
    print()
    print("Summary")
    print("-------")
    print(
        f"Semantic pass rate: {semantic_hits}/{count} "
        f"({semantic_hits / count:.1%})"
    )
    print(
        f"Non-empty replies : {nonempty}/{count} "
        f"({nonempty / count:.1%})"
    )
    print(
        f"Repetition sanity : {sane_repetition}/{count} "
        f"({sane_repetition / count:.1%})"
    )
    print(f"Mean reply length : {total_chars / count:.1f} characters")
    print()
    print(
        "A case passes only when every required semantic group is present "
        "and no conflicting/forbidden phrase is detected."
    )

    print()
    print("Technical contrast")
    print("------------------")
    technical_hits = 0

    for index, case in enumerate(
        TECHNICAL_CONTRAST_CASES, start=1
    ):
        user_text = str(case["prompt"])
        reply = generate_case(
            model, tokenizer, user_text, args.max_new_tokens
        )
        passed, missing, forbidden_hits = semantic_match(
            reply,
            case["required_all"],
            case["forbidden"],
        )
        technical_hits += int(passed)

        print(f"[T{index:02d}] 人: {user_text}")
        print(f"      AI: {reply}")
        print_match_details(
            passed, missing, forbidden_hits, indent="      "
        )

    technical_count = len(TECHNICAL_CONTRAST_CASES)
    print()
    print(
        f"Technical semantic rate: {technical_hits}/{technical_count} "
        f"({technical_hits / technical_count:.1%})"
    )


if __name__ == "__main__":
    main()
