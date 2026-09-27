# evaluate_generalization_v07.py
#
# Held-out paraphrase generalization test for LLM_GPU v0.8.
# IMPORTANT: Prompts in this file are intentionally different from the
# training examples in conversation-ja.txt and instruction-ja.txt.

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from chat import AI_PREFIX, USER_PREFIX, generate_reply
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat.pt"


CASES: List[Dict[str, object]] = [
    # GPU / CPU: held-out wording
    {"intent": "gpu", "prompt": "大量の同時計算に向いているのはGPUですか。何が得意ですか。",
     "required_all": [["GPU"], ["並列", "同時", "大量"]], "forbidden": ["汎用的な命令実行を得意"]},
    {"intent": "gpu", "prompt": "GPUの強みを一文で述べてください。",
     "required_all": [["GPU"], ["並列", "大量"]], "forbidden": ["コンピュータ全体の汎用処理"]},
    {"intent": "gpu", "prompt": "機械学習でGPUがよく使われる理由は何ですか。",
     "required_all": [["GPU"], ["並列", "大量", "計算"]], "forbidden": ["言語モデルです"]},
    {"intent": "cpu", "prompt": "CPUの得意分野を一文で説明してください。",
     "required_all": [["CPU"], ["汎用", "命令", "制御"]], "forbidden": ["大量の並列計算を得意"]},
    {"intent": "cpu", "prompt": "コンピュータの中心で多様な命令を処理する装置は何ですか。",
     "required_all": [["CPU"], ["命令", "汎用", "処理"]], "forbidden": ["GPUは"]},
    {"intent": "cpu", "prompt": "CPUの特徴をGPUと混同しないように説明してください。",
     "required_all": [["CPU"], ["汎用", "命令", "制御"]], "forbidden": ["CPUは大量の並列"]},

    # Other technical concepts
    {"intent": "llm", "prompt": "文章を学習して文章を生成するモデルは何ですか。",
     "required_all": [["LLM", "言語モデル"], ["文章", "言語"]], "forbidden": ["演算装置"]},
    {"intent": "transformer", "prompt": "Attentionを中心に使う代表的な構造は何ですか。",
     "required_all": [["Transformer"], ["Attention"]], "forbidden": ["プログラミング言語"]},
    {"intent": "cuda", "prompt": "NVIDIA製GPUで汎用計算を行う仕組みを何と呼びますか。",
     "required_all": [["CUDA"], ["GPU"]], "forbidden": ["言語モデル"]},
    {"intent": "python", "prompt": "読みやすさで知られる汎用言語を一つ挙げてください。",
     "required_all": [["Python"], ["言語"]], "forbidden": ["演算装置"]},

    # Conversational control
    {"intent": "short", "prompt": "回答は一言か二言にしてください。",
     "required_all": [["短", "簡潔", "要点", "はい"]], "forbidden": []},
    {"intent": "short", "prompt": "長い説明はいりません。",
     "required_all": [["短", "簡潔", "要点", "わかりました", "はい"]], "forbidden": []},
    {"intent": "topic", "prompt": "別件について話したいです。",
     "required_all": [["話題", "どうぞ", "何について", "いいですよ"]], "forbidden": ["休憩してください"]},
    {"intent": "topic", "prompt": "今のテーマはやめて別のテーマへ移りたいです。",
     "required_all": [["話題", "テーマ", "どうぞ", "いいですよ"]], "forbidden": ["お疲れさまでした"]},
    {"intent": "repeat", "prompt": "今の説明では理解できませんでした。",
     "required_all": [["説明", "分かり", "簡単", "もちろん"]], "forbidden": []},
    {"intent": "repeat", "prompt": "別の言い方でもう一度お願いします。",
     "required_all": [["説明", "言い方", "もちろん", "簡単"]], "forbidden": []},
    {"intent": "end", "prompt": "この会話はここで終わりにしましょう。",
     "required_all": [["お疲れ", "また", "終"]], "forbidden": ["新しい話題"]},
    {"intent": "end", "prompt": "続きは次回にします。",
     "required_all": [["また", "続き", "お疲れ"]], "forbidden": []},

    # Debug / research
    {"intent": "error", "prompt": "コード実行に失敗しました。最初に何を見ますか。",
     "required_all": [["エラー", "メッセージ", "コード", "原因"]], "forbidden": []},
    {"intent": "error", "prompt": "不具合の原因を切り分けたいです。",
     "required_all": [["エラー", "原因", "確認", "条件"]], "forbidden": []},
    {"intent": "compare", "prompt": "二つの実験を公平に比べるにはどうしますか。",
     "required_all": [["比較", "比べ"], ["条件", "指標"]], "forbidden": []},
    {"intent": "compare", "prompt": "モデルAとBの差を検証したいです。",
     "required_all": [["比較", "条件", "指標", "同じ"]], "forbidden": []},

    # General facts / simple conversation
    {"intent": "capital", "prompt": "日本で政府の中心となる都市はどこですか。",
     "required_all": [["東京"]], "forbidden": []},
    {"intent": "greeting", "prompt": "やあ、調子はどうですか。",
     "required_all": [["こんにちは", "元気", "どう", "話"]], "forbidden": []},
    {"intent": "fatigue", "prompt": "少し休みたいくらい疲れています。",
     "required_all": [["休", "お疲れ"]], "forbidden": []},
    {"intent": "thanks", "prompt": "助かりました、感謝します。",
     "required_all": [["どういたしまして", "役に立", "また"]], "forbidden": []},

    # Contrast questions
    {"intent": "gpu_cpu", "prompt": "並列計算向きなのはCPUとGPUのどちらですか。",
     "required_all": [["GPU"], ["並列"]], "forbidden": ["CPUは大量の並列"]},
    {"intent": "gpu_cpu", "prompt": "汎用処理を主に担当するのはCPUとGPUのどちらですか。",
     "required_all": [["CPU"], ["汎用"]], "forbidden": ["GPUは汎用処理"]},
    {"intent": "cuda_gpu", "prompt": "CUDAはハードウェアそのものですか。",
     "required_all": [["CUDA"], ["GPU"], ["技術", "仕組み", "いいえ"]], "forbidden": ["CUDAは演算装置"]},
    {"intent": "llm_transformer", "prompt": "LLMとTransformerは同じ意味ですか。",
     "required_all": [["LLM"], ["Transformer"], ["構造", "モデル", "いいえ"]], "forbidden": []},
]



ENTITY_TERMS = {
    "gpu": ["GPU"],
    "cpu": ["CPU"],
    "llm": ["LLM", "言語モデル"],
    "transformer": ["Transformer"],
    "cuda": ["CUDA"],
    "python": ["Python"],
    "capital": ["東京"],
    "gpu_cpu": ["GPU", "CPU"],
    "cuda_gpu": ["CUDA", "GPU"],
    "llm_transformer": ["LLM", "Transformer"],
}


def _group_is_prompt_entity(
    group: Sequence[str],
    prompt: str,
    intent: str,
) -> bool:
    expected = set(ENTITY_TERMS.get(intent, []))
    if not expected:
        return False

    entity_hits = set(group) & expected
    if not entity_hits:
        return False

    return any(term in prompt for term in entity_hits)


def split_semantic_requirements(
    prompt: str,
    intent: str,
    required_all: Sequence[Sequence[str]],
) -> Tuple[List[Sequence[str]], List[Sequence[str]]]:
    content_groups = []
    entity_groups = []

    for group in required_all:
        if _group_is_prompt_entity(group, prompt, intent):
            entity_groups.append(group)
        else:
            content_groups.append(group)

    return content_groups, entity_groups


def entity_explicitness(
    text: str,
    entity_groups: Sequence[Sequence[str]],
) -> Tuple[bool, List[str]]:
    if not entity_groups:
        return True, []

    missing = [
        "/".join(group)
        for group in entity_groups
        if not any(term in text for term in group)
    ]
    return not missing, missing


def fluency_check(text: str) -> Tuple[bool, List[str]]:
    """Conservative deterministic surface-quality diagnostics."""
    issues = []
    stripped = text.strip()

    if not stripped:
        issues.append("empty")
        return False, issues

    if "�" in stripped:
        issues.append("replacement-char")

    if re.search(r"[A-Za-z_]{12,}", stripped):
        issues.append("long-ascii-fragment")

    if re.search(r"_[A-Za-z]", stripped):
        issues.append("underscore-fragment")

    if re.search(r"(.{2,10})\1", stripped):
        issues.append("repeated-fragment")

    chunks = re.findall(r"[一-龯ぁ-んァ-ヶA-Za-z0-9]+", stripped)
    counts = {}
    for chunk in chunks:
        if len(chunk) >= 2:
            counts[chunk] = counts.get(chunk, 0) + 1
    if counts and max(counts.values()) >= 4:
        issues.append("excessive-repetition")

    return not issues, issues


def dimension_match(
    text: str,
    prompt: str,
    intent: str,
    required_all: Sequence[Sequence[str]],
    forbidden: Sequence[str],
):
    content_groups, entity_groups = split_semantic_requirements(
        prompt, intent, required_all
    )

    missing_content = [
        "/".join(group)
        for group in content_groups
        if not any(term in text for term in group)
    ]
    conflicts = [term for term in forbidden if term in text]
    semantic_ok = not missing_content and not conflicts

    entity_ok, missing_entity = entity_explicitness(
        text, entity_groups
    )
    fluent_ok, fluency_issues = fluency_check(text)

    strict_ok = semantic_ok and entity_ok and fluent_ok
    return {
        "semantic_ok": semantic_ok,
        "entity_ok": entity_ok,
        "fluent_ok": fluent_ok,
        "strict_ok": strict_ok,
        "missing_content": missing_content,
        "missing_entity": missing_entity,
        "conflicts": conflicts,
        "fluency_issues": fluency_issues,
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out v0.7 generalization evaluation.")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


def semantic_match(
    text: str,
    required_all: Sequence[Sequence[str]],
    forbidden: Sequence[str],
) -> Tuple[bool, List[str], List[str]]:
    missing = [
        "/".join(group)
        for group in required_all
        if not any(term in text for term in group)
    ]
    conflicts = [term for term in forbidden if term in text]
    return (not missing and not conflicts), missing, conflicts


def main() -> None:
    args = parse_args()
    if not Path(args.tokenizer).exists():
        raise FileNotFoundError(args.tokenizer)
    if not Path(args.model).exists():
        raise FileNotFoundError(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)

    print()
    print("====================================")
    print(" LLM_GPU v0.8 Generalization Test")
    print("====================================")
    print("Device          :", device)
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Held-out cases  :", len(CASES))
    print()

    legacy_pass = 0
    semantic_pass = 0
    entity_pass = 0
    fluent_pass = 0
    strict_pass = 0
    per_intent = {}

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        prompt = f"{USER_PREFIX}{prompt_text}\n{AI_PREFIX}"
        reply, _ = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.05,
        )

        legacy_ok, legacy_missing, legacy_conflicts = semantic_match(
            reply,
            case["required_all"],
            case["forbidden"],
        )

        intent = str(case["intent"])
        dims = dimension_match(
            reply,
            prompt_text,
            intent,
            case["required_all"],
            case["forbidden"],
        )

        legacy_pass += int(legacy_ok)
        semantic_pass += int(dims["semantic_ok"])
        entity_pass += int(dims["entity_ok"])
        fluent_pass += int(dims["fluent_ok"])
        strict_pass += int(dims["strict_ok"])

        stats = per_intent.get(
            intent,
            {"semantic": 0, "strict": 0, "n": 0},
        )
        stats["semantic"] += int(dims["semantic_ok"])
        stats["strict"] += int(dims["strict_ok"])
        stats["n"] += 1
        per_intent[intent] = stats

        print(f"[G{idx:02d}] {intent:14s} 人: {prompt_text}")
        print(f"      AI: {reply}")
        print(
            "      semantic-content="
            + ("PASS" if dims["semantic_ok"] else "MISS")
            + " | entity="
            + ("PASS" if dims["entity_ok"] else "MISS")
            + " | fluency="
            + ("PASS" if dims["fluent_ok"] else "MISS")
            + " | strict="
            + ("PASS" if dims["strict_ok"] else "MISS")
        )
        if dims["missing_content"]:
            print("      missing-content: " + ", ".join(dims["missing_content"]))
        if dims["missing_entity"]:
            print("      missing-entity : " + ", ".join(dims["missing_entity"]))
        if dims["conflicts"]:
            print("      conflict       : " + ", ".join(dims["conflicts"]))
        if dims["fluency_issues"]:
            print("      fluency-issue  : " + ", ".join(dims["fluency_issues"]))
        if legacy_ok != dims["strict_ok"]:
            print(
                "      legacy-rule    : "
                + ("PASS" if legacy_ok else "MISS")
            )

    count = len(CASES)
    print()
    print("Summary")
    print("-------")
    print(
        f"Semantic-content rate : {semantic_pass}/{count} "
        f"({semantic_pass / count:.1%})"
    )
    print(
        f"Entity-explicit rate  : {entity_pass}/{count} "
        f"({entity_pass / count:.1%})"
    )
    print(
        f"Fluency rate          : {fluent_pass}/{count} "
        f"({fluent_pass / count:.1%})"
    )
    print(
        f"Strict composite rate : {strict_pass}/{count} "
        f"({strict_pass / count:.1%})"
    )
    print(
        f"Legacy rule rate      : {legacy_pass}/{count} "
        f"({legacy_pass / count:.1%})"
    )
    print()
    print("Per-intent")
    print("----------")
    print("intent         semantic      strict")
    for intent in sorted(per_intent):
        stats = per_intent[intent]
        n = stats["n"]
        sem = stats["semantic"]
        strict = stats["strict"]
        print(
            f"{intent:14s}: "
            f"{sem}/{n} ({sem / n:.1%})  "
            f"{strict}/{n} ({strict / n:.1%})"
        )


if __name__ == "__main__":
    main()
