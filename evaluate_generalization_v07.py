# evaluate_generalization_v07.py
#
# Held-out paraphrase generalization test for LLM_GPU v0.7.
# IMPORTANT: Prompts in this file are intentionally different from the
# training examples in conversation-ja.txt and instruction-ja.txt.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from chat import AI_PREFIX, USER_PREFIX, generate_reply
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.7-chat.pt"


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
    print(" LLM_GPU v0.7 Generalization Test")
    print("====================================")
    print("Device          :", device)
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Held-out cases  :", len(CASES))
    print()

    total_pass = 0
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

        passed, missing, conflicts = semantic_match(
            reply,
            case["required_all"],
            case["forbidden"],
        )
        total_pass += int(passed)

        intent = str(case["intent"])
        ok, n = per_intent.get(intent, (0, 0))
        per_intent[intent] = (ok + int(passed), n + 1)

        print(f"[G{idx:02d}] {intent:14s} 人: {prompt_text}")
        print(f"      AI: {reply}")
        print("      semantic=" + ("PASS" if passed else "MISS"))
        if missing:
            print("      missing : " + ", ".join(missing))
        if conflicts:
            print("      conflict: " + ", ".join(conflicts))

    count = len(CASES)
    print()
    print("Summary")
    print("-------")
    print(
        f"Generalization semantic rate: {total_pass}/{count} "
        f"({total_pass / count:.1%})"
    )
    print()
    print("Per-intent")
    print("----------")
    for intent in sorted(per_intent):
        ok, n = per_intent[intent]
        print(f"{intent:14s}: {ok}/{n} ({ok / n:.1%})")


if __name__ == "__main__":
    main()
