# augment_sft_v07.py
#
# Deterministic intent-aware SFT augmentation for LLM_GPU v0.7.
# No external LLM/API is required. The script expands semantic intents with
# structurally different Japanese prompt templates while keeping evaluation
# prompts separate.

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


Pair = Tuple[str, str]
LabeledPair = Tuple[str, str, str]


INTENT_LABELS = [
    "tech_gpu",
    "tech_cpu",
    "tech_llm",
    "tech_transformer",
    "tech_cuda",
    "tech_python",
    "control_short",
    "control_topic",
    "control_repeat",
    "control_end",
    "debug_error",
    "research_compare",
    "greeting",
    "fatigue",
    "thanks",
    "capital",
    "other",
]


def classify_intent(text: str) -> str:
    upper = text.upper()

    # Contrast prompts are assigned by their requested focus when obvious.
    if "GPU" in upper and "CPU" not in upper:
        return "tech_gpu"
    if "CPU" in upper and "GPU" not in upper:
        return "tech_cpu"
    if "LLM" in upper and "TRANSFORMER" not in upper:
        return "tech_llm"
    if "TRANSFORMER" in upper and "LLM" not in upper:
        return "tech_transformer"
    if "CUDA" in upper and "PYTHON" not in upper:
        return "tech_cuda"
    if "PYTHON" in upper and "CUDA" not in upper:
        return "tech_python"

    rules = [
        ("control_short", ("短く", "簡潔", "要点", "一言")),
        ("control_topic", ("話題", "別の話", "違う話", "テーマ")),
        ("control_repeat", ("もう一度", "分かりやす", "説明", "言い方")),
        ("control_end", ("ここまで", "終わり", "終わります", "次回")),
        ("debug_error", ("エラー", "動かない", "失敗", "不具合")),
        ("research_compare", ("比較", "比べ", "実験結果", "モデルA", "モデルB")),
        ("greeting", ("こんにちは", "おはよう", "こんばんは", "元気", "調子")),
        ("fatigue", ("疲れ", "眠い", "休みたい")),
        ("thanks", ("ありがとう", "感謝", "助かり")),
        ("capital", ("首都", "東京", "政府の中心")),
    ]
    for label, keys in rules:
        if any(key in text for key in keys):
            return label

    # Mixed contrast questions are useful but should not dominate one class.
    if "GPU" in upper and "CPU" in upper:
        return "tech_gpu"
    if "LLM" in upper and "TRANSFORMER" in upper:
        return "tech_llm"
    if "CUDA" in upper and "GPU" in upper:
        return "tech_cuda"
    if "PYTHON" in upper and "CUDA" in upper:
        return "tech_python"

    return "other"


AUGMENT_BANK: Dict[str, Dict[str, Sequence[str]]] = {
    "tech_gpu": {
        "prompts": [
            "GPUについて要点を教えてください。",
            "GPUの主な特徴は何ですか。",
            "GPUが得意な仕事を説明してください。",
            "GPUはどのような計算に向いていますか。",
            "GPUを短く定義してください。",
            "GPUの計算上の強みは何ですか。",
            "GPUが高速化に役立つのはどんな処理ですか。",
            "GPUの用途を計算の観点から説明してください。",
        ],
        "answers": [
            "GPUは大量の計算を並列に処理するのが得意な演算装置です。",
            "GPUは同種の計算を多数並列に実行する処理を得意とします。",
            "GPUの強みは大量の並列計算を効率よく処理できることです。",
        ],
    },
    "tech_cpu": {
        "prompts": [
            "CPUについて要点を教えてください。",
            "CPUの主な特徴は何ですか。",
            "CPUが得意な仕事を説明してください。",
            "CPUはどのような処理に向いていますか。",
            "CPUを短く定義してください。",
            "CPUの計算上の強みは何ですか。",
            "CPUが中心的に担当する処理は何ですか。",
            "CPUの用途を制御の観点から説明してください。",
        ],
        "answers": [
            "CPUは幅広い命令を実行し、汎用処理や制御を担当する演算装置です。",
            "CPUは複雑な命令実行や順序依存の強い汎用処理を得意とします。",
            "CPUの強みは多様な処理を柔軟に実行できることです。",
        ],
    },
    "tech_llm": {
        "prompts": [
            "LLMについて要点を教えてください。",
            "LLMの主な役割は何ですか。",
            "LLMが扱う情報は何ですか。",
            "LLMを短く定義してください。",
            "LLMは何を学習するモデルですか。",
        ],
        "answers": [
            "LLMは大量の文章から言語のパターンを学び、文章を理解・生成する言語モデルです。",
            "LLMは言語データを学習して文章を扱うモデルです。",
        ],
    },
    "tech_transformer": {
        "prompts": [
            "Transformerについて要点を教えてください。",
            "Transformerの中心的な仕組みは何ですか。",
            "Transformerを短く説明してください。",
            "Transformerは何を使って文脈を処理しますか。",
        ],
        "answers": [
            "TransformerはAttentionを中心に文脈の情報関係を処理するニューラルネットワーク構造です。",
            "Transformerの中心的な仕組みはAttentionです。",
        ],
    },
    "tech_cuda": {
        "prompts": [
            "CUDAについて要点を教えてください。",
            "CUDAは何に使う技術ですか。",
            "CUDAを短く説明してください。",
            "CUDAとGPUの関係を説明してください。",
        ],
        "answers": [
            "CUDAはNVIDIAのGPUを汎用計算に利用するための技術です。",
            "CUDAはNVIDIA GPU上で計算処理を実行するための計算基盤です。",
        ],
    },
    "tech_python": {
        "prompts": [
            "Pythonについて要点を教えてください。",
            "Pythonは何のためのものですか。",
            "Pythonを短く説明してください。",
            "Pythonはどの種類の言語ですか。",
        ],
        "answers": [
            "Pythonは読みやすい文法を持つ汎用プログラミング言語です。",
            "Pythonは幅広い用途で使われる汎用プログラミング言語です。",
        ],
    },
    "control_short": {
        "prompts": [
            "答えは簡潔にお願いします。",
            "要点だけお願いします。",
            "長くせず短めに答えてください。",
            "一言でまとめてください。",
            "簡単な返答にしてください。",
        ],
        "answers": [
            "はい。短く答えます。",
            "はい。要点に絞って答えます。",
        ],
    },
    "control_topic": {
        "prompts": [
            "別の話に移りたいです。",
            "今とは違う話をしましょう。",
            "新しいテーマに変えましょう。",
            "別件へ移ってください。",
            "話を切り替えたいです。",
        ],
        "answers": [
            "いいですよ。新しい話題をどうぞ。",
            "はい。別の話題に移りましょう。",
        ],
    },
    "control_repeat": {
        "prompts": [
            "別の表現で説明してください。",
            "理解できなかったのでもう一度お願いします。",
            "もっと簡単な言葉で説明してください。",
            "説明を言い換えてください。",
            "同じ内容をもう一度教えてください。",
        ],
        "answers": [
            "もちろんです。分かりやすく言い換えて説明します。",
            "はい。簡単な言葉でもう一度説明します。",
        ],
    },
    "control_end": {
        "prompts": [
            "今日はこれで終わります。",
            "この続きはまた今度にします。",
            "ここで会話を終了します。",
            "今日は終了にしましょう。",
            "次回また続けましょう。",
        ],
        "answers": [
            "お疲れさまでした。また続きから始めましょう。",
            "はい。また次回お話ししましょう。",
        ],
    },
    "debug_error": {
        "prompts": [
            "コードが失敗しました。何を確認しますか。",
            "不具合を調査する最初の手順は何ですか。",
            "実行時の問題を切り分けたいです。",
            "プログラムの原因調査はどこから始めますか。",
            "動作しないコードを診断したいです。",
        ],
        "answers": [
            "まずエラーメッセージと実行条件を確認して原因を絞り込みます。",
            "エラー内容と関連するコードを順に確認しましょう。",
        ],
    },
    "research_compare": {
        "prompts": [
            "二つの結果を公平に比較する方法は何ですか。",
            "実験AとBを比べるとき何をそろえますか。",
            "モデルの差を正しく評価したいです。",
            "比較実験では何を固定すべきですか。",
            "研究結果の差を検証する方法を教えてください。",
        ],
        "answers": [
            "同じ条件と同じ評価指標をそろえて比較します。",
            "比較する条件をそろえ、同一の指標で評価しましょう。",
        ],
    },
    "greeting": {
        "prompts": [
            "やあ、元気ですか。",
            "こんにちは、調子はどうですか。",
            "今日の調子はどうですか。",
        ],
        "answers": [
            "こんにちは。元気です。今日は何について話しましょうか。",
            "はい、元気です。何について話しましょうか。",
        ],
    },
    "fatigue": {
        "prompts": [
            "少し疲れて休みたいです。",
            "今日はかなり疲れています。",
            "休憩したい気分です。",
        ],
        "answers": [
            "お疲れさまです。無理をせず少し休憩してください。",
            "お疲れさまです。休めるなら少し休みましょう。",
        ],
    },
    "thanks": {
        "prompts": [
            "助かりました。ありがとう。",
            "ありがとうございます、助かりました。",
            "感謝します。",
        ],
        "answers": [
            "どういたしまして。",
            "どういたしまして。また必要なら聞いてください。",
        ],
    },
    "capital": {
        "prompts": [
            "日本の首都は何という都市ですか。",
            "日本の首都名を答えてください。",
            "日本の政治の中心となる首都はどこですか。",
        ],
        "answers": [
            "日本の首都は東京です。",
            "東京です。",
        ],
    },
}


def augment_pairs(
    base_pairs: Sequence[Pair],
    variants_per_intent: int = 24,
) -> List[LabeledPair]:
    """Return deduplicated base + deterministic synthetic paraphrase pairs."""
    output: List[LabeledPair] = []
    seen = set()

    def add(prompt: str, answer: str, label: str) -> None:
        key = (prompt.strip(), answer.strip())
        if not prompt.strip() or not answer.strip() or key in seen:
            return
        seen.add(key)
        output.append((key[0], key[1], label))

    for prompt, answer in base_pairs:
        add(prompt, answer, classify_intent(prompt))

    for label, bank in AUGMENT_BANK.items():
        prompts = list(bank["prompts"])
        answers = list(bank["answers"])
        if not prompts or not answers:
            continue

        # Cycle answers across prompt templates. Repeated passes add controlled
        # punctuation/politeness variants rather than copying the held-out set.
        count = max(len(prompts), int(variants_per_intent))
        suffixes = ["", "。", "お願いします。", "教えてください。"]
        for i in range(count):
            prompt = prompts[i % len(prompts)]
            if i >= len(prompts):
                suffix = suffixes[(i // len(prompts)) % len(suffixes)]
                if suffix and not prompt.endswith(suffix):
                    prompt = prompt.rstrip("。") + "。" + suffix
            answer = answers[i % len(answers)]
            add(prompt, answer, label)

    return output


def intent_vocabulary(rows: Iterable[LabeledPair]) -> List[str]:
    labels = sorted({label for _, _, label in rows})
    return labels
