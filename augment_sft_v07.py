# augment_sft_v07.py
#
# Deterministic intent-aware SFT augmentation for LLM_GPU v0.7.
# No external LLM/API is required. The script expands semantic intents with
# structurally different Japanese prompt templates while keeping evaluation
# prompts separate.

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


Pair = Tuple[str, str]
LabeledPair = Tuple[str, str, Tuple[str, ...]]


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
    "relation_compare",
    "relation_distinction",
    "relation_usage",
    "property_parallel",
    "property_general",
    "property_language",
    "property_attention",
    "other",
]


def classify_tags(text: str) -> Tuple[str, ...]:
    upper = text.upper()
    tags = set()

    if "GPU" in upper:
        tags.add("tech_gpu")
    if "CPU" in upper:
        tags.add("tech_cpu")
    if "LLM" in upper:
        tags.add("tech_llm")
    if "TRANSFORMER" in upper:
        tags.add("tech_transformer")
    if "CUDA" in upper:
        tags.add("tech_cuda")
    if "PYTHON" in upper:
        tags.add("tech_python")

    rules = [
        ("control_short", ("短く", "簡潔", "要点", "一言")),
        ("control_topic", ("話題", "別の話", "違う話", "テーマ", "別件")),
        ("control_repeat", ("もう一度", "分かりやす", "説明", "言い方", "理解でき")),
        ("control_end", ("ここまで", "終わり", "終わります", "終了", "次回", "続きは")),
        ("debug_error", ("エラー", "動かない", "失敗", "不具合", "原因")),
        ("research_compare", ("比較", "比べ", "実験結果", "モデルA", "モデルB", "差を")),
        ("greeting", ("こんにちは", "おはよう", "こんばんは", "元気", "調子", "やあ")),
        ("fatigue", ("疲れ", "眠い", "休みたい", "休憩したい")),
        ("thanks", ("ありがとう", "感謝", "助かり")),
        ("capital", ("首都", "東京", "政府の中心")),
    ]
    for label, keys in rules:
        if any(key in text for key in keys):
            tags.add(label)

    if (
        ("GPU" in upper and "CPU" in upper)
        or ("LLM" in upper and "TRANSFORMER" in upper)
        or ("PYTHON" in upper and "CUDA" in upper)
    ):
        tags.add("relation_compare")

    if any(key in text for key in ("同じ", "違い", "混同", "区別", "どちら")):
        tags.add("relation_distinction")

    if any(key in text for key in ("何に使", "用途", "利用", "役割", "仕事")):
        tags.add("relation_usage")

    if any(key in text for key in ("並列", "同時", "大量", "高速化")):
        tags.add("property_parallel")

    if any(key in text for key in ("汎用", "命令", "制御", "逐次")):
        tags.add("property_general")

    if any(key in text for key in ("言語", "文章", "プログラミング")):
        tags.add("property_language")

    if "ATTENTION" in upper or "Attention" in text:
        tags.add("property_attention")

    if not tags:
        tags.add("other")

    return tuple(sorted(tags))


def classify_intent(text: str) -> str:
    """Backward-compatible primary label for diagnostics/splitting."""
    tags = classify_tags(text)
    priority = [
        "tech_gpu", "tech_cpu", "tech_llm", "tech_transformer",
        "tech_cuda", "tech_python", "control_short", "control_topic",
        "control_repeat", "control_end", "debug_error",
        "research_compare", "greeting", "fatigue", "thanks", "capital",
    ]
    for label in priority:
        if label in tags:
            return label
    return tags[0]


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






# v0.8 technical concept binding refinement.
# Shared prompt structures form minimal pairs: only the concept changes while
# the requested semantic role stays constant. Canonical answers always name
# the concept explicitly to strengthen entity <-> property binding.
TECHNICAL_CONCEPTS = {
    "GPU": {
        "label": "tech_gpu",
        "definition": "GPUは大量の計算を並列に処理するのが得意な演算装置です。",
        "role": "GPUの中心的な役割は大量の並列計算を効率よく処理することです。",
        "category": "GPUは並列計算を得意とする演算装置です。",
        "contrast": "GPUはCPUのような汎用制御ではなく、大量の並列計算を得意とします。",
    },
    "CPU": {
        "label": "tech_cpu",
        "definition": "CPUは幅広い命令を実行し、汎用処理や制御を担当する演算装置です。",
        "role": "CPUの中心的な役割は多様な命令を実行し、汎用処理を制御することです。",
        "category": "CPUは汎用処理と制御を担当する演算装置です。",
        "contrast": "CPUはGPUのような大量並列計算より、汎用処理や複雑な命令実行を得意とします。",
    },
    "LLM": {
        "label": "tech_llm",
        "definition": "LLMは大量の文章から言語パターンを学び、文章を扱う言語モデルです。",
        "role": "LLMの中心的な役割は言語を理解し、文章を生成することです。",
        "category": "LLMは言語を扱う言語モデルです。",
        "contrast": "LLMは演算装置ではなく、言語を扱うモデルです。",
    },
    "Transformer": {
        "label": "tech_transformer",
        "definition": "TransformerはAttentionを中心に文脈を処理するニューラルネットワーク構造です。",
        "role": "Transformerの中心的な役割はAttentionで情報間の関係を処理することです。",
        "category": "TransformerはAttentionを使うニューラルネットワーク構造です。",
        "contrast": "Transformerはプログラミング言語ではなく、Attentionを中心とするモデル構造です。",
    },
    "CUDA": {
        "label": "tech_cuda",
        "definition": "CUDAはNVIDIA GPUを汎用計算に利用するための技術です。",
        "role": "CUDAの中心的な役割はNVIDIA GPU上で汎用計算を実行できるようにすることです。",
        "category": "CUDAはNVIDIA GPU向けの計算技術です。",
        "contrast": "CUDAはGPUというハードウェアそのものではなく、NVIDIA GPUを利用する計算技術です。",
    },
    "Python": {
        "label": "tech_python",
        "definition": "Pythonは読みやすい文法を持つ汎用プログラミング言語です。",
        "role": "Pythonの中心的な役割はプログラムを記述し、さまざまな処理を実装することです。",
        "category": "Pythonは汎用プログラミング言語です。",
        "contrast": "PythonはGPUやCUDAのような演算装置やGPU技術ではなく、プログラミング言語です。",
    },
}


TECHNICAL_SHARED_TEMPLATES = [
    ("{name}の中心的な役割を説明してください。", "role"),
    ("{name}を簡潔に定義してください。", "definition"),
    ("{name}はどの種類の技術・モデル・装置ですか。", "category"),
    ("{name}の性質を一文で説明してください。", "definition"),
    ("{name}について、他の技術と混同しない説明をしてください。", "contrast"),
    ("{name}の要点を名称を含めて答えてください。", "definition"),
]


def technical_minimal_pairs() -> List[Tuple[str, str, str]]:
    rows = []
    for name, info in TECHNICAL_CONCEPTS.items():
        for template, answer_key in TECHNICAL_SHARED_TEMPLATES:
            rows.append((
                template.format(name=name),
                str(info[answer_key]),
                str(info["label"]),
            ))
    return rows


RELATION_AUGMENT_ROWS: Sequence[Tuple[str, str]] = [
    (
        "CPUとGPUの役割の違いを説明してください。",
        "CPUは汎用処理や制御を担当し、GPUは大量の並列計算を得意とします。",
    ),
    (
        "CPUとGPUは何が違いますか。",
        "CPUは多様な命令の汎用処理、GPUは大量の並列計算を得意とします。",
    ),
    (
        "並列処理ではCPUとGPUのどちらが向いていますか。",
        "大量の並列処理にはGPUが向いています。",
    ),
    (
        "汎用的な制御ではCPUとGPUのどちらが向いていますか。",
        "汎用的な制御や複雑な命令実行にはCPUが向いています。",
    ),
    (
        "LLMとTransformerの関係を教えてください。",
        "LLMは言語モデルで、TransformerはLLMの実装によく使われるニューラルネットワーク構造です。",
    ),
    (
        "LLMとTransformerは同じものですか。",
        "いいえ。LLMは言語モデルで、Transformerはモデル構造です。",
    ),
    (
        "CUDAとGPUの違いを説明してください。",
        "GPUは演算装置で、CUDAはNVIDIA GPUを汎用計算に使うための技術です。",
    ),
    (
        "CUDAはGPUそのものですか。",
        "いいえ。GPUはハードウェアで、CUDAはGPUを利用するための技術です。",
    ),
    (
        "PythonとCUDAの違いを説明してください。",
        "Pythonはプログラミング言語で、CUDAはNVIDIA GPU向けの計算技術です。",
    ),
    (
        "PythonとCUDAは同じ種類ですか。",
        "いいえ。Pythonは言語で、CUDAはGPU計算技術です。",
    ),
]


def augment_pairs(
    base_pairs: Sequence[Pair],
    variants_per_intent: int = 24,
) -> List[LabeledPair]:
    """Return deduplicated base + deterministic synthetic multi-label rows."""
    output: List[LabeledPair] = []
    seen = set()

    def add(prompt: str, answer: str, seed_label: str = "") -> None:
        key = (prompt.strip(), answer.strip())
        if not prompt.strip() or not answer.strip() or key in seen:
            return
        seen.add(key)
        tags = set(classify_tags(prompt))
        if seed_label:
            tags.add(seed_label)
        output.append((key[0], key[1], tuple(sorted(tags))))

    for prompt, answer in base_pairs:
        add(prompt, answer)

    for prompt, answer, label in technical_minimal_pairs():
        add(prompt, answer, label)

    for prompt, answer in RELATION_AUGMENT_ROWS:
        add(prompt, answer)

    for label, bank in AUGMENT_BANK.items():
        prompts = list(bank["prompts"])
        answers = list(bank["answers"])
        if not prompts or not answers:
            continue

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
    labels = sorted({label for _, _, tags in rows for label in tags})
    return labels
