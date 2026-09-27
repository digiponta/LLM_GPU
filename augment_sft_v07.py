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
        ("control_repeat", ("もう一度", "分かりやす", "言い換", "言い方", "理解でき", "別の表現")),
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
        or ("CUDA" in upper and "GPU" in upper)
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




# v0.8 reverse-definition binding.
# Prompts deliberately omit the target concept name. The model must infer the
# entity from its defining properties and produce the canonical concept name.
# These rows target the concepts that remained weak in the refined evaluator.
REVERSE_DEFINITION_ROWS: Sequence[Tuple[str, str, str]] = [
    # GPU / CPU reverse identification
    (
        "大量の同種計算を同時並行で処理するのが得意な演算装置は何ですか。",
        "GPUです。GPUは大量の並列計算を得意とする演算装置です。",
        "tech_gpu",
    ),
    (
        "多数の計算を並列に処理して機械学習を高速化する装置は何ですか。",
        "GPUです。GPUは多数の計算を並列に処理することを得意とします。",
        "tech_gpu",
    ),
    (
        "汎用処理や複雑な命令実行、制御を主に担当する演算装置は何ですか。",
        "CPUです。CPUは汎用処理や制御を担当する演算装置です。",
        "tech_cpu",
    ),
    (
        "コンピュータで多様な命令を順序立てて実行する中心的な装置は何ですか。",
        "CPUです。CPUは多様な命令を実行し、汎用処理を担当します。",
        "tech_cpu",
    ),

    # Transformer reverse identification
    (
        "Attentionを主要な仕組みとして文脈中の情報関係を扱うモデル構造は何ですか。",
        "Transformerです。TransformerはAttentionを中心に文脈を処理するモデル構造です。",
        "tech_transformer",
    ),
    (
        "Self-Attentionを使って系列中の要素同士の関係を処理する代表的な構造は何ですか。",
        "Transformerです。TransformerはAttentionを中心とするニューラルネットワーク構造です。",
        "tech_transformer",
    ),
    (
        "RNNの逐次処理に頼らずAttentionを中心に系列を扱う構造を何と呼びますか。",
        "Transformerです。TransformerはAttentionを利用して系列を処理するモデル構造です。",
        "tech_transformer",
    ),

    # CUDA reverse identification
    (
        "NVIDIAのGPUを一般的な計算処理に利用するための計算基盤は何ですか。",
        "CUDAです。CUDAはNVIDIA GPUを汎用計算に利用するための技術です。",
        "tech_cuda",
    ),
    (
        "NVIDIA製GPU上で汎用的なプログラムを実行するための技術は何ですか。",
        "CUDAです。CUDAはNVIDIA GPU向けの汎用計算技術です。",
        "tech_cuda",
    ),
    (
        "NVIDIA GPUをGPGPU用途で利用するための代表的なプラットフォームは何ですか。",
        "CUDAです。CUDAはNVIDIA GPUを汎用計算に使うための計算基盤です。",
        "tech_cuda",
    ),

    # Python reverse identification
    (
        "読みやすい文法と幅広い用途で知られる汎用プログラミング言語は何ですか。",
        "Pythonです。Pythonは読みやすい文法を持つ汎用プログラミング言語です。",
        "tech_python",
    ),
    (
        "簡潔で読みやすい記述が特徴の汎用プログラミング言語を一つ挙げてください。",
        "Pythonです。Pythonは幅広い用途で使われる汎用プログラミング言語です。",
        "tech_python",
    ),
    (
        "機械学習やデータ処理でも広く使われる読みやすい汎用言語は何ですか。",
        "Pythonです。Pythonは汎用プログラミング言語です。",
        "tech_python",
    ),

    # Hard-negative selection: same relation, opposite answer.
    (
        "大量の並列計算と汎用制御のうち、前者を得意とする装置はCPUとGPUのどちらですか。",
        "GPUです。GPUは大量の並列計算を得意とします。",
        "tech_gpu",
    ),
    (
        "大量の並列計算と汎用制御のうち、後者を主に担当する装置はCPUとGPUのどちらですか。",
        "CPUです。CPUは汎用処理や制御を担当します。",
        "tech_cpu",
    ),
]




# v0.8 pairwise hard-negative binding.
# Each pair uses the same question structure with opposite semantic properties.
# The model must choose the correct concept and bind it to the correct property.
PAIRWISE_HARD_NEGATIVE_ROWS: Sequence[Tuple[str, str, str]] = [
    # GPU <-> CPU
    (
        "CPUとGPUのうち、大量の同種計算を並列に処理する側を選び、理由も答えてください。",
        "GPUです。GPUは大量の同種計算を並列に処理することを得意とします。",
        "tech_gpu",
    ),
    (
        "CPUとGPUのうち、多様な命令実行や汎用制御を主に担当する側を選び、理由も答えてください。",
        "CPUです。CPUは多様な命令実行や汎用処理、制御を担当します。",
        "tech_cpu",
    ),
    (
        "並列計算の得意な装置としてCPUとGPUのどちらを選びますか。もう一方との違いも短く答えてください。",
        "GPUです。GPUは並列計算を得意とし、CPUは汎用処理や制御を得意とします。",
        "tech_gpu",
    ),
    (
        "汎用処理の中心となる装置としてCPUとGPUのどちらを選びますか。もう一方との違いも短く答えてください。",
        "CPUです。CPUは汎用処理や制御を担当し、GPUは大量の並列計算を得意とします。",
        "tech_cpu",
    ),

    # Transformer <-> CUDA
    (
        "TransformerとCUDAのうち、Attentionを中心に情報関係を処理するモデル構造はどちらですか。",
        "Transformerです。TransformerはAttentionを中心に情報関係を処理するモデル構造です。",
        "tech_transformer",
    ),
    (
        "TransformerとCUDAのうち、NVIDIA GPUを汎用計算に利用するための技術はどちらですか。",
        "CUDAです。CUDAはNVIDIA GPUを汎用計算に利用するための技術です。",
        "tech_cuda",
    ),
    (
        "Attentionという特徴に対応するのはTransformerとCUDAのどちらですか。理由も答えてください。",
        "Transformerです。AttentionはTransformerの中心的な仕組みです。",
        "tech_transformer",
    ),
    (
        "NVIDIA GPU向け計算基盤という特徴に対応するのはTransformerとCUDAのどちらですか。理由も答えてください。",
        "CUDAです。CUDAはNVIDIA GPU向けの汎用計算基盤です。",
        "tech_cuda",
    ),

    # CUDA <-> Python
    (
        "CUDAとPythonのうち、NVIDIA GPUで汎用計算を行うための技術はどちらですか。",
        "CUDAです。CUDAはNVIDIA GPUを汎用計算に利用するための技術です。",
        "tech_cuda",
    ),
    (
        "CUDAとPythonのうち、読みやすい文法を持つ汎用プログラミング言語はどちらですか。",
        "Pythonです。Pythonは読みやすい文法を持つ汎用プログラミング言語です。",
        "tech_python",
    ),
    (
        "GPU計算技術という分類に当てはまるのはCUDAとPythonのどちらですか。もう一方の分類も答えてください。",
        "CUDAです。CUDAはGPU計算技術で、Pythonはプログラミング言語です。",
        "tech_cuda",
    ),
    (
        "プログラミング言語という分類に当てはまるのはCUDAとPythonのどちらですか。もう一方の分類も答えてください。",
        "Pythonです。Pythonはプログラミング言語で、CUDAはGPU計算技術です。",
        "tech_python",
    ),
]


# v0.9 Targeted Boundary Training.
# Matched contrasts target the seven residual development failures without
# copying the fixed 30 benchmark prompts verbatim. Surface forms are kept
# deliberately similar while the decisive semantic cue changes.
TARGETED_BOUNDARY_ROWS: Sequence[Tuple[str, str, str]] = [
    # CPU <-> GPU: reverse identification and selection boundaries.
    (
        "多様な種類の命令を実行し、システム全体の制御を担う側はCPUとGPUのどちらですか。",
        "CPUです。CPUは多様な命令の実行や汎用処理、制御を担当します。",
        "tech_cpu",
    ),
    (
        "同じ種類の計算を大量に同時実行する側はCPUとGPUのどちらですか。",
        "GPUです。GPUは大量の同種計算を並列に処理することを得意とします。",
        "tech_gpu",
    ),
    (
        "幅広い処理を順序立てて実行する中心的な演算装置を答えてください。",
        "CPUです。CPUは幅広い命令を実行する汎用的な演算装置です。",
        "tech_cpu",
    ),
    (
        "多数の演算を同時に進める用途に適した演算装置を答えてください。",
        "GPUです。GPUは多数の演算を並列に処理することを得意とします。",
        "tech_gpu",
    ),

    # Transformer <-> CUDA <-> Python category boundaries.
    (
        "情報同士の関係をAttentionで処理するニューラルネットワーク方式を答えてください。",
        "Transformerです。TransformerはAttentionを中心に情報関係を処理するモデル構造です。",
        "tech_transformer",
    ),
    (
        "NVIDIA製GPUを計算資源として利用するためのソフトウェア技術を答えてください。",
        "CUDAです。CUDAはNVIDIA GPUを汎用計算に利用するための技術です。",
        "tech_cuda",
    ),
    (
        "読みやすい記述を特徴とする汎用プログラミング言語を答えてください。",
        "Pythonです。Pythonは読みやすい文法を持つ汎用プログラミング言語です。",
        "tech_python",
    ),
    (
        "Transformer・CUDA・Pythonのうち、Attentionを中心に系列を扱うものはどれですか。",
        "Transformerです。TransformerはAttentionを中心とするモデル構造です。",
        "tech_transformer",
    ),
    (
        "Transformer・CUDA・Pythonのうち、NVIDIA GPU向け計算技術はどれですか。",
        "CUDAです。CUDAはNVIDIA GPUを利用するための計算技術です。",
        "tech_cuda",
    ),
    (
        "Transformer・CUDA・Pythonのうち、汎用プログラミング言語はどれですか。",
        "Pythonです。Pythonは汎用プログラミング言語です。",
        "tech_python",
    ),
    (
        "Attentionを使う構造とGPU計算技術を区別してください。前者は何ですか。",
        "前者はTransformerです。TransformerはAttentionを使うモデル構造で、CUDAはGPU計算技術です。",
        "tech_transformer",
    ),
    (
        "GPU計算技術とプログラミング言語を区別してください。後者は何ですか。",
        "後者はPythonです。Pythonはプログラミング言語で、CUDAはGPU計算技術です。",
        "tech_python",
    ),

    # Conversational control matched boundaries:
    # short = response length, repeat = re-explain, topic = switch subject,
    # end = stop/continue later.
    (
        "内容はそのままで、返答だけ短めにしてください。",
        "はい。要点だけ短く答えます。",
        "control_short",
    ),
    (
        "内容はそのままで、もっと簡単な表現でもう一度説明してください。",
        "もちろんです。簡単な言葉に言い換えてもう一度説明します。",
        "control_repeat",
    ),
    (
        "この内容の説明は続けず、別の話へ切り替えましょう。",
        "いいですよ。別の話題に切り替えましょう。",
        "control_topic",
    ),
    (
        "この内容の続きは今は扱わず、また後日にしましょう。",
        "はい。また次回、続きから始めましょう。",
        "control_end",
    ),
    (
        "説明の長さだけを減らしてください。説明し直す必要はありません。",
        "はい。説明し直さず、要点だけ簡潔に答えます。",
        "control_short",
    ),
    (
        "長さを短くするのではなく、理解しやすい言葉に言い換えてください。",
        "もちろんです。分かりやすい言葉に言い換えて説明します。",
        "control_repeat",
    ),
    (
        "言い換えではなく、今とは別の話を始めたいです。",
        "はい。別の話題へ移りましょう。",
        "control_topic",
    ),
    (
        "別の話を始めるのではなく、今日はここで区切って続きは後日にします。",
        "お疲れさまでした。また次回、続きから始めましょう。",
        "control_end",
    ),
    (
        "返答を簡潔にしてください。話題は変更しません。",
        "はい。同じ話題のまま簡潔に答えます。",
        "control_short",
    ),
    (
        "話題は変更しません。分からなかったので別の言葉で説明してください。",
        "もちろんです。同じ話題を別の言葉で説明します。",
        "control_repeat",
    ),
    (
        "説明し直す必要はありません。別のテーマに移ってください。",
        "はい。別のテーマに移りましょう。",
        "control_topic",
    ),
    (
        "新しいテーマには移らず、この会話はここで終えてください。",
        "はい。ここで終わりにしましょう。また必要なときに続けましょう。",
        "control_end",
    ),
]


# v0.9 Targeted Boundary Training v2.
# This second boundary set focuses only on the five residual development
# failures after v1: CPU reverse lookup, Transformer category completion,
# short/repeat separation, and end/topic separation.
TARGETED_BOUNDARY_V2_ROWS: Sequence[Tuple[str, str, str]] = [
    # CPU reverse-identification: emphasize "center/core + diverse commands"
    # without copying the fixed benchmark wording.
    (
        "コンピュータ全体の中心で幅広い種類の命令を処理する演算装置を答えてください。",
        "CPUです。CPUは多様な命令を実行し、汎用処理や制御を担当する演算装置です。",
        "tech_cpu",
    ),
    (
        "OSやアプリからのさまざまな命令を順に処理する中心的な演算装置は何ですか。",
        "CPUです。CPUは幅広い命令を処理する汎用的な演算装置です。",
        "tech_cpu",
    ),
    (
        "多数の異なる命令を扱い、機器全体の制御にも使われる演算装置を一つ挙げてください。",
        "CPUです。CPUは多様な命令実行と汎用制御を担当します。",
        "tech_cpu",
    ),
    (
        "並列計算専用ではなく、汎用的な命令処理の中心になる装置は何ですか。",
        "CPUです。CPUは汎用処理や命令実行の中心となる演算装置です。",
        "tech_cpu",
    ),

    # Transformer: force complete entity + mechanism + category answers.
    (
        "Attentionを主要機構として使う代表的なニューラルネットワーク構造を答えてください。",
        "Transformerです。TransformerはAttentionを中心に情報を処理するニューラルネットワーク構造です。",
        "tech_transformer",
    ),
    (
        "Self-Attentionを中心に系列情報を扱うモデル構造の名称を答えてください。",
        "Transformerです。TransformerはAttentionを中心に系列を処理するモデル構造です。",
        "tech_transformer",
    ),
    (
        "Attentionが中心的な仕組みである代表的なモデルアーキテクチャは何ですか。",
        "Transformerです。TransformerはAttentionを使うニューラルネットワークのモデル構造です。",
        "tech_transformer",
    ),
    (
        "RNNとは異なりAttentionを中心に系列を扱うニューラルネットワーク方式は何ですか。",
        "Transformerです。TransformerはAttentionを中心とするニューラルネットワーク構造です。",
        "tech_transformer",
    ),

    # short <-> repeat: same topic, different requested operation.
    (
        "説明内容は変えず、返答の長さだけ短くしてください。",
        "はい。同じ内容のまま要点だけ簡潔に答えます。",
        "control_short",
    ),
    (
        "詳しい説明は不要なので、結論だけ簡潔にしてください。",
        "はい。結論だけ短く答えます。",
        "control_short",
    ),
    (
        "説明が理解できなかったので、内容を別の言葉で説明し直してください。",
        "もちろんです。分かりやすい言葉に言い換えてもう一度説明します。",
        "control_repeat",
    ),
    (
        "短くする必要はありません。理解しやすい表現でもう一度説明してください。",
        "はい。簡単な表現に言い換えてもう一度説明します。",
        "control_repeat",
    ),
    (
        "長さを減らすのではなく、分からなかった部分を言い換えて説明してください。",
        "もちろんです。内容を言い換えて分かりやすく説明します。",
        "control_repeat",
    ),
    (
        "説明し直しではなく、返答をもっと短くしてください。",
        "はい。説明し直さず、要点だけ短く答えます。",
        "control_short",
    ),

    # end <-> topic: distinguish "stop and continue later" from "switch now".
    (
        "今日はここで区切り、残りは次の機会に続けましょう。",
        "はい。また次回、続きから始めましょう。",
        "control_end",
    ),
    (
        "今はここまでにして、この続きは後日に回します。",
        "お疲れさまでした。また後日、続きから始めましょう。",
        "control_end",
    ),
    (
        "会話をいったん終えて、残りは次の機会にしましょう。",
        "はい。今回はここで終わりにして、また次回続けましょう。",
        "control_end",
    ),
    (
        "終了はせず、今すぐ別のテーマへ切り替えてください。",
        "はい。会話は続けたまま、別の話題へ移りましょう。",
        "control_topic",
    ),
    (
        "次回へ持ち越すのではなく、今この場で別の話を始めましょう。",
        "いいですよ。今すぐ別の話題に移りましょう。",
        "control_topic",
    ),
    (
        "今日は終わりにせず、現在の話題だけ変更したいです。",
        "はい。会話は続けて、別のテーマに切り替えましょう。",
        "control_topic",
    ),
]


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
    include_targeted_boundary: bool = False,
    include_targeted_boundary_v2: bool = False,
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

    for prompt, answer, label in REVERSE_DEFINITION_ROWS:
        add(prompt, answer, label)

    for prompt, answer, label in PAIRWISE_HARD_NEGATIVE_ROWS:
        add(prompt, answer, label)

    if include_targeted_boundary:
        for prompt, answer, label in TARGETED_BOUNDARY_ROWS:
            add(prompt, answer, label)

    if include_targeted_boundary_v2:
        for prompt, answer, label in TARGETED_BOUNDARY_V2_ROWS:
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
