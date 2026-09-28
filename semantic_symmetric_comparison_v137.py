# semantic_symmetric_comparison_v137.py
# Six-class symmetric comparison dataset.
# Complete graph over 6 classes: 15 unordered class pairs.
# Two semantic topics per pair, with reciprocal prompts for both classes.
# Each class participates in 5 pairs x 2 topics = 10 prompts.
# Total = 60 prompts.

CLASSES = ["gpu","cpu","llm","transformer","cuda","python"]

PAIR_TOPICS = {
    ("gpu","cpu"): [
        ("大量の同型演算を高スループットで並列処理する演算装置",
         "複雑な分岐や逐次命令を柔軟に処理する中央演算装置"),
        ("多数のデータ要素へ同じ数値演算を同時適用するプロセッサ",
         "OS制御や一般アプリケーション命令を実行するプロセッサ"),
    ],
    ("gpu","llm"): [
        ("行列積やテンソル計算を高速に実行する演算ハードウェア",
         "文章を生成し自然言語で応答する大規模モデル"),
        ("多数の並列演算器を持つ計算装置",
         "大量の文章を学習して次トークンを予測するモデル"),
    ],
    ("gpu","transformer"): [
        ("大規模なデータ並列演算に使うプロセッサ",
         "Self-Attentionを中心にしたニューラルネット構造"),
        ("多数の演算を同時実行するハードウェア",
         "AttentionとFFNブロックを積み重ねるアーキテクチャ"),
    ],
    ("gpu","cuda"): [
        ("並列数値計算を実行する演算装置そのもの",
         "NVIDIA GPU向けの並列計算ソフトウェア基盤"),
        ("多数の演算コアを備えるハードウェア",
         "GPUカーネルを起動するNVIDIAのプログラミング基盤"),
    ],
    ("gpu","python"): [
        ("大量の並列演算を実行する演算ハードウェア",
         "読みやすい構文を持つ高水準プログラミング言語"),
        ("AI学習の数値計算を高速化するプロセッサ",
         "AI実験コードやスクリプトを書く汎用言語"),
    ],
    ("cpu","llm"): [
        ("一般プログラムやOSを実行する中央処理装置",
         "文章を生成し質問に回答する大規模言語モデル"),
        ("複雑な命令列や分岐を実行する汎用プロセッサ",
         "大量のテキストから言語パターンを学ぶモデル"),
    ],
    ("cpu","transformer"): [
        ("一般用途の命令を実行する中央演算装置",
         "Self-Attentionを使うニューラルネット構造"),
        ("低遅延の逐次処理を行う汎用プロセッサ",
         "Attentionブロックを積み重ねる系列モデル構造"),
    ],
    ("cpu","cuda"): [
        ("OSや一般アプリケーションを実行する中央処理装置",
         "NVIDIA GPU向けの並列計算ソフトウェア基盤"),
        ("複雑な命令制御を行う汎用プロセッサ",
         "GPUカーネルとランタイムを提供するNVIDIA技術"),
    ],
    ("cpu","python"): [
        ("一般的な命令を実行する中央演算装置",
         "コードを記述する高水準プログラミング言語"),
        ("OS制御や逐次処理を担当するプロセッサ",
         "自動化やデータ分析に使う汎用言語"),
    ],
    ("llm","transformer"): [
        ("自然言語を生成する学習済み大規模モデル",
         "その内部で使われるSelf-Attention中心のアーキテクチャ"),
        ("質問応答や文章生成を行うモデルそのもの",
         "AttentionとFFNからなるモデル構造"),
    ],
    ("llm","cuda"): [
        ("文章を生成する大規模言語モデル",
         "NVIDIA GPUで計算カーネルを実行するソフトウェア基盤"),
        ("大量の文章を学習する言語生成モデル",
         "GPU向け並列プログラミング環境"),
    ],
    ("llm","python"): [
        ("文章生成や質問応答を行う大規模モデル",
         "そのモデルを実装する際にも使えるプログラミング言語"),
        ("自然言語を処理する学習済みモデル",
         "コードやスクリプトを書く高水準言語"),
    ],
    ("transformer","cuda"): [
        ("Self-Attentionを中心とするニューラルネット構造",
         "NVIDIA GPU向けの並列計算ソフトウェア基盤"),
        ("AttentionとFFNを積み重ねるモデルアーキテクチャ",
         "GPUカーネル実行を提供するNVIDIA技術"),
    ],
    ("transformer","python"): [
        ("Attentionベースのニューラルネットアーキテクチャ",
         "コードを記述する高水準プログラミング言語"),
        ("系列データをSelf-Attentionで処理するモデル構造",
         "AI実験や自動化に使う汎用言語"),
    ],
    ("cuda","python"): [
        ("NVIDIA GPU向けの並列計算ソフトウェア基盤",
         "汎用的なコードを書く高水準プログラミング言語"),
        ("GPUカーネル実行やデバイス制御を行うNVIDIA技術",
         "NumPyやPyTorchと組み合わせて使うプログラミング言語"),
    ],
}

def reciprocal_prompt(target_desc, other_desc):
    return f"「{target_desc}」を指し、「{other_desc}」ではないものは何ですか。"

DATASET = []
PAIR_META = []
pair_id = 0

for (a,b), topics in PAIR_TOPICS.items():
    assert len(topics) == 2
    for topic_id,(a_desc,b_desc) in enumerate(topics, start=1):
        pair_id += 1
        pa = reciprocal_prompt(a_desc,b_desc)
        pb = reciprocal_prompt(b_desc,a_desc)
        DATASET.append((a,"comparison",pa))
        DATASET.append((b,"comparison",pb))
        PAIR_META.append((pair_id,topic_id,a,b,pa,pb))

assert len(DATASET) == 60
assert pair_id == 30
for c in CLASSES:
    assert sum(1 for x in DATASET if x[0] == c) == 10
