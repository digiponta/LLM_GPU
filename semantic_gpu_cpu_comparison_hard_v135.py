# semantic_gpu_cpu_comparison_hard_v135.py
# Symmetric hard-contrast comparison axis for GPU vs CPU.
# Same topic and nearly identical syntax; only the preferred device/criterion flips.
# 10 paired topics = 20 prompts.

PAIRS = [
    (
        "大量の同型演算を高スループットで処理したい場合、CPUとGPUのどちらを選ぶべきですか。",
        "複雑な条件分岐を低遅延で処理したい場合、CPUとGPUのどちらを選ぶべきですか。",
    ),
    (
        "多数の独立した数値演算を同時に実行したい場合、CPUとGPUのどちらが適していますか。",
        "逐次的で依存関係の強い命令列を実行したい場合、CPUとGPUのどちらが適していますか。",
    ),
    (
        "大規模な行列積を高速化したい場合、CPUとGPUのどちらを使うのが一般的ですか。",
        "OS制御や割り込み処理を担当させたい場合、CPUとGPUのどちらを使うのが一般的ですか。",
    ),
    (
        "多数の画素へ同じ処理を一斉に適用したい場合、CPUとGPUのどちらが向いていますか。",
        "多種類の命令を柔軟に切り替えながら実行したい場合、CPUとGPUのどちらが向いていますか。",
    ),
    (
        "レイテンシより演算スループットを重視する場合、CPUとGPUのどちらを選びますか。",
        "スループットより単一スレッドの応答性を重視する場合、CPUとGPUのどちらを選びますか。",
    ),
    (
        "多数の軽量な演算器を使う方が有利な処理では、CPUとGPUのどちらを選ぶべきですか。",
        "少数の高機能コアを使う方が有利な処理では、CPUとGPUのどちらを選ぶべきですか。",
    ),
    (
        "データ並列性が高いワークロードでは、CPUとGPUのどちらが適していますか。",
        "分岐と制御依存が多いワークロードでは、CPUとGPUのどちらが適していますか。",
    ),
    (
        "ニューラルネットのテンソル計算を大量に行う場合、CPUとGPUのどちらを使うのが一般的ですか。",
        "一般的なアプリケーション命令を幅広く実行する場合、CPUとGPUのどちらを使うのが一般的ですか。",
    ),
    (
        "同じ種類の演算を多数のデータへ適用する場合、CPUとGPUのどちらが向いていますか。",
        "異なる種類の処理を順番に判断しながら実行する場合、CPUとGPUのどちらが向いていますか。",
    ),
    (
        "大規模な並列数値計算を重視する場合、CPUとGPUのどちらを選びますか。",
        "一般用途の制御と逐次命令処理を重視する場合、CPUとGPUのどちらを選びますか。",
    ),
]

DATASET = []
for pair_id, (gpu_prompt, cpu_prompt) in enumerate(PAIRS, start=1):
    DATASET.append(("gpu", "comparison_hard", pair_id, gpu_prompt))
    DATASET.append(("cpu", "comparison_hard", pair_id, cpu_prompt))

assert len(DATASET) == 20
