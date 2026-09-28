# evaluate_heldout_generalization_v1001.py
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from evaluate_conversational_intent_repair_v01118 import (
    DEFAULT_TOKENIZER, DEFAULT_MODEL, DEFAULT_INTENT, DEFAULT_ROLE,
    DEFAULT_BINDING, DEFAULT_REPAIR,
    generate as generate_v01118,
)
from evaluate_transformer_continuation_category_repair_v01117 import generate as generate_v01117
from evaluate_post_entity_boundary_binding_v01115 import build_boundary_block_ids
from selective_intent_repair_v01110 import load_checkpoint as load_repair
from multi_concept_safe_binding_v0118 import load_checkpoint as load_binding
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

# Completely new prompts relative to the 30-case benchmark.
# Each required group is OR; all groups must be satisfied.
CASES = [
    # GPU
    ("gpu","映像処理のような大量の同型演算に向く装置は何ですか。",[["GPU"],["並列","大量","同時"]],[]),
    ("gpu","ニューラルネットの学習を高速化しやすい演算装置を説明してください。",[["GPU"],["並列","計算"]],[]),
    ("gpu","多数の要素を一度に計算する用途では何が有利ですか。",[["GPU"],["並列","大量"]],[]),
    ("gpu","行列演算をまとめて処理するのが得意なプロセッサは何ですか。",[["GPU"],["並列","計算"]],[]),
    ("gpu","画像生成AIでGPUが使われる主な理由を一文で答えてください。",[["GPU"],["並列","大量","計算"]],[]),
    # CPU
    ("cpu","OSの制御や一般的な処理を主に担う装置は何ですか。",[["CPU"],["汎用","制御","命令"]],[]),
    ("cpu","多種類の命令を順次扱うのが得意なプロセッサを説明してください。",[["CPU"],["命令","汎用","制御"]],[]),
    ("cpu","コンピュータ全体の制御役として中心になる演算装置は何ですか。",[["CPU"],["制御","汎用"]],[]),
    ("cpu","分岐や複雑な命令処理に向くのはCPUとGPUのどちらですか。",[["CPU"],["命令","汎用","制御"]],[]),
    ("cpu","一般用途のプログラム実行を担当する装置は何ですか。",[["CPU"],["汎用","命令","処理"]],[]),
    # LLM
    ("llm","自然言語を学習して文を作る大規模モデルを何と呼びますか。",[["LLM","言語モデル"],["言語","文章"]],[]),
    ("llm","文章の続きを予測しながら生成するモデルの代表例を答えてください。",[["LLM","言語モデル"],["文章","言語"]],[]),
    ("llm","大量のテキストから言語パターンを学ぶモデルは何ですか。",[["LLM","言語モデル"],["言語","文章"]],[]),
    ("llm","会話文を生成するために使われる大規模言語モデルを説明してください。",[["LLM","言語モデル"],["文章","言語"]],[]),
    ("llm","テキストを入力してテキストを返す生成モデルの種類を一つ挙げてください。",[["LLM","言語モデル"],["文章","言語"]],[]),
    # Transformer
    ("transformer","自己注意機構を中心に構成された代表的モデル構造は何ですか。",[["Transformer"],["Attention","注意"],["構造","モデル"]],[]),
    ("transformer","Self-Attentionを主要要素に持つニューラルネット構造を答えてください。",[["Transformer"],["Attention","Self-Attention"],["構造","モデル"]],[]),
    ("transformer","系列処理でAttentionを利用する代表的アーキテクチャは何ですか。",[["Transformer"],["Attention"],["構造","モデル"]],[]),
    ("transformer","LLMの基盤として広く使われるAttention型構造を説明してください。",[["Transformer"],["Attention"],["構造","モデル"]],[]),
    ("transformer","RNNではなくAttention主体で系列を処理する構造は何ですか。",[["Transformer"],["Attention"],["構造","モデル"]],[]),
    # CUDA
    ("cuda","NVIDIA GPU向けの汎用並列計算環境は何ですか。",[["CUDA"],["GPU"],["技術","仕組み","計算"]],[]),
    ("cuda","NVIDIAのGPU上で計算カーネルを実行する基盤を何と呼びますか。",[["CUDA"],["GPU"],["技術","計算"]],[]),
    ("cuda","GPUを一般計算に使うNVIDIAのプラットフォーム名を答えてください。",[["CUDA"],["GPU"],["技術","仕組み","計算"]],[]),
    ("cuda","NVIDIA製GPUのGPGPUプログラミングで使う技術は何ですか。",[["CUDA"],["GPU"],["技術","計算"]],[]),
    ("cuda","GPU計算用にNVIDIAが提供している仕組みを一つ挙げてください。",[["CUDA"],["GPU"],["技術","仕組み"]],[]),
    # Python
    ("python","初心者にも読みやすいことで知られるプログラミング言語は何ですか。",[["Python"],["言語"]],[]),
    ("python","機械学習で広く使われる高水準言語を一つ答えてください。",[["Python"],["言語"]],[]),
    ("python","インデントでブロックを表す代表的な汎用言語は何ですか。",[["Python"],["言語"]],[]),
    ("python","可読性を重視した文法を持つ言語として何が有名ですか。",[["Python"],["言語"]],[]),
    ("python","データ分析でもよく使われる汎用プログラミング言語を挙げてください。",[["Python"],["言語"]],[]),
    # short
    ("short","一言でお願いします。",[["短","簡潔","はい","要点"]],[]),
    ("short","できるだけ短く答えてください。",[["短","簡潔","はい","要点"]],[]),
    ("short","要点だけにしてください。",[["短","簡潔","はい","要点"]],[]),
    ("short","説明は最小限で構いません。",[["短","簡潔","はい","要点"]],[]),
    ("short","二文以内で答えてください。",[["短","簡潔","はい","要点"]],[]),
    # topic
    ("topic","別の話題に切り替えたいです。",[["話題","テーマ","どうぞ","いいですよ"]],[]),
    ("topic","この件は終えて別件へ移りましょう。",[["話題","別件","どうぞ","いいですよ"]],[]),
    ("topic","話題を変えてもいいですか。",[["話題","どうぞ","いいですよ"]],[]),
    ("topic","ほかのテーマについて話しましょう。",[["話題","テーマ","どうぞ","いいですよ"]],[]),
    ("topic","今とは違う話をしたいです。",[["話題","どうぞ","いいですよ"]],[]),
    # repeat
    ("repeat","もう少し分かりやすく説明し直してください。",[["説明","分かり","もちろん","言い換"]],[]),
    ("repeat","理解できなかったので別の表現でお願いします。",[["説明","分かり","もちろん","言い換"]],[]),
    ("repeat","同じ内容をやさしく言い換えてください。",[["説明","簡単","もちろん","言い換"]],[]),
    ("repeat","もう一度説明してもらえますか。",[["説明","もちろん","もう一度"]],[]),
    ("repeat","先ほどの説明を簡単にし直してください。",[["説明","簡単","分かり","もちろん"]],[]),
    # compare
    ("compare","二つの結果を同じ基準で比較するにはどうすればよいですか。",[["比較","比べ"],["条件","指標","基準","同じ"]],[]),
    ("compare","実験Aと実験Bを公正に比べたいです。",[["比較","比べ"],["条件","指標","基準","同じ"]],[]),
    ("compare","モデルの性能差を評価する際に揃えるべきものは何ですか。",[["比較","差","評価"],["条件","指標","基準","同じ"]],[]),
    ("compare","二つの手法の優劣を同条件で検証したいです。",[["比較","検証","比べ"],["条件","指標","同じ"]],[]),
    ("compare","比較実験で公平性を保つには何を統一しますか。",[["比較"],["条件","指標","基準","同じ"]],[]),
    # error
    ("error","プログラムが落ちました。最初に確認するものは何ですか。",[["エラー","原因","ログ","メッセージ","コード"]],[]),
    ("error","例外が出たときの切り分け手順を教えてください。",[["エラー","原因","確認","コード"]],[]),
    ("error","動作しない原因を調べるには何から見ればよいですか。",[["原因","確認","エラー","条件"]],[]),
    ("error","実行時エラーの調査で最初に見る情報は何ですか。",[["エラー","メッセージ","ログ","原因"]],[]),
    ("error","不具合を再現できた後、何を確認すべきですか。",[["原因","条件","確認","コード"]],[]),
    # end
    ("end","今日はここまでにします。",[["また","終","お疲れ"]],[]),
    ("end","会話を終了してください。",[["また","終","お疲れ"]],[]),
    ("end","続きは別の日にしましょう。",[["また","続き","お疲れ"]],[]),
    ("end","ここで話を終えたいです。",[["また","終","お疲れ"]],[]),
    ("end","今回はこれで終わりです。",[["また","終","お疲れ"]],[]),
]

CONVERSATIONAL = {"short","topic","repeat","compare"}


def score(text, required, forbidden):
    missing = ["/".join(group) for group in required if not any(term in text for term in group)]
    conflicts = [term for term in forbidden if term in text]
    return (not missing and not conflicts), missing, conflicts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--role-checkpoint", default=DEFAULT_ROLE)
    p.add_argument("--binding", default=DEFAULT_BINDING)
    p.add_argument("--repair", default=DEFAULT_REPAIR)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    intent_head, _, intent_labels = load_intent_head(args.intent_head, model, device)
    role_head, _, _ = load_role_checkpoint(args.role_checkpoint, intent_labels, device)
    binding, binding_ck = load_binding(args.binding, intent_labels, device)
    repair, _ = load_repair(args.repair, intent_labels, device)
    boundary_block_ids = build_boundary_block_ids(tok)

    print("="*62)
    print(" LLM_GPU v1.0.1 Held-out Generalization Test")
    print("="*62)
    print("Device:", device)
    print("Held-out cases:", len(CASES))
    print("Cases per intent: 5")
    print()

    base_pass = final_pass = 0
    repair_tp = repair_fp = repair_fn = 0
    per_intent = {}

    for idx,(intent,prompt,required,forbidden) in enumerate(CASES, start=1):
        base_reply, base_dbg = generate_v01117(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,prompt,boundary_block_ids
        )
        final_reply, final_dbg = generate_v01118(
            model,tok,intent_head,intent_labels,role_head,binding,binding_ck,repair,prompt,boundary_block_ids
        )

        b_ok,_,_ = score(base_reply,required,forbidden)
        f_ok,missing,conflicts = score(final_reply,required,forbidden)
        active = bool(final_dbg.get("conversation_active"))

        base_pass += int(b_ok)
        final_pass += int(f_ok)
        stats = per_intent.setdefault(intent, {"n":0,"base":0,"final":0,"repair":0})
        stats["n"] += 1; stats["base"] += int(b_ok); stats["final"] += int(f_ok); stats["repair"] += int(active)

        repair_needed = (intent in CONVERSATIONAL and not b_ok)
        if active and repair_needed: repair_tp += 1
        elif active and not repair_needed: repair_fp += 1
        elif repair_needed and not active: repair_fn += 1

        print(f"[H{idx:02d}] {intent:12s} 人: {prompt}")
        print("      baseline:", base_reply)
        print("      final   :", final_reply)
        print(
            f"      base={'PASS' if b_ok else 'MISS'} "
            f"final={'PASS' if f_ok else 'MISS'} "
            f"repair={'ON' if active else 'OFF'}"
        )
        if missing: print("      missing:", ", ".join(missing))
        if conflicts: print("      conflict:", ", ".join(conflicts))

    precision = repair_tp / (repair_tp + repair_fp) if (repair_tp + repair_fp) else 1.0
    recall = repair_tp / (repair_tp + repair_fn) if (repair_tp + repair_fn) else 1.0

    print()
    print("Summary")
    print("-------")
    print(f"Baseline semantic rate : {base_pass}/{len(CASES)} ({base_pass/len(CASES):.1%})")
    print(f"Final semantic rate    : {final_pass}/{len(CASES)} ({final_pass/len(CASES):.1%})")
    print(f"Repair precision       : {precision:.1%}  TP={repair_tp} FP={repair_fp}")
    print(f"Repair recall          : {recall:.1%}  TP={repair_tp} FN={repair_fn}")
    print(f"False activation rate  : {repair_fp}/{len(CASES)} ({repair_fp/len(CASES):.1%})")
    print()
    print("Per-intent")
    print("----------")
    for intent in sorted(per_intent):
        s=per_intent[intent]
        print(
            f"{intent:12s} base={s['base']}/{s['n']} "
            f"final={s['final']}/{s['n']} repairs={s['repair']}"
        )


if __name__ == "__main__":
    main()
