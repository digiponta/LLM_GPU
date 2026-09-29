# chat_gate_benchmark_v159.py
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List, Tuple

import torch

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from chat import (
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    UNKNOWN_REPLY,
    build_prompt,
    generate_reply,
    evaluate_unknown_gate,
    semantic_consistency_check,
    calibrated_agreement_threshold,
)

BENCHMARK = [
    # known definitions
    dict(group="known_definition", prompt="CPUとは", expected="known", keywords=["CPU", "汎用"]),
    dict(group="known_definition", prompt="GPUとは", expected="known", keywords=["GPU", "並列"]),
    dict(group="known_definition", prompt="LLMとは", expected="known", keywords=["LLM", "言語モデル"]),
    dict(group="known_definition", prompt="CUDAとは", expected="known", keywords=["CUDA"]),
    dict(group="known_definition", prompt="AIとは", expected="known", keywords=["AI"]),

    # comparison / multi-focus
    dict(group="comparison", prompt="CPUとGPUの違いは", expected="known", keywords=["CPU", "GPU"]),
    dict(group="comparison", prompt="CPUとGPUを比較して", expected="known", keywords=["CPU", "GPU"]),
    dict(group="comparison", prompt="LLMとAIの違いは", expected="known", keywords=["LLM", "AI"]),

    # greeting / general
    dict(group="greeting", prompt="こんにちは", expected="known", keywords=["こんにちは"]),
    dict(group="greeting", prompt="おはよう", expected="known", keywords=["おはよう"]),
    dict(group="general", prompt="今日は元気ですか", expected="known", keywords=[]),

    # expected unknown / unsupported or deliberately obscure
    dict(group="unknown", prompt="ZXQ-91 HyperFluxとは", expected="unknown", keywords=[]),
    dict(group="unknown", prompt="NeuroCrystal Bus v7とは", expected="unknown", keywords=[]),
    dict(group="unknown", prompt="量子ねこプロトコルA17とは", expected="unknown", keywords=[]),
    dict(group="unknown", prompt="架空CPU PONTA-X9000の仕様は", expected="unknown", keywords=[]),
    dict(group="unknown", prompt="存在しないLLM KappaOmega-42とは", expected="unknown", keywords=[]),

    # history contamination: unrelated prior accepted turns should not pollute generation
    dict(
        group="history_contamination",
        prompt="GPUとは",
        expected="known",
        keywords=["GPU", "並列"],
        history=[("こんにちは", "こんにちは。今日は何について話しましょうか。"),
                 ("CPUとは", "CPUは汎用的な命令実行と制御処理を担当する中央処理装置です。")],
    ),
    dict(
        group="history_contamination",
        prompt="CPUとは",
        expected="known",
        keywords=["CPU", "汎用"],
        history=[("LLMとは", "LLMは大量の文章から言語のパターンを学ぶ言語モデルです。"),
                 ("こんにちは", "こんにちは。")],
    ),
    dict(
        group="history_contamination",
        prompt="CPUとGPUの違いは",
        expected="known",
        keywords=["CPU", "GPU"],
        history=[("CPUとは", "CPUは汎用処理を担当します。"),
                 ("GPUとは", "GPUは並列計算を得意とします。")],
    ),
    dict(
        group="history_contamination",
        prompt="こんにちは",
        expected="known",
        keywords=["こんにちは"],
        history=[("GPUとは", "GPUは並列計算を得意とします。"),
                 ("LLMとは", "LLMは言語モデルです。")],
    ),
]


def keyword_ok(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return bool(text.strip())
    low = text.lower()
    alias_map = {
        "ai": ("ai", "人工知能", "artificial intelligence"),
        "llm": ("llm", "大規模言語モデル", "large language model", "言語モデル"),
        "cpu": ("cpu", "中央処理装置"),
        "gpu": ("gpu", "画像処理装置"),
    }
    for keyword in keywords:
        aliases = alias_map.get(keyword.lower(), (keyword.lower(),))
        if not any(alias.lower() in low for alias in aliases):
            return False
    return True


def run_case(model, tokenizer, case, args, device):
    history: List[Tuple[str, str]] = list(case.get("history", []))
    prompt, selected_history = build_prompt(
        history=history,
        user_text=case["prompt"],
        history_turns=args["history_turns"],
    )

    results = [
        generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args["max_new_tokens"],
            temperature=0.0,
            top_k=args["top_k"],
            repetition_penalty=args["repetition_penalty"],
            seed=0,
        )
    ]
    for probe in range(args["probe_count"] - 1):
        results.append(
            generate_reply(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args["max_new_tokens"],
                temperature=args["probe_temperature"],
                top_k=args["probe_top_k"],
                repetition_penalty=args["repetition_penalty"],
                seed=1000 + probe,
            )
        )

    primary = results[0]

    qa_sim = 0.0
    prev_sim = -1.0
    intent = "general"
    slots = []
    slot_cov = 1.0

    (
        semantic_ok,
        qa_sim,
        prev_sim,
        semantic_reason,
        intent,
        slots,
        slot_cov,
    ) = semantic_consistency_check(
        model=model,
        tokenizer=tokenizer,
        current_question=case["prompt"],
        answer=primary.text,
        history=history,
        contamination_margin=args["history_contamination_margin"],
    )

    effective_agreement = calibrated_agreement_threshold(
        intent=intent,
        semantic_ok=semantic_ok,
        slot_coverage=slot_cov,
        base_threshold=args["min_agreement"],
    )

    accepted, confidence, agreement, reason = evaluate_unknown_gate(
        results,
        min_confidence=args["min_confidence"],
        min_agreement=effective_agreement,
    )

    if accepted and not semantic_ok:
        accepted = False
        reason = semantic_reason
    elif accepted:
        reason = semantic_reason

    reply = primary.text if accepted else UNKNOWN_REPLY

    if case["expected"] == "unknown":
        correct = not accepted
    else:
        correct = accepted and keyword_ok(primary.text, case["keywords"])

    return dict(
        group=case["group"],
        prompt=case["prompt"],
        expected=case["expected"],
        accepted=accepted,
        correct=correct,
        reply=reply,
        primary=primary.text,
        confidence=confidence,
        agreement=agreement,
        intent=intent,
        slots="|".join(slots),
        slot_coverage=slot_cov,
        qa_similarity=qa_sim,
        previous_similarity=prev_sim,
        context_turns=len(selected_history),
        agreement_threshold=effective_agreement,
        reason=reason,
    )


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", default="results/chat_gate_benchmark_v159")
    ns = ap.parse_args()

    cfg = dict(
        max_new_tokens=96,
        top_k=20,
        repetition_penalty=1.05,
        history_turns=3,
        probe_count=3,
        probe_temperature=0.30,
        probe_top_k=10,
        min_confidence=0.18,
        min_agreement=0.35,
        history_contamination_margin=0.05,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(ns.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(ns.model, device=device)
    model.eval()

    print("="*118)
    print(" LLM_GPU v1.5.9 Chat Gate Benchmark / Known-Unknown Evaluation")
    print("="*118)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Model           :", ns.model)
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Cases           :", len(BENCHMARK))
    print("Context policy  : v1.5.5 minimal")
    print("Fallback        :", UNKNOWN_REPLY)
    print()

    rows = []
    started = time.perf_counter()

    print("Per-case evaluation")
    print("-"*118)
    for i, case in enumerate(BENCHMARK, 1):
        row = run_case(model, tokenizer, case, cfg, device)
        rows.append(row)
        status = "PASS" if row["correct"] else "FAIL"
        gate = "KNOWN" if row["accepted"] else "UNKNOWN"
        print(
            f"{i:02d} {status:4s} {case['group']:<22s} gate={gate:<7s} "
            f"conf={row['confidence']:.3f} agr={row['agreement']:.3f} "
            f"th={row['agreement_threshold']:.2f} "
            f"ctx={row['context_turns']} :: {case['prompt']}"
        )
        print("   ->", row["reply"])
        if not row["correct"]:
            print("   reason:", row["reason"])

    elapsed = time.perf_counter() - started

    known = [r for r in rows if r["expected"] == "known"]
    unknown = [r for r in rows if r["expected"] == "unknown"]
    accepted = [r for r in rows if r["accepted"]]
    wrong_accepted = [r for r in accepted if not r["correct"]]
    false_reject = [r for r in known if not r["accepted"]]
    unknown_rejected = [r for r in unknown if not r["accepted"]]

    known_accuracy = pct(sum(r["correct"] for r in known), len(known))
    unknown_rejection = pct(len(unknown_rejected), len(unknown))
    false_rejection = pct(len(false_reject), len(known))
    coverage = pct(len(accepted), len(rows))
    accepted_accuracy = pct(sum(r["correct"] for r in accepted), len(accepted))
    wrong_accepted_rate = pct(len(wrong_accepted), len(rows))
    overall = pct(sum(r["correct"] for r in rows), len(rows))

    print()
    print("Benchmark summary")
    print("-"*72)
    print(f"Overall benchmark accuracy        : {overall:.1f}%")
    print(f"Known accuracy                    : {known_accuracy:.1f}%")
    print(f"Unknown rejection rate            : {unknown_rejection:.1f}%")
    print(f"False rejection rate (Known)      : {false_rejection:.1f}%")
    print(f"Coverage                          : {coverage:.1f}%")
    print(f"Accepted accuracy                 : {accepted_accuracy:.1f}%")
    print(f"Wrong-accepted rate               : {wrong_accepted_rate:.1f}%")
    print(f"Runtime                           : {elapsed:.2f}s")

    print()
    print("Per-group")
    print("-"*72)
    groups = sorted(set(r["group"] for r in rows))
    for group in groups:
        rr = [r for r in rows if r["group"] == group]
        print(
            f"{group:<24s} "
            f"{sum(r['correct'] for r in rr):>2d}/{len(rr):<2d} "
            f"({pct(sum(r['correct'] for r in rr), len(rr)):.1f}%)"
        )

    out = Path(ns.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_case.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["overall_accuracy", overall / 100.0])
        w.writerow(["known_accuracy", known_accuracy / 100.0])
        w.writerow(["unknown_rejection_rate", unknown_rejection / 100.0])
        w.writerow(["false_rejection_rate", false_rejection / 100.0])
        w.writerow(["coverage", coverage / 100.0])
        w.writerow(["accepted_accuracy", accepted_accuracy / 100.0])
        w.writerow(["wrong_accepted_rate", wrong_accepted_rate / 100.0])

    print()
    print("Output dir:", out)
    print("Per-case :", out / "per_case.csv")
    print("Summary  :", out / "summary.csv")


if __name__ == "__main__":
    main()
