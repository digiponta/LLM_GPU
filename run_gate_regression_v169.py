# run_gate_regression_v169.py
#
# LLM_GPU v1.6.9 gate regression benchmark.
# Evaluates the current chat model + gate end-to-end on a fixed prompt set.

from __future__ import annotations

import csv
from pathlib import Path

import torch

import chat
from model import LanguageModel
from tokenizer_bpe import Tokenizer


CASES = [
    ("known_definition", "AIとは", True),
    ("known_definition", "CPUとは", True),
    ("known_definition", "GPUとは", True),
    ("known_definition", "LLMとは", True),
    ("known_paraphrase", "AIについて説明してください", True),
    ("known_paraphrase", "GPUについて教えてください", True),
    ("known_relation", "AIは、LLMで実現される", True),
    ("typo", "GPU都", False),
    ("unknown_entity", "QZX-91とは", False),
    ("unknown_entity", "XK-2048について教えて", False),
    ("nonsense", "CPU GPU AI とはとは", False),
    ("nonsense", "未定義概念XYZとは", False),
]


def evaluate_case(
    model: LanguageModel,
    tokenizer: Tokenizer,
    question: str,
):
    prompt, _ = chat.build_prompt([], question, history_turns=0)

    results = [
        chat.generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=96,
            temperature=0.45,
            top_k=20,
            repetition_penalty=1.05,
            seed=0,
        )
    ]
    for probe in range(2):
        results.append(
            chat.generate_reply(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=96,
                temperature=0.30,
                top_k=10,
                repetition_penalty=1.05,
                seed=1000 + probe,
            )
        )

    primary = results[0]

    (
        semantic_ok,
        qa_similarity,
        previous_similarity,
        semantic_reason,
        intent,
        slots,
        slot_coverage,
    ) = chat.semantic_consistency_check(
        model=model,
        tokenizer=tokenizer,
        current_question=question,
        answer=primary.text,
        history=[],
        contamination_margin=0.05,
    )

    effective_agreement = chat.calibrated_agreement_threshold(
        intent=intent,
        semantic_ok=semantic_ok,
        slot_coverage=slot_coverage,
        base_threshold=0.35,
    )

    (
        accepted,
        confidence,
        agreement,
        semantic_agreement,
        reason,
    ) = chat.evaluate_unknown_gate(
        model=model,
        tokenizer=tokenizer,
        results=results,
        min_confidence=0.18,
        min_token_confidence=0.02,
        min_mean_margin=0.01,
        min_agreement=effective_agreement,
        min_semantic_agreement=0.82,
        allow_semantic_rescue=(semantic_ok and slot_coverage >= 1.0),
    )

    if accepted:
        if not semantic_ok:
            accepted = False
            reason = semantic_reason
        elif reason == "accepted":
            reason = semantic_reason

    return {
        "accepted": accepted,
        "answer": primary.text,
        "confidence": confidence,
        "min_token_confidence": primary.min_confidence,
        "mean_margin": primary.mean_top2_margin,
        "agreement": agreement,
        "semantic_agreement": semantic_agreement,
        "intent": intent,
        "slots": "|".join(slots),
        "slot_coverage": slot_coverage,
        "qa_similarity": qa_similarity,
        "reason": reason,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_path = Path(chat.DEFAULT_TOKENIZER)
    model_path = chat.choose_startup_model(chat.DEFAULT_MODEL)
    calibration_path = Path(chat.DEFAULT_CONCEPT_CALIBRATION)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(model_path),
        device=device,
    )
    chat.load_concept_calibration(calibration_path, device)

    print("=" * 92)
    print(" LLM_GPU v1.6.9 Gate Regression Benchmark")
    print("=" * 92)
    print("Device     :", device)
    if device.type == "cuda":
        print("GPU        :", torch.cuda.get_device_name(0))
    print("Model      :", model_path)
    print("Base loss  :", checkpoint.get("loss"))
    print("Cases      :", len(CASES))
    print()

    rows = []
    tp = tn = fp = fn = 0

    for index, (group, question, expected_known) in enumerate(CASES, 1):
        result = evaluate_case(model, tokenizer, question)
        actual_known = bool(result["accepted"])
        passed = actual_known == expected_known

        if expected_known and actual_known:
            tp += 1
        elif not expected_known and not actual_known:
            tn += 1
        elif not expected_known and actual_known:
            fp += 1
        else:
            fn += 1

        row = {
            "index": index,
            "group": group,
            "question": question,
            "expected": "KNOWN" if expected_known else "UNKNOWN",
            "actual": "KNOWN" if actual_known else "UNKNOWN",
            "pass": passed,
            **result,
        }
        rows.append(row)

        print(
            f"{index:02d} "
            f"{'PASS' if passed else 'FAIL':4s} "
            f"expected={row['expected']:7s} actual={row['actual']:7s} "
            f"lex={result['agreement']:.3f} sem={result['semantic_agreement']:.3f} "
            f"q={question}"
        )
        print(f"   answer={result['answer']}")
        print(f"   reason={result['reason']}")

    total = len(rows)
    correct = tp + tn
    accuracy = correct / total if total else 0.0
    known_recall = tp / (tp + fn) if tp + fn else 0.0
    unknown_recall = tn / (tn + fp) if tn + fp else 0.0
    balanced = (known_recall + unknown_recall) / 2.0

    output = Path("results/gate_regression_v169.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8-sig", newline="") as fp_out:
        writer = csv.DictWriter(fp_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 92)
    print("Summary")
    print("=" * 92)
    print(f"Accuracy       : {accuracy * 100:.2f}% ({correct}/{total})")
    print(f"Known recall   : {known_recall * 100:.2f}%")
    print(f"Unknown recall : {unknown_recall * 100:.2f}%")
    print(f"Balanced acc   : {balanced * 100:.2f}%")
    print(f"TP/TN/FP/FN    : {tp}/{tn}/{fp}/{fn}")
    print("Saved          :", output)


if __name__ == "__main__":
    main()
