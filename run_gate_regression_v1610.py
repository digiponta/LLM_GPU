# run_gate_regression_v1610.py
#
# LLM_GPU v1.6.10 Input Quality + Answer Correctness Benchmark.
# Evaluates both gate behavior and simple answer-content constraints.

from __future__ import annotations

import csv
from pathlib import Path

import torch

import chat
from model import LanguageModel
from tokenizer_bpe import Tokenizer


CASES = [
    {
        "group": "known_definition",
        "question": "AIとは",
        "expected_known": True,
        "required_any": ["人工知能", "技術", "知的"],
        "forbidden": [],
    },
    {
        "group": "known_definition",
        "question": "CPUとは",
        "expected_known": True,
        "required_any": ["中央処理装置", "プロセッサ", "命令実行"],
        "forbidden": ["人工知能"],
    },
    {
        "group": "known_definition",
        "question": "GPUとは",
        "expected_known": True,
        "required_any": ["並列", "演算", "プロセッサ", "画像処理装置"],
        "forbidden": ["人工知能", "LLM"],
    },
    {
        "group": "known_definition",
        "question": "LLMとは",
        "expected_known": True,
        "required_any": ["言語モデル", "大規模言語モデル"],
        "forbidden": ["画像処理装置"],
    },
    {
        "group": "known_paraphrase",
        "question": "AIについて説明してください",
        "expected_known": True,
        "required_any": ["人工知能", "技術", "知的"],
        "forbidden": [],
    },
    {
        "group": "known_paraphrase",
        "question": "GPUについて教えてください",
        "expected_known": True,
        "required_any": ["GPU", "並列", "演算", "プロセッサ", "画像処理装置"],
        "forbidden": ["人工知能", "LLM"],
    },
    {
        "group": "known_relation",
        "question": "AIは、LLMで実現される",
        "expected_known": True,
        "required_any": ["AI", "LLM", "技術"],
        "forbidden": [],
    },
    {
        "group": "typo",
        "question": "GPU都",
        "expected_known": False,
        "required_any": [],
        "forbidden": [],
    },
    {
        "group": "unknown_entity",
        "question": "QZX-91とは",
        "expected_known": False,
        "required_any": [],
        "forbidden": [],
    },
    {
        "group": "unknown_entity",
        "question": "XK-2048について教えて",
        "expected_known": False,
        "required_any": [],
        "forbidden": [],
    },
    {
        "group": "nonsense",
        "question": "CPU GPU AI とはとは",
        "expected_known": False,
        "required_any": [],
        "forbidden": [],
    },
    {
        "group": "nonsense",
        "question": "未定義概念XYZとは",
        "expected_known": False,
        "required_any": [],
        "forbidden": [],
    },
]


def answer_correct(answer: str, required_any: list[str], forbidden: list[str]) -> tuple[bool, str]:
    if required_any and not any(term.lower() in answer.lower() for term in required_any):
        return False, "missing required answer concept"
    bad = [term for term in forbidden if term.lower() in answer.lower()]
    if bad:
        return False, "forbidden answer concept: " + ", ".join(bad)
    return True, "answer content accepted"


def evaluate_case(model: LanguageModel, tokenizer: Tokenizer, case: dict):
    question = case["question"]

    input_ok, input_reason = chat.input_quality_check(question)
    if not input_ok:
        return {
            "accepted": False,
            "answer": "",
            "confidence": 0.0,
            "min_token_confidence": 0.0,
            "mean_margin": 0.0,
            "agreement": 0.0,
            "semantic_agreement": 0.0,
            "intent": "input_quality",
            "slots": "",
            "slot_coverage": 0.0,
            "qa_similarity": 0.0,
            "gate_reason": input_reason,
            "answer_ok": True,
            "answer_reason": "not evaluated for rejected input",
        }

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
        _previous_similarity,
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
        gate_reason,
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
            gate_reason = semantic_reason
        elif gate_reason == "accepted":
            gate_reason = semantic_reason

    answer_ok, answer_reason = answer_correct(
        primary.text,
        case["required_any"],
        case["forbidden"],
    )

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
        "gate_reason": gate_reason,
        "answer_ok": answer_ok,
        "answer_reason": answer_reason,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_path = Path(chat.DEFAULT_TOKENIZER)
    model_path = chat.choose_startup_model(chat.DEFAULT_MODEL)
    calibration_path = Path(chat.DEFAULT_CONCEPT_CALIBRATION)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(str(model_path), device=device)
    chat.load_concept_calibration(calibration_path, device)

    print("=" * 100)
    print(" LLM_GPU v1.6.10 Input Quality + Answer Correctness Benchmark")
    print("=" * 100)
    print("Device     :", device)
    if device.type == "cuda":
        print("GPU        :", torch.cuda.get_device_name(0))
    print("Model      :", model_path)
    print("Base loss  :", checkpoint.get("loss"))
    print("Cases      :", len(CASES))
    print()

    rows = []
    tp = tn = fp = fn = 0
    answer_pass = 0
    end_to_end_pass = 0

    for index, case in enumerate(CASES, 1):
        result = evaluate_case(model, tokenizer, case)

        expected_known = bool(case["expected_known"])
        actual_known = bool(result["accepted"])
        gate_pass = actual_known == expected_known

        if expected_known and actual_known:
            tp += 1
        elif not expected_known and not actual_known:
            tn += 1
        elif not expected_known and actual_known:
            fp += 1
        else:
            fn += 1

        if result["answer_ok"]:
            answer_pass += 1

        # For expected UNKNOWN cases, a correct rejection is enough.
        # For expected KNOWN cases, both gate acceptance and content correctness are required.
        end_to_end = gate_pass and (
            (not expected_known) or result["answer_ok"]
        )
        if end_to_end:
            end_to_end_pass += 1

        row = {
            "index": index,
            "group": case["group"],
            "question": case["question"],
            "expected": "KNOWN" if expected_known else "UNKNOWN",
            "actual": "KNOWN" if actual_known else "UNKNOWN",
            "gate_pass": gate_pass,
            "answer_ok": result["answer_ok"],
            "end_to_end_pass": end_to_end,
            **result,
        }
        rows.append(row)

        print(
            f"{index:02d} "
            f"{'PASS' if end_to_end else 'FAIL':4s} "
            f"expected={row['expected']:7s} actual={row['actual']:7s} "
            f"answer={'OK' if result['answer_ok'] else 'NG':2s} "
            f"lex={result['agreement']:.3f} sem={result['semantic_agreement']:.3f} "
            f"q={case['question']}"
        )
        print(f"   answer={result['answer']}")
        print(f"   gate={result['gate_reason']}")
        print(f"   content={result['answer_reason']}")

    total = len(rows)
    correct_gate = tp + tn
    gate_accuracy = correct_gate / total if total else 0.0
    known_recall = tp / (tp + fn) if tp + fn else 0.0
    unknown_recall = tn / (tn + fp) if tn + fp else 0.0
    balanced = (known_recall + unknown_recall) / 2.0
    answer_accuracy = answer_pass / total if total else 0.0
    end_to_end_accuracy = end_to_end_pass / total if total else 0.0

    output = Path("results/gate_regression_v1610.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8-sig", newline="") as fp_out:
        writer = csv.DictWriter(fp_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"Gate accuracy        : {gate_accuracy * 100:.2f}% ({correct_gate}/{total})")
    print(f"Known recall         : {known_recall * 100:.2f}%")
    print(f"Unknown recall       : {unknown_recall * 100:.2f}%")
    print(f"Balanced accuracy    : {balanced * 100:.2f}%")
    print(f"Answer content pass  : {answer_accuracy * 100:.2f}%")
    print(f"End-to-end accuracy  : {end_to_end_accuracy * 100:.2f}%")
    print(f"TP/TN/FP/FN          : {tp}/{tn}/{fp}/{fn}")
    print("Saved                :", output)


if __name__ == "__main__":
    main()
