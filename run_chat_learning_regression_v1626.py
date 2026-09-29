# run_chat_learning_regression_v1626.py
#
# LLM_GPU v1.6.26 regression suite for chat input quality, teaching validation,
# resolution routing, and concept-safe forgetting recovery.
#
# This suite does not modify the user's real learning log/state/queues.
# Recovery tests use a temporary directory.

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import chat


def check(name: str, condition: bool, detail: str = "") -> tuple[str, bool, str]:
    return name, bool(condition), detail


def run() -> int:
    results: list[tuple[str, bool, str]] = []

    # ------------------------------------------------------------------
    # 1. Input quality regression
    # ------------------------------------------------------------------
    input_cases = [
        ("bare-definition-marker", "とは", False, "missing query subject"),
        ("broken-short-query", "LLMは", False, "malformed short technical query"),
        ("bad-suffix", "GPU都", False, "malformed technical-term suffix"),
        ("repeated-intent", "AIとはとは", False, "malformed repeated intent"),
        ("known-definition", "AIとは", True, "input accepted"),
        ("known-definition-punct", "LLMとは、", True, "input accepted"),
        ("bare-acronym", "AI", True, "input accepted"),
    ]
    for label, text, expected_ok, expected_reason in input_cases:
        ok, reason = chat.input_quality_check(text)
        results.append(
            check(
                f"input:{label}",
                ok == expected_ok and reason == expected_reason,
                f"got ok={ok}, reason={reason}",
            )
        )

    # ------------------------------------------------------------------
    # 2. Intent/slot regression
    # ------------------------------------------------------------------
    intent_cases = [
        ("AIとは", "definition", ["AI"]),
        ("LLMとは、", "definition", ["LLM"]),
        ("CPUとGPUの違い", "comparison", ["CPU", "GPU"]),
        ("こんにちは", "greeting", []),
    ]
    for text, exp_intent, exp_slots in intent_cases:
        intent, slots = chat.classify_intent_and_slots(text)
        results.append(
            check(
                f"intent:{text}",
                intent == exp_intent and slots == exp_slots,
                f"got intent={intent}, slots={slots}",
            )
        )

    # ------------------------------------------------------------------
    # 2b. Greeting consistency regression
    # ------------------------------------------------------------------
    ok, coverage, reason = chat.slot_coverage_check(
        "greeting",
        [],
        "AIは人工知能です。",
    )
    results.append(
        check(
            "greeting:non-greeting answer rejected",
            (not ok) and coverage == 0.0 and reason == "greeting intent mismatch",
            f"got ok={ok}, coverage={coverage}, reason={reason}",
        )
    )

    ok, coverage, reason = chat.slot_coverage_check(
        "greeting",
        [],
        "こんにちは。今日は何について話しましょうか。",
    )
    results.append(
        check(
            "greeting:greeting answer accepted",
            ok and coverage == 1.0 and reason == "greeting matched",
            f"got ok={ok}, coverage={coverage}, reason={reason}",
        )
    )

    # ------------------------------------------------------------------
    # 3. Teaching validation regression
    # ------------------------------------------------------------------
    teaching_cases = [
        (
            "AI echo rejected",
            "AIとは",
            "AIは、人工知能です。",
            False,
        ),
        (
            "AI informative accepted",
            "AIとは",
            "AIは人間の知的な処理をコンピュータで実現する技術です。",
            True,
        ),
        (
            "LLM informative accepted",
            "LLMとは",
            "LLMは大量のテキストから学習し、言語を扱う大規模言語モデルです。",
            True,
        ),
        (
            "LLM wrong AI definition rejected",
            "LLMとは",
            "AIは人間の知的な処理をコンピュータで実現する技術です。",
            False,
        ),
        (
            "GPU correct accepted",
            "GPUとは",
            "GPUは多数の演算を並列に処理するプロセッサです。",
            True,
        ),
    ]
    for label, question, answer, expected_ok in teaching_cases:
        ok, reason = chat.validate_teaching_answer(question, answer)
        results.append(
            check(
                f"teaching:{label}",
                ok == expected_ok,
                f"got ok={ok}, reason={reason}",
            )
        )

    # ------------------------------------------------------------------
    # 4. Resolution routing regression
    # ------------------------------------------------------------------
    resolution_cases = [
        (
            "accept",
            True,
            "definition slot covered",
            "AIとは",
            ("ACCEPT", "none"),
        ),
        (
            "input reject",
            False,
            "missing query subject",
            "とは",
            ("INPUT_REJECT", "ask/rephrase"),
        ),
        (
            "learning gap",
            False,
            "concept slot missing: LLM",
            "LLMとは",
            ("LEARNING_GAP", "teach/train"),
        ),
        (
            "unknown knowledge",
            False,
            "unknown entity missing: QZX-91",
            "QZX-91とは",
            ("UNKNOWN_KNOWLEDGE", "retrieve/teach"),
        ),
        (
            "gate review",
            False,
            "low response agreement",
            "AIについて説明してください",
            ("GATE_REVIEW", "review/gate"),
        ),
    ]
    for label, accepted, reason, question, expected in resolution_cases:
        actual = chat.classify_resolution(accepted, reason, question)
        results.append(
            check(
                f"resolution:{label}",
                actual == expected,
                f"got={actual}",
            )
        )

    # ------------------------------------------------------------------
    # 5. Concept-safe recovery regression (temporary files only)
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="llm_gpu_v1626_") as tmp:
        root = Path(tmp)
        log = root / "chat_history.jsonl"
        state = root / "chat_learning_state.json"
        queue = root / "teaching_queue.jsonl"

        ai_q = "AIとは"
        ai_a = "AIは人間の知的な処理をコンピュータで実現する技術です。"
        llm_q = "LLMとは"
        llm_a = "LLMは大量のテキストから学習し、言語を扱う大規模言語モデルです。"

        rows = [
            {"user": ai_q, "assistant": ai_a, "source": "chat-manual"},
            {"user": llm_q, "assistant": llm_a, "source": "chat-manual"},
            {
                "user": ai_q,
                "assistant": "AIは人工知能です。",
                "source": "chat-manual",
            },
        ]
        log.write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
            encoding="utf-8",
        )

        trained = {
            chat.pair_fingerprint(ai_q, ai_a),
            chat.pair_fingerprint(llm_q, llm_a),
        }
        state.write_text(
            json.dumps(
                {
                    "version": "v1.6.26-test",
                    "trained_fingerprints": sorted(trained),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        queue_rows = [
            {
                "fingerprint": "test-ai",
                "resolution": "LEARNING_GAP",
                "action": "teach/train",
                "user": ai_q,
                "candidate_answer": "AIは人工知能です。",
                "reason": "definition answer is only an echo",
                "status": "pending",
            },
            {
                "fingerprint": "test-llm",
                "resolution": "LEARNING_GAP",
                "action": "teach/train",
                "user": llm_q,
                "candidate_answer": ai_a,
                "reason": "concept slot missing: LLM",
                "status": "pending",
            },
        ]
        queue.write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in queue_rows),
            encoding="utf-8",
        )

        matched, reactivated, rejected, selections = (
            chat.recover_forgotten_pairs_from_queue(
                queue,
                log,
                state,
                target_question=llm_q,
            )
        )

        chosen = [
            item for item in selections
            if item[0] == llm_q and item[1]
        ]
        results.append(
            check(
                "recovery:LLM exact-concept teacher selected",
                matched == 1
                and reactivated == 1
                and len(chosen) == 1
                and chosen[0][1] == llm_a,
                (
                    f"matched={matched}, reactivated={reactivated}, "
                    f"rejected={rejected}, selections={selections}"
                ),
            )
        )

        results.append(
            check(
                "recovery:AI teacher not used for LLM",
                all(item[1] != ai_a for item in chosen),
                f"selections={selections}",
            )
        )

        # Missing teacher must not fall back to a different concept.
        queue2 = root / "teaching_queue_missing.jsonl"
        queue2.write_text(
            json.dumps(
                {
                    "fingerprint": "test-missing",
                    "resolution": "LEARNING_GAP",
                    "action": "teach/train",
                    "user": "CUDAとは",
                    "candidate_answer": ai_a,
                    "reason": "concept slot missing: CUDA",
                    "status": "pending",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        m2, r2, _, s2 = chat.recover_forgotten_pairs_from_queue(
            queue2,
            log,
            state,
            target_question="CUDAとは",
        )
        results.append(
            check(
                "recovery:missing teacher stays NONE",
                m2 == 0
                and r2 == 0
                and len(s2) == 1
                and s2[0][1] == "",
                f"matched={m2}, reactivated={r2}, selections={s2}",
            )
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed

    print("=" * 78)
    print(" LLM_GPU v1.6.27 Chat/Learning Regression")
    print("=" * 78)

    for index, (name, ok, detail) in enumerate(results, 1):
        status = "PASS" if ok else "FAIL"
        print(f"{index:02d}. [{status}] {name}")
        if not ok and detail:
            print(f"    {detail}")

    print("-" * 78)
    print(f"Passed : {passed}/{len(results)}")
    print(f"Failed : {failed}/{len(results)}")
    print(f"Result : {'PASS' if failed == 0 else 'FAIL'}")
    print("=" * 78)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
