from __future__ import annotations

from ndc_phase12 import lexical_ndc_hint
from ndc_taxonomy_phase12 import PHASE1, PHASE2, phase2_children, validate_taxonomy


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return bool(cond)


def main():
    print("=" * 88)
    print(" LLM_GPU v1.7.2 NDC Phase 1/2 Hybrid Router Regression")
    print("=" * 88)

    passed = []
    try:
        validate_taxonomy()
        passed.append(check("taxonomy-validation", True))
    except Exception as e:
        print("[FAIL] taxonomy-validation:", e)
        passed.append(False)

    passed.append(check("phase1-count=10", len(PHASE1) == 10))
    passed.append(check("phase2-count=100", len(PHASE2) == 100))
    passed.append(check("phase2-complete-00-99", set(PHASE2) == {f"{i:02d}" for i in range(100)}))
    passed.append(check("children-4", phase2_children("4") == [f"4{i}" for i in range(10)]))

    probes = {
        "量子力学とは": "42",
        "AIとは": "00",
        "数学とは": "41",
        "日本文学とは": "91",
        "電気回路とは": "54",
        "経済学とは": "33",
        "日本史とは": "21",
    }
    for text, expected in probes.items():
        hint = lexical_ndc_hint(text)
        actual = hint[0] if hint else None
        passed.append(check(f"anchor:{text}->{expected}", actual == expected))

    passed.append(check(
        "longest-anchor-wins-日本文学",
        lexical_ndc_hint("日本文学について")[0] == "91",
    ))

    print("-" * 88)
    print(f"RESULT: {'PASS' if all(passed) else 'FAIL'} ({sum(passed)}/{len(passed)})")
    raise SystemExit(0 if all(passed) else 1)


if __name__ == "__main__":
    main()
