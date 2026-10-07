from __future__ import annotations

from ndc_taxonomy_phase12 import PHASE1, PHASE2, phase2_children, validate_taxonomy


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return bool(cond)


def main():
    print("=" * 88)
    print(" LLM_GPU v1.7.1 NDC Phase 1/2 Structural Regression")
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
    passed.append(check("00-general", PHASE2["00"] == "総記"))
    passed.append(check("42-physics", PHASE2["42"] == "物理学"))
    passed.append(check("54-electrical", PHASE2["54"] == "電気工学"))
    passed.append(check("91-japanese-literature", PHASE2["91"] == "日本文学"))

    print("-" * 88)
    print(f"RESULT: {'PASS' if all(passed) else 'FAIL'} ({sum(passed)}/{len(passed)})")
    raise SystemExit(0 if all(passed) else 1)


if __name__ == "__main__":
    main()
