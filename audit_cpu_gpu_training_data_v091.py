# audit_cpu_gpu_training_data_v091.py
#
# Audit CPU/GPU definition symmetry in the training corpora.

from __future__ import annotations

from pathlib import Path

from train_sft_v09 import parse_dialogues, parse_instruction_pairs


CONVERSATION = Path("data/conversation-ja.txt")
INSTRUCTION = Path("data/instruction-ja.txt")

CPU_DIRECT_PREFIXES = (
    "CPUとは",
    "CPUって",
    "CPUを簡単",
    "CPUの定義",
    "CPUは何をする",
)
GPU_DIRECT_PREFIXES = (
    "GPUとは",
    "GPUって",
    "GPUを簡単",
    "GPUの定義",
    "GPUは何をする",
)

CPU_POSITIVE = ("CPU", "汎用", "命令", "制御", "中央処理", "Central Processing Unit")
GPU_POSITIVE = ("GPU", "並列", "同時", "大量")


def direct_rows(pairs, prefixes):
    return [
        (q, a)
        for q, a in pairs
        if any(q.startswith(prefix) for prefix in prefixes)
    ]


def has_any(text, words):
    return any(word in text for word in words)


def main():
    conversation = parse_dialogues(CONVERSATION.read_text(encoding="utf-8"))
    instruction = parse_instruction_pairs(INSTRUCTION.read_text(encoding="utf-8"))
    pairs = conversation + instruction

    cpu = direct_rows(pairs, CPU_DIRECT_PREFIXES)
    gpu = direct_rows(pairs, GPU_DIRECT_PREFIXES)

    print()
    print("========================================")
    print(" CPU/GPU Training Data Audit v0.9.1")
    print("========================================")
    print("Conversation pairs :", len(conversation))
    print("Instruction pairs  :", len(instruction))
    print("CPU direct rows    :", len(cpu))
    print("GPU direct rows    :", len(gpu))
    print()

    failures = []

    print("CPU direct definitions")
    print("----------------------")
    for q, a in cpu:
        ok = has_any(a, CPU_POSITIVE)
        print(f"{'PASS' if ok else 'MISS'} | {q} -> {a}")
        if not ok:
            failures.append(("cpu", q, a))

    print()
    print("GPU direct definitions")
    print("----------------------")
    for q, a in gpu:
        ok = has_any(a, GPU_POSITIVE)
        print(f"{'PASS' if ok else 'MISS'} | {q} -> {a}")
        if not ok:
            failures.append(("gpu", q, a))

    print()
    print("Symmetry")
    print("--------")
    print("CPU direct rows :", len(cpu))
    print("GPU direct rows :", len(gpu))
    ratio = len(cpu) / max(1, len(gpu))
    print("CPU/GPU ratio   :", f"{ratio:.3f}")

    if len(cpu) < 4:
        failures.append(("coverage", "CPU direct rows", str(len(cpu))))
    if not any(q.rstrip("。") == "CPUとは" for q, _ in cpu):
        failures.append(("coverage", "CPUとは", "missing"))
    if not any(q == "CPUとは何ですか。" for q, _ in cpu):
        failures.append(("coverage", "CPUとは何ですか。", "missing"))

    print()
    if failures:
        print("AUDIT RESULT: MISS")
        for row in failures:
            print("  ", row)
        raise SystemExit(1)

    print("AUDIT RESULT: PASS")


if __name__ == "__main__":
    main()
