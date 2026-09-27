# evaluate_semantic_encoder_adapter_v06.py
#
# Evaluate Semantic Encoder Adapter v0.6: Instruction Semantics.

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluate_semantic_encoder_adapter_v05 import (
    DISPLAY,
    TARGET_CASES,
    build_centroids,
    encode_adapted,
    encode_raw,
    heldout_technical_accuracy,
    print_case,
)
from model import LanguageModel
from semantic_encoder_adapter_v06 import (
    HIERARCHY_LABELS,
    load_semantic_adapter_v06_checkpoint,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v06.pt"


INSTRUCTION_PROBES = [
    (
        "G05",
        TARGET_CASES["G05"],
        "CPU-like",
    ),
    (
        "CPU non-arithmetic",
        "分岐、メモリアクセス、I/O制御など計算以外の命令も扱う装置は何ですか。",
        "CPU-like",
    ),
    (
        "CPU heterogeneous",
        "ロード、ストア、比較、ジャンプのような異なる種類の命令を柔軟に処理する装置は何ですか。",
        "CPU-like",
    ),
    (
        "GPU homogeneous",
        "同じ種類の演算を大量のデータに並列適用する装置は何ですか。",
        "GPU-like",
    ),
    (
        "GPU throughput",
        "多数の似た計算を高スループットで同時実行するプロセッサは何ですか。",
        "GPU-like",
    ),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Semantic Encoder Adapter v0.6."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    return p.parse_args()


def hierarchy_values(hierarchy_head, adapted):
    probs = torch.sigmoid(hierarchy_head(adapted.unsqueeze(0))[0])
    return {
        label: float(probs[i].item())
        for i, label in enumerate(HIERARCHY_LABELS)
    }


def style_scores(values):
    cpu_style = (
        values["general_purpose"]
        + values["control_oriented"]
        + values["heterogeneous_instruction"]
    ) / 3.0
    gpu_style = (
        values["throughput_oriented"]
        + values["data_parallel"]
        + values["homogeneous_computation"]
    ) / 3.0
    return cpu_style, gpu_style


def main():
    args = parse_args()

    for filename in (args.tokenizer, args.model, args.adapter):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    adapter, heads, hierarchy_head, checkpoint = (
        load_semantic_adapter_v06_checkpoint(args.adapter, device)
    )

    model.eval()
    adapter.eval()
    heads.eval()
    hierarchy_head.eval()

    raw_centroids = build_centroids(model, tokenizer, None)
    adapted_centroids = build_centroids(model, tokenizer, adapter)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.6 Evaluation")
    print("====================================================")
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", base_checkpoint.get("loss"))
    print("Adapter loss        :", checkpoint.get("loss"))
    print("Hierarchy labels    :", ", ".join(HIERARCHY_LABELS))
    print()

    g05 = print_case(
        "G05",
        TARGET_CASES["G05"],
        "tech_cpu",
        model,
        tokenizer,
        adapter,
        heads,
        raw_centroids,
        adapted_centroids,
    )
    print()

    g08 = print_case(
        "G08",
        TARGET_CASES["G08"],
        "tech_transformer",
        model,
        tokenizer,
        adapter,
        heads,
        raw_centroids,
        adapted_centroids,
    )
    print()

    raw_correct, total, _ = heldout_technical_accuracy(
        model,
        tokenizer,
        raw_centroids,
        None,
    )
    adapted_correct, _total2, rows = heldout_technical_accuracy(
        model,
        tokenizer,
        adapted_centroids,
        adapter,
    )

    print("Technical held-out centroid accuracy")
    print("------------------------------------")
    print(f"Raw     : {raw_correct}/{total} ({raw_correct / total:.1%})")
    print(
        f"Adapted : {adapted_correct}/{total} "
        f"({adapted_correct / total:.1%})"
    )
    print()

    for idx, intent, expected, adapted_top, margin, ok in rows:
        print(
            f"G{idx:02d} {intent:12s} "
            f"adapted={DISPLAY[adapted_top]:12s} "
            f"expected={DISPLAY[expected]:12s} "
            f"margin={margin:.6f} "
            f"{'PASS' if ok else 'MISS'}"
        )
    print()

    print("Instruction semantics probe")
    print("---------------------------")
    probe_results = {}

    for name, prompt, expected_style in INSTRUCTION_PROBES:
        raw = encode_raw(model, tokenizer, prompt)
        adapted = adapter(raw.unsqueeze(0))[0]
        values = hierarchy_values(hierarchy_head, adapted)
        cpu_style, gpu_style = style_scores(values)
        predicted = "CPU-like" if cpu_style > gpu_style else "GPU-like"
        probe_results[name] = (values, cpu_style, gpu_style, predicted)

        print(f"{name}: {prompt}")
        print(f"  expected style             : {expected_style}")
        print(f"  predicted style            : {predicted}")
        for label in HIERARCHY_LABELS:
            print(f"  {label:26s}: {values[label]:.6f}")
        print(f"  CPU-style score            : {cpu_style:.6f}")
        print(f"  GPU-style score            : {gpu_style:.6f}")
        print(f"  CPU-minus-GPU              : {cpu_style - gpu_style:.6f}")
        print()

    g05_values, g05_cpu, g05_gpu, g05_style = probe_results["G05"]
    nonarith = probe_results["CPU non-arithmetic"]
    hetero = probe_results["CPU heterogeneous"]
    gpu_homo = probe_results["GPU homogeneous"]

    print("Instruction Semantics Gate")
    print("--------------------------")
    checks = {
        "G05 centroid -> CPU": g05["adapted_top"] == "tech_cpu",
        "G08 centroid -> Transformer": g08["adapted_top"] == "tech_transformer",
        "Held-out technical non-regression": adapted_correct >= raw_correct,
        "G05 CPU-style > GPU-style": g05_cpu > g05_gpu,
        "G05 heterogeneous_instruction > homogeneous_computation":
            g05_values["heterogeneous_instruction"]
            > g05_values["homogeneous_computation"],
        "Non-arithmetic instruction -> CPU-like": nonarith[3] == "CPU-like",
        "Heterogeneous instruction -> CPU-like": hetero[3] == "CPU-like",
        "Homogeneous computation -> GPU-like": gpu_homo[3] == "GPU-like",
    }

    for name, passed in checks.items():
        print(f"{name:62s}: {'PASS' if passed else 'MISS'}")

    passed = sum(int(v) for v in checks.values())
    print()
    print(f"Gate score: {passed}/{len(checks)}")

    if passed == len(checks):
        print(
            "Result: Instruction Semantics gate passed. "
            "The semantic representation now explicitly distinguishes "
            "heterogeneous instruction execution from homogeneous computation."
        )
    else:
        print(
            "Result: Instruction Semantics gate is incomplete. "
            "Inspect the failed axes before generation integration."
        )


if __name__ == "__main__":
    main()
