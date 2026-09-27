# evaluate_semantic_encoder_adapter_v061.py
#
# Evaluate Semantic Encoder Adapter v0.6.1:
# Instruction Binding Refinement.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate_semantic_encoder_adapter_v05 import (
    DISPLAY,
    TARGET_CASES,
    build_centroids,
    encode_raw,
    heldout_technical_accuracy,
    print_case,
)
from evaluate_semantic_encoder_adapter_v06 import (
    INSTRUCTION_PROBES,
    hierarchy_values,
    style_scores,
)
from model import LanguageModel
from semantic_encoder_adapter_v06 import HIERARCHY_LABELS
from semantic_encoder_adapter_v061 import (
    load_semantic_adapter_v061_checkpoint,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v061.pt"


EXTRA_BINDING_PROBES = [
    (
        "Diverse commands",
        "多様な命令を扱う装置は、同じ計算だけを繰り返す装置とは異なります。どちらがCPU側ですか。",
        "CPU-like",
    ),
    (
        "Mixed instruction stream",
        "分岐、比較、ロード、ストアを含む混在した命令列を処理するのは何向きですか。",
        "CPU-like",
    ),
    (
        "Repeated homogeneous math",
        "同一形式の数値演算を大量のデータに繰り返す処理は何向きですか。",
        "GPU-like",
    ),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Semantic Encoder Adapter v0.6.1."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    return p.parse_args()


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
        load_semantic_adapter_v061_checkpoint(args.adapter, device)
    )

    model.eval()
    adapter.eval()
    heads.eval()
    hierarchy_head.eval()

    raw_centroids = build_centroids(model, tokenizer, None)
    adapted_centroids = build_centroids(model, tokenizer, adapter)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.6.1 Evaluation")
    print("====================================================")
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", base_checkpoint.get("loss"))
    print("Adapter loss        :", checkpoint.get("loss"))
    print("Init adapter        :", checkpoint.get("init_adapter"))
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
        model, tokenizer, raw_centroids, None
    )
    adapted_correct, _total2, _rows = heldout_technical_accuracy(
        model, tokenizer, adapted_centroids, adapter
    )

    print("Technical held-out centroid accuracy")
    print("------------------------------------")
    print(f"Raw     : {raw_correct}/{total} ({raw_correct / total:.1%})")
    print(f"Adapted : {adapted_correct}/{total} ({adapted_correct / total:.1%})")
    print()

    probes = list(INSTRUCTION_PROBES) + list(EXTRA_BINDING_PROBES)
    results = {}

    print("Instruction binding probes")
    print("--------------------------")
    for name, prompt, expected_style in probes:
        raw = encode_raw(model, tokenizer, prompt)
        adapted = adapter(raw.unsqueeze(0))[0]
        values = hierarchy_values(hierarchy_head, adapted)
        cpu_style, gpu_style = style_scores(values)
        predicted = "CPU-like" if cpu_style > gpu_style else "GPU-like"
        direct_delta = (
            values["heterogeneous_instruction"]
            - values["homogeneous_computation"]
        )
        results[name] = (
            expected_style,
            predicted,
            values,
            cpu_style,
            gpu_style,
            direct_delta,
        )

        print(f"{name}: {prompt}")
        print(f"  expected style             : {expected_style}")
        print(f"  predicted style            : {predicted}")
        print(
            f"  heterogeneous_instruction : "
            f"{values['heterogeneous_instruction']:.6f}"
        )
        print(
            f"  homogeneous_computation   : "
            f"{values['homogeneous_computation']:.6f}"
        )
        print(f"  direct hetero-minus-homo   : {direct_delta:.6f}")
        print(f"  CPU-style score            : {cpu_style:.6f}")
        print(f"  GPU-style score            : {gpu_style:.6f}")
        print()

    g05_result = results["G05"]
    nonarith = results["CPU non-arithmetic"]
    hetero = results["CPU heterogeneous"]
    gpu_homo = results["GPU homogeneous"]
    diverse = results["Diverse commands"]
    mixed = results["Mixed instruction stream"]
    repeated = results["Repeated homogeneous math"]

    checks = {
        "G05 centroid -> CPU": g05["adapted_top"] == "tech_cpu",
        "G08 centroid -> Transformer": g08["adapted_top"] == "tech_transformer",
        "Held-out technical non-regression": adapted_correct >= raw_correct,
        "G05 style -> CPU-like": g05_result[1] == "CPU-like",
        "G05 hetero > homo": g05_result[5] > 0.0,
        "Non-arithmetic -> CPU-like": nonarith[1] == "CPU-like",
        "Heterogeneous instructions -> CPU-like": hetero[1] == "CPU-like",
        "Homogeneous computation -> GPU-like": gpu_homo[1] == "GPU-like",
        "Diverse commands -> CPU-like": diverse[1] == "CPU-like",
        "Mixed instruction stream -> CPU-like": mixed[1] == "CPU-like",
        "Repeated homogeneous math -> GPU-like": repeated[1] == "GPU-like",
    }

    print("Instruction Binding Refinement Gate")
    print("-----------------------------------")
    for name, passed in checks.items():
        print(f"{name:52s}: {'PASS' if passed else 'MISS'}")

    passed = sum(int(v) for v in checks.values())
    print()
    print(f"Gate score: {passed}/{len(checks)}")
    if passed == len(checks):
        print(
            "Result: v0.6.1 binding gate passed. "
            "Instruction diversity is directly bound to the heterogeneous axis."
        )
    else:
        print(
            "Result: v0.6.1 binding gate is incomplete. "
            "Do not retrain generation yet."
        )


if __name__ == "__main__":
    main()
