# evaluate_semantic_encoder_adapter_v07.py
#
# Evaluate Semantic Encoder Adapter v0.7:
# Instruction-Computation Hierarchy.

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from evaluate_semantic_encoder_adapter_v05 import (
    TARGET_CASES,
    build_centroids,
    encode_raw,
    heldout_technical_accuracy,
    print_case,
)
from model import LanguageModel
from semantic_encoder_adapter_v07 import (
    SEMANTIC_HIERARCHY_LABELS,
    load_semantic_adapter_v07_checkpoint,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v07.pt"


PROBES = [
    (
        "Instruction includes computation",
        "計算は命令実行の一部ですか。",
        "relation",
    ),
    (
        "Instruction not equal computation",
        "命令実行は計算だけで構成されますか。",
        "non-computation",
    ),
    (
        "Branch",
        "条件分岐は命令実行ですが数値計算そのものではありません。",
        "non-computation",
    ),
    (
        "Memory",
        "ロードとストアはメモリを扱う命令です。",
        "non-computation",
    ),
    (
        "CPU mixed stream",
        "算術、分岐、ロード、ストアを組み合わせた多様な命令列を処理する装置は何ですか。",
        "cpu",
    ),
    (
        "GPU repeated compute",
        "同じ計算を大量のデータへ並列適用する処理は何向きですか。",
        "gpu",
    ),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    return p.parse_args()


def values_for(head, adapted):
    probs = torch.sigmoid(head(adapted.unsqueeze(0))[0])
    return {
        label: float(probs[i].item())
        for i, label in enumerate(SEMANTIC_HIERARCHY_LABELS)
    }


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
        load_semantic_adapter_v07_checkpoint(args.adapter, device)
    )

    model.eval()
    adapter.eval()
    heads.eval()
    hierarchy_head.eval()

    raw_centroids = build_centroids(model, tokenizer, None)
    adapted_centroids = build_centroids(model, tokenizer, adapter)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.7 Evaluation")
    print("====================================================")
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", base_checkpoint.get("loss"))
    print("Adapter loss        :", checkpoint.get("loss"))
    print("Hierarchy labels    :", ", ".join(SEMANTIC_HIERARCHY_LABELS))
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
    print(f"Raw     : {raw_correct}/{total} ({raw_correct/total:.1%})")
    print(f"Adapted : {adapted_correct}/{total} ({adapted_correct/total:.1%})")
    print()

    print("Instruction-Computation probes")
    print("------------------------------")
    results = {}

    for name, prompt, kind in PROBES:
        adapted = adapter(
            encode_raw(model, tokenizer, prompt).unsqueeze(0)
        )[0]
        v = values_for(hierarchy_head, adapted)
        results[name] = v

        print(f"{name}: {prompt}")
        for label in SEMANTIC_HIERARCHY_LABELS:
            print(f"  {label:32s}: {v[label]:.6f}")
        print()

    g05_adapted = adapter(
        encode_raw(model, tokenizer, TARGET_CASES["G05"]).unsqueeze(0)
    )[0]
    g05v = values_for(hierarchy_head, g05_adapted)

    checks = {
        "G05 centroid -> CPU": g05["adapted_top"] == "tech_cpu",
        "G08 centroid -> Transformer": g08["adapted_top"] == "tech_transformer",
        "Held-out technical non-regression": adapted_correct >= raw_correct,
        "G05 instruction_execution >= computation":
            g05v["instruction_execution"] >= g05v["computation"],
        "G05 heterogeneous stream strong":
            g05v["heterogeneous_instruction_stream"] >= 0.5,
        "Computation <= instruction_execution":
            results["Instruction includes computation"]["computation"]
            <= results["Instruction includes computation"]["instruction_execution"],
        "Branch is instruction execution":
            results["Branch"]["instruction_execution"] >= 0.5,
        "Branch not forced to computation":
            results["Branch"]["instruction_execution"]
            > results["Branch"]["computation"],
        "Memory is instruction execution":
            results["Memory"]["instruction_execution"] >= 0.5,
        "CPU mixed stream heterogeneous":
            results["CPU mixed stream"]["heterogeneous_instruction_stream"]
            >= results["CPU mixed stream"]["repeated_computation"],
        "GPU repeated compute repeated":
            results["GPU repeated compute"]["repeated_computation"]
            >= results["GPU repeated compute"]["heterogeneous_instruction_stream"],
        "GPU repeated compute is still instruction execution":
            results["GPU repeated compute"]["instruction_execution"] >= 0.5,
    }

    print("Instruction-Computation Hierarchy Gate")
    print("--------------------------------------")
    for name, ok in checks.items():
        print(f"{name:58s}: {'PASS' if ok else 'MISS'}")

    passed = sum(int(v) for v in checks.values())
    print()
    print(f"Gate score: {passed}/{len(checks)}")
    if passed == len(checks):
        print(
            "Result: v0.7 hierarchy gate passed. "
            "Computation is represented as a subtype of instruction execution."
        )
    else:
        print(
            "Result: v0.7 hierarchy gate is incomplete. "
            "Inspect the failed relation before generation integration."
        )


if __name__ == "__main__":
    main()
