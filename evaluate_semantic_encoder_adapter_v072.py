# evaluate_semantic_encoder_adapter_v072.py
#
# Evaluate Semantic Encoder Adapter v0.7.2:
# Program Composition Hierarchy.

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
from semantic_encoder_adapter_v072 import (
    SEMANTIC_HIERARCHY_LABELS,
    load_semantic_adapter_v072_checkpoint,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v072.pt"


PROBES = [
    ("Program composition", "プログラム処理は計算、分岐、ロード、ストアの組み合わせです。"),
    ("Program execution", "プログラム実行は複数の命令からなる命令列を処理することです。"),
    ("Instruction sequence", "命令列には計算命令だけでなく分岐やメモリ命令も含まれます。"),
    ("Computation child", "計算は命令実行の一部です。"),
    ("Branch child", "条件分岐は命令実行の一部ですが数値計算そのものではありません。"),
    ("Memory child", "ロードとストアはメモリ操作命令です。"),
    ("GPU repeated", "同じ計算を大量のデータへ並列適用する処理はGPU向きです。"),
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
    model, base_checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    adapter, heads, hierarchy_head, checkpoint = load_semantic_adapter_v072_checkpoint(
        args.adapter, device
    )

    model.eval()
    adapter.eval()
    heads.eval()
    hierarchy_head.eval()

    raw_centroids = build_centroids(model, tokenizer, None)
    adapted_centroids = build_centroids(model, tokenizer, adapter)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.7.2 Evaluation")
    print("====================================================")
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", base_checkpoint.get("loss"))
    print("Adapter loss        :", checkpoint.get("loss"))
    print("Hierarchy labels    :", ", ".join(SEMANTIC_HIERARCHY_LABELS))
    print()

    g05 = print_case(
        "G05", TARGET_CASES["G05"], "tech_cpu",
        model, tokenizer, adapter, heads, raw_centroids, adapted_centroids
    )
    print()
    g08 = print_case(
        "G08", TARGET_CASES["G08"], "tech_transformer",
        model, tokenizer, adapter, heads, raw_centroids, adapted_centroids
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

    results = {}
    print("Program Composition probes")
    print("--------------------------")
    for name, prompt in PROBES:
        adapted = adapter(encode_raw(model, tokenizer, prompt).unsqueeze(0))[0]
        values = values_for(hierarchy_head, adapted)
        results[name] = values
        print(name + ": " + prompt)
        for label in SEMANTIC_HIERARCHY_LABELS:
            print(f"  {label:32s}: {values[label]:.6f}")
        print()

    g05_values = values_for(
        hierarchy_head,
        adapter(encode_raw(model, tokenizer, TARGET_CASES["G05"]).unsqueeze(0))[0],
    )

    pc = results["Program composition"]
    pe = results["Program execution"]
    ins = results["Instruction sequence"]
    comp = results["Computation child"]
    branch = results["Branch child"]
    mem = results["Memory child"]
    gpu = results["GPU repeated"]

    checks = {
        "G05 centroid -> CPU": g05["adapted_top"] == "tech_cpu",
        "G08 centroid -> Transformer": g08["adapted_top"] == "tech_transformer",
        "Held-out technical non-regression": adapted_correct >= raw_correct,
        "G05 instruction_execution > computation":
            g05_values["instruction_execution"] > g05_values["computation"],
        "Program execution >= instruction sequence":
            pe["program_execution"] >= pe["instruction_sequence"],
        "Instruction sequence >= instruction execution":
            ins["instruction_sequence"] >= ins["instruction_execution"],
        "Program composition > computation":
            pc["program_execution"] > pc["computation"],
        "Program composition instruction sequence > computation":
            pc["instruction_sequence"] > pc["computation"],
        "Computation <= instruction execution":
            comp["computation"] <= comp["instruction_execution"],
        "Branch <= instruction execution":
            branch["control_flow"] <= branch["instruction_execution"],
        "Memory <= instruction execution":
            mem["memory_operation"] <= mem["instruction_execution"],
        "GPU repeated <= computation":
            gpu["repeated_computation"] <= gpu["computation"],
        "GPU computation <= instruction execution":
            gpu["computation"] <= gpu["instruction_execution"],
    }

    print("Program Composition Hierarchy Gate")
    print("----------------------------------")
    for name, ok in checks.items():
        print(f"{name:60s}: {'PASS' if ok else 'MISS'}")

    passed = sum(int(v) for v in checks.values())
    print()
    print(f"Gate score: {passed}/{len(checks)}")
    if passed == len(checks):
        print("Result: v0.7.2 program composition hierarchy gate passed.")
    else:
        print("Result: v0.7.2 program composition hierarchy is incomplete.")


if __name__ == "__main__":
    main()
