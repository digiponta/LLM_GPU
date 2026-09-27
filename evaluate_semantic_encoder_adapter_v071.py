# evaluate_semantic_encoder_adapter_v071.py
#
# Evaluate Semantic Encoder Adapter v0.7.1:
# G05 Instruction-Computation Relation Refinement.

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
from evaluate_semantic_encoder_adapter_v07 import PROBES
from model import LanguageModel
from semantic_encoder_adapter_v07 import SEMANTIC_HIERARCHY_LABELS
from semantic_encoder_adapter_v071 import (
    load_semantic_adapter_v071_checkpoint,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v071.pt"


RELATION_NEIGHBOR_PROBES = [
    (
        "Central diverse instructions",
        "コンピュータの中心で幅広い命令を処理する装置は何ですか。",
    ),
    (
        "Core varied instructions",
        "中核となる装置はさまざまな命令を実行します。",
    ),
    (
        "Instruction broader than compute",
        "命令実行は計算を含みますが、計算だけではありません。",
    ),
    (
        "Program composition",
        "プログラム処理は計算、分岐、ロード、ストアの組み合わせです。",
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


def print_relation(name, values):
    instruction = values["instruction_execution"]
    computation = values["computation"]
    print(f"{name}")
    print(f"  instruction_execution          : {instruction:.6f}")
    print(f"  computation                    : {computation:.6f}")
    print(f"  instruction-minus-computation  : {instruction - computation:.6f}")
    print(
        f"  heterogeneous_instruction_stream: "
        f"{values['heterogeneous_instruction_stream']:.6f}"
    )
    print(
        f"  repeated_computation           : "
        f"{values['repeated_computation']:.6f}"
    )


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
        load_semantic_adapter_v071_checkpoint(args.adapter, device)
    )

    model.eval()
    adapter.eval()
    heads.eval()
    hierarchy_head.eval()

    raw_centroids = build_centroids(model, tokenizer, None)
    adapted_centroids = build_centroids(model, tokenizer, adapter)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.7.1 Evaluation")
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
        model,
        tokenizer,
        raw_centroids,
        None,
    )
    adapted_correct, _total2, _rows = heldout_technical_accuracy(
        model,
        tokenizer,
        adapted_centroids,
        adapter,
    )

    print("Technical held-out centroid accuracy")
    print("------------------------------------")
    print(f"Raw     : {raw_correct}/{total} ({raw_correct/total:.1%})")
    print(f"Adapted : {adapted_correct}/{total} ({adapted_correct/total:.1%})")
    print()

    g05_adapted = adapter(
        encode_raw(model, tokenizer, TARGET_CASES["G05"]).unsqueeze(0)
    )[0]
    g05_values = values_for(hierarchy_head, g05_adapted)

    print("G05 relation")
    print("------------")
    print_relation("G05", g05_values)
    print()

    relation_results = {}
    print("Relation neighborhood probes")
    print("----------------------------")
    for name, prompt in RELATION_NEIGHBOR_PROBES:
        adapted = adapter(
            encode_raw(model, tokenizer, prompt).unsqueeze(0)
        )[0]
        values = values_for(hierarchy_head, adapted)
        relation_results[name] = values
        print_relation(name, values)
        print()

    old_probe_results = {}
    print("v0.7 hierarchy regression probes")
    print("--------------------------------")
    for name, prompt, _kind in PROBES:
        adapted = adapter(
            encode_raw(model, tokenizer, prompt).unsqueeze(0)
        )[0]
        old_probe_results[name] = values_for(hierarchy_head, adapted)

    checks = {
        "G05 centroid -> CPU":
            g05["adapted_top"] == "tech_cpu",
        "G08 centroid -> Transformer":
            g08["adapted_top"] == "tech_transformer",
        "Held-out technical non-regression":
            adapted_correct >= raw_correct,
        "G05 instruction_execution > computation":
            g05_values["instruction_execution"]
            > g05_values["computation"],
        "G05 heterogeneous stream strong":
            g05_values["heterogeneous_instruction_stream"] >= 0.5,
        "Central diverse instruction > computation":
            relation_results["Central diverse instructions"]["instruction_execution"]
            > relation_results["Central diverse instructions"]["computation"],
        "Core varied instruction > computation":
            relation_results["Core varied instructions"]["instruction_execution"]
            > relation_results["Core varied instructions"]["computation"],
        "Explicit broader relation":
            relation_results["Instruction broader than compute"]["instruction_execution"]
            > relation_results["Instruction broader than compute"]["computation"],
        "Program composition relation":
            relation_results["Program composition"]["instruction_execution"]
            > relation_results["Program composition"]["computation"],
        "Branch still instruction > computation":
            old_probe_results["Branch"]["instruction_execution"]
            > old_probe_results["Branch"]["computation"],
        "GPU repeated computation retained":
            old_probe_results["GPU repeated compute"]["repeated_computation"]
            >= old_probe_results["GPU repeated compute"]["heterogeneous_instruction_stream"],
        "GPU repeated compute remains instruction":
            old_probe_results["GPU repeated compute"]["instruction_execution"] >= 0.5,
    }

    print("G05 Instruction-Computation Relation Gate")
    print("-----------------------------------------")
    for name, ok in checks.items():
        print(f"{name:58s}: {'PASS' if ok else 'MISS'}")

    passed = sum(int(v) for v in checks.values())
    print()
    print(f"Gate score: {passed}/{len(checks)}")
    if passed == len(checks):
        print(
            "Result: v0.7.1 relation gate passed. "
            "G05 now preserves instruction_execution > computation."
        )
    else:
        print(
            "Result: v0.7.1 relation gate is incomplete. "
            "Do not reconnect generation yet."
        )


if __name__ == "__main__":
    main()
