# evaluate_semantic_encoder_adapter_v08.py
#
# Evaluate v0.8 Hierarchy-Constrained Semantic Head.

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
from evaluate_semantic_encoder_adapter_v072 import PROBES
from model import LanguageModel
from semantic_encoder_adapter_v08 import (
    PARENT,
    SEMANTIC_HIERARCHY_LABELS,
    load_semantic_adapter_v08_checkpoint,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"


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


def hierarchy_violations(values, tolerance=1e-7):
    violations = []
    for child, parent in PARENT.items():
        if parent is None:
            continue
        if values[child] > values[parent] + tolerance:
            violations.append(
                (child, parent, values[child], values[parent])
            )
    return violations


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
        load_semantic_adapter_v08_checkpoint(args.adapter, device)
    )

    model.eval()
    adapter.eval()
    heads.eval()
    hierarchy_head.eval()

    raw_centroids = build_centroids(model, tokenizer, None)
    adapted_centroids = build_centroids(model, tokenizer, adapter)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.8 Evaluation")
    print("====================================================")
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", base_checkpoint.get("loss"))
    print("Adapter loss        :", checkpoint.get("loss"))
    print("Head type           : hierarchy constrained")
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

    all_violations = []
    results = {}

    print("Hierarchy-constrained probes")
    print("----------------------------")
    for name, prompt in PROBES:
        adapted = adapter(
            encode_raw(model, tokenizer, prompt).unsqueeze(0)
        )[0]
        values = values_for(hierarchy_head, adapted)
        results[name] = values
        violations = hierarchy_violations(values)
        all_violations.extend((name,) + v for v in violations)

        print(name + ": " + prompt)
        for label in SEMANTIC_HIERARCHY_LABELS:
            print(f"  {label:32s}: {values[label]:.6f}")
        print("  hierarchy violations            :", len(violations))
        print()

    g05_values = values_for(
        hierarchy_head,
        adapter(
            encode_raw(model, tokenizer, TARGET_CASES["G05"]).unsqueeze(0)
        )[0],
    )
    g05_violations = hierarchy_violations(g05_values)

    pc = results["Program composition"]
    pe = results["Program execution"]
    ins = results["Instruction sequence"]
    comp = results["Computation child"]
    branch = results["Branch child"]
    mem = results["Memory child"]
    gpu = results["GPU repeated"]

    checks = {
        "G05 centroid -> CPU":
            g05["adapted_top"] == "tech_cpu",
        "G08 centroid -> Transformer":
            g08["adapted_top"] == "tech_transformer",
        "Held-out technical non-regression":
            adapted_correct >= raw_correct,
        "G05 has zero hierarchy violations":
            len(g05_violations) == 0,
        "All probes have zero hierarchy violations":
            len(all_violations) == 0,
        "Program execution >= instruction sequence":
            pe["program_execution"] >= pe["instruction_sequence"],
        "Instruction sequence >= instruction execution":
            ins["instruction_sequence"] >= ins["instruction_execution"],
        "Program composition >= computation":
            pc["program_execution"] >= pc["computation"],
        "Program sequence >= computation":
            pc["instruction_sequence"] >= pc["computation"],
        "Computation >= arithmetic logic":
            comp["computation"] >= comp["arithmetic_logic"],
        "Branch <= instruction execution":
            branch["control_flow"] <= branch["instruction_execution"],
        "Memory <= instruction execution":
            mem["memory_operation"] <= mem["instruction_execution"],
        "GPU computation >= repeated computation":
            gpu["computation"] >= gpu["repeated_computation"],
        "GPU instruction >= computation":
            gpu["instruction_execution"] >= gpu["computation"],
    }

    print("Hierarchy-Constrained Semantic Head Gate")
    print("----------------------------------------")
    for name, ok in checks.items():
        print(f"{name:60s}: {'PASS' if ok else 'MISS'}")

    passed = sum(int(v) for v in checks.values())
    print()
    print(f"Gate score: {passed}/{len(checks)}")
    print("Total structural violations:", len(all_violations) + len(g05_violations))
    if passed == len(checks):
        print(
            "Result: v0.8 constrained hierarchy gate passed. "
            "Parent-child ordering is guaranteed by construction."
        )
    else:
        print(
            "Result: structural ordering is guaranteed, but semantic calibration "
            "still needs refinement before generation integration."
        )


if __name__ == "__main__":
    main()
