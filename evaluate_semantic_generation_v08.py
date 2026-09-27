# evaluate_semantic_generation_v08.py
#
# Fixed 30-case evaluation:
# v0.8 constrained semantic head -> original v0.9 additive integration.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chat_semantic_generation_v062 import (
    AI_PREFIX,
    USER_PREFIX,
    generate_reply,
)
from evaluate_partial_intent_v09 import (
    CASES,
    dimension_match,
    semantic_match,
)
from semantic_encoder_adapter_v08 import SEMANTIC_HIERARCHY_LABELS
from semantic_generation_integration_v08 import (
    load_frozen_semantic_path_v08,
    load_integration_checkpoint_v08,
)
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_INTEGRATION = "model/model-gpu-v0.9.1-semantic-generation-v08.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--integration", default=DEFAULT_INTEGRATION)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


def main():
    args = parse_args()
    for filename in (
        args.tokenizer,
        args.base_model,
        args.semantic_adapter,
        args.integration,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)

    (
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        base_checkpoint,
        semantic_checkpoint,
    ) = load_frozen_semantic_path_v08(
        args.base_model, args.semantic_adapter, device
    )
    generation_model, projection, integration_checkpoint = (
        load_integration_checkpoint_v08(args.integration, device)
    )

    print()
    print("====================================================")
    print(" Semantic v0.8 -> v0.9 Integration Evaluation")
    print("====================================================")
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss :", base_checkpoint.get("loss"))
    print("Semantic adapter loss:", semantic_checkpoint.get("loss"))
    print("Integration val loss :", integration_checkpoint.get("loss"))
    print("Feature dim          :", projection.input_dim)
    print("Hierarchy dim        :", projection.hierarchy_dim)
    print("Hierarchy labels     :", ", ".join(SEMANTIC_HIERARCHY_LABELS))
    print("Held-out cases       :", len(CASES))
    print()

    semantic_pass = strict_pass = fluent_pass = legacy_pass = 0
    entity_pass = entity_total = entity_na = 0
    hard = {}

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        prompt = f"{USER_PREFIX}{prompt_text}\n{AI_PREFIX}"

        reply, _, details = generate_reply(
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            projection,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.05,
        )

        legacy_ok, _m, _c = semantic_match(
            reply, case["required_all"], case["forbidden"]
        )
        dims = dimension_match(
            reply,
            prompt_text,
            str(case["intent"]),
            case["required_all"],
            case["forbidden"],
        )

        legacy_pass += int(legacy_ok)
        semantic_pass += int(dims["semantic_ok"])
        fluent_pass += int(dims["fluent_ok"])
        strict_pass += int(dims["strict_ok"])
        if dims["entity_ok"] is None:
            entity_na += 1
        else:
            entity_total += 1
            entity_pass += int(dims["entity_ok"])

        if idx in (5, 8, 9, 28):
            hard[idx] = details["hierarchy_prob"]

        print(f"[G{idx:02d}] {case['intent']:14s} 人: {prompt_text}")
        print(f"      AI: {reply}")
        print(
            "      semantic-content="
            + ("PASS" if dims["semantic_ok"] else "MISS")
            + " | entity="
            + ("N/A" if dims["entity_ok"] is None
               else ("PASS" if dims["entity_ok"] else "MISS"))
            + " | fluency="
            + ("PASS" if dims["fluent_ok"] else "MISS")
            + " | strict="
            + ("PASS" if dims["strict_ok"] else "MISS")
        )
        if idx in hard:
            print(
                "      hierarchy      : "
                + ", ".join(
                    f"{label}={value:.3f}"
                    for label, value in zip(
                        SEMANTIC_HIERARCHY_LABELS,
                        hard[idx],
                    )
                )
            )

    n = len(CASES)
    print()
    print("Summary")
    print("-------")
    print(f"Semantic-content rate : {semantic_pass}/{n} ({semantic_pass/n:.1%})")
    print(
        f"Entity-explicit rate  : {entity_pass}/{entity_total} "
        f"({entity_pass/entity_total:.1%})  [N/A={entity_na}]"
        if entity_total else f"Entity-explicit rate  : N/A [N/A={entity_na}]"
    )
    print(f"Fluency rate          : {fluent_pass}/{n} ({fluent_pass/n:.1%})")
    print(f"Strict composite rate : {strict_pass}/{n} ({strict_pass/n:.1%})")
    print(f"Legacy rule rate      : {legacy_pass}/{n} ({legacy_pass/n:.1%})")
    print()
    print("Reference")
    print("---------")
    print("v0.6.2 -> v0.9 integration : semantic 26/30, strict 26/30")
    print("original v0.9 integration  : semantic 26/30, strict 25/30")
    print()
    print("Hard-case constrained hierarchy")
    print("--------------------------------")
    for idx in (5, 8, 9, 28):
        if idx not in hard:
            continue
        print(f"G{idx:02d}:")
        for label, value in zip(SEMANTIC_HIERARCHY_LABELS, hard[idx]):
            print(f"  {label:30s}: {value:.6f}")
        print()


if __name__ == "__main__":
    main()
