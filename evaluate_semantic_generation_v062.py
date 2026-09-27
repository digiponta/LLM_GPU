# evaluate_semantic_generation_v062.py
#
# Fixed 30-case evaluation for
# Semantic Encoder Adapter v0.6.2 -> original v0.9 Generation Integration.

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
from semantic_encoder_adapter_v06 import HIERARCHY_LABELS
from semantic_generation_integration_v062 import (
    load_frozen_semantic_path_v062,
    load_integration_checkpoint_v062,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v062.pt"
DEFAULT_INTEGRATION = "model/model-gpu-v0.9.1-semantic-generation-v062.pt"


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate v0.6.2 semantic adapter with original v0.9 integration."
    )
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
    ) = load_frozen_semantic_path_v062(
        args.base_model,
        args.semantic_adapter,
        device,
    )

    generation_model, projection, integration_checkpoint = (
        load_integration_checkpoint_v062(args.integration, device)
    )

    print()
    print("====================================================")
    print(" Semantic v0.6.2 -> v0.9 Integration Evaluation")
    print("====================================================")
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss :", base_checkpoint.get("loss"))
    print("Semantic adapter loss:", semantic_checkpoint.get("loss"))
    print("Integration val loss :", integration_checkpoint.get("loss"))
    print("Feature dim          :", projection.input_dim)
    print("Hierarchy dim        :", projection.hierarchy_dim)
    print("Hierarchy labels     :", ", ".join(HIERARCHY_LABELS))
    print("Inject after         :", projection.inject_after)
    print("Alpha                :", projection.alpha)
    print("Held-out cases       :", len(CASES))
    print()

    legacy_pass = 0
    semantic_pass = 0
    entity_pass = 0
    entity_total = 0
    entity_na = 0
    fluent_pass = 0
    strict_pass = 0
    per_intent = {}
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

        legacy_ok, _missing, _conflicts = semantic_match(
            reply,
            case["required_all"],
            case["forbidden"],
        )

        intent = str(case["intent"])
        dims = dimension_match(
            reply,
            prompt_text,
            intent,
            case["required_all"],
            case["forbidden"],
        )

        legacy_pass += int(legacy_ok)
        semantic_pass += int(dims["semantic_ok"])
        if dims["entity_ok"] is None:
            entity_na += 1
        else:
            entity_total += 1
            entity_pass += int(dims["entity_ok"])
        fluent_pass += int(dims["fluent_ok"])
        strict_pass += int(dims["strict_ok"])

        stats = per_intent.get(intent, {"semantic": 0, "strict": 0, "n": 0})
        stats["semantic"] += int(dims["semantic_ok"])
        stats["strict"] += int(dims["strict_ok"])
        stats["n"] += 1
        per_intent[intent] = stats

        if idx in (5, 8, 9, 28):
            hard[idx] = details

        print(f"[G{idx:02d}] {intent:14s} 人: {prompt_text}")
        print(f"      AI: {reply}")
        print(
            "      semantic-content="
            + ("PASS" if dims["semantic_ok"] else "MISS")
            + " | entity="
            + (
                "N/A"
                if dims["entity_ok"] is None
                else ("PASS" if dims["entity_ok"] else "MISS")
            )
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
                        HIERARCHY_LABELS,
                        details["hierarchy_prob"],
                    )
                )
            )

    count = len(CASES)
    print()
    print("Summary")
    print("-------")
    print(
        f"Semantic-content rate : {semantic_pass}/{count} "
        f"({semantic_pass / count:.1%})"
    )
    if entity_total:
        print(
            f"Entity-explicit rate  : {entity_pass}/{entity_total} "
            f"({entity_pass / entity_total:.1%})  [N/A={entity_na}]"
        )
    else:
        print(f"Entity-explicit rate  : N/A  [N/A={entity_na}]")
    print(
        f"Fluency rate          : {fluent_pass}/{count} "
        f"({fluent_pass / count:.1%})"
    )
    print(
        f"Strict composite rate : {strict_pass}/{count} "
        f"({strict_pass / count:.1%})"
    )
    print(
        f"Legacy rule rate      : {legacy_pass}/{count} "
        f"({legacy_pass / count:.1%})"
    )

    print()
    print("Reference")
    print("---------")
    print("Original v0.9 integration (semantic v0.5): semantic 26/30, strict 25/30")
    print("v0.8 balanced tokens                   : semantic 25/30, strict 25/30")
    print("v0.9 cross-attention                   : semantic 25/30, strict 25/30")

    print()
    print("Per-intent")
    print("----------")
    print("intent         semantic      strict")
    for intent in sorted(per_intent):
        stats = per_intent[intent]
        n = stats["n"]
        sem = stats["semantic"]
        strict = stats["strict"]
        print(
            f"{intent:14s}: "
            f"{sem}/{n} ({sem / n:.1%})  "
            f"{strict}/{n} ({strict / n:.1%})"
        )

    print()
    print("Hard-case semantic signals")
    print("--------------------------")
    for idx in (5, 8, 9, 28):
        if idx not in hard:
            continue
        print(f"G{idx:02d}:")
        for label, value in zip(
            HIERARCHY_LABELS,
            hard[idx]["hierarchy_prob"],
        ):
            print(f"  {label:26s}: {value:.6f}")

        values = dict(zip(
            HIERARCHY_LABELS,
            hard[idx]["hierarchy_prob"],
        ))
        if (
            "heterogeneous_instruction" in values
            and "homogeneous_computation" in values
        ):
            print(
                "  hetero-minus-homo         : "
                f"{values['heterogeneous_instruction'] - values['homogeneous_computation']:.6f}"
            )
        print()


if __name__ == "__main__":
    main()
