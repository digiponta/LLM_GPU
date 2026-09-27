# evaluate_semantic_token_conditioning_v091.py
#
# Fixed 30-case development evaluation for
# Semantic-to-Generation Interface v0.7.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chat_semantic_token_conditioning_v091 import (
    AI_PREFIX,
    USER_PREFIX,
    generate_reply,
)
from evaluate_partial_intent_v09 import (
    CASES,
    dimension_match,
    semantic_match,
)
from semantic_generation_integration_v09 import load_frozen_semantic_path
from semantic_token_conditioning_v091 import (
    SEMANTIC_TOKEN_LABELS,
    load_semantic_token_checkpoint,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9-semantic-adapter-v05.pt"
DEFAULT_TOKEN_MODEL = "model/model-gpu-v0.9.1-semantic-token-v07.pt"


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate semantic-token conditioning v0.7."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--token-model", default=DEFAULT_TOKEN_MODEL)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


def main():
    args = parse_args()

    for filename in (
        args.tokenizer,
        args.base_model,
        args.semantic_adapter,
        args.token_model,
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
    ) = load_frozen_semantic_path(
        args.base_model,
        args.semantic_adapter,
        device,
    )

    generation_model, projector, token_checkpoint = (
        load_semantic_token_checkpoint(args.token_model, device)
    )

    print()
    print("====================================================")
    print(" Semantic-to-Generation Interface v0.7 Evaluation")
    print("====================================================")
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", base_checkpoint.get("loss"))
    print("Semantic adapter loss:", semantic_checkpoint.get("loss"))
    print("Token val loss      :", token_checkpoint.get("loss"))
    print("Semantic tokens     :", ", ".join(SEMANTIC_TOKEN_LABELS))
    print("Insert after        :", projector.inject_after)
    print("Token scale         :", projector.scale)
    print("Held-out cases      :", len(CASES))
    print()

    legacy_pass = 0
    semantic_pass = 0
    entity_pass = 0
    entity_total = 0
    entity_na = 0
    fluent_pass = 0
    strict_pass = 0
    per_intent = {}
    hard_case_details = {}

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        prompt = f"{USER_PREFIX}{prompt_text}\n{AI_PREFIX}"

        reply, _, details = generate_reply(
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            projector,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.05,
        )

        legacy_ok, _legacy_missing, _legacy_conflicts = semantic_match(
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

        stats = per_intent.get(
            intent,
            {"semantic": 0, "strict": 0, "n": 0},
        )
        stats["semantic"] += int(dims["semantic_ok"])
        stats["strict"] += int(dims["strict_ok"])
        stats["n"] += 1
        per_intent[intent] = stats

        if idx in (5, 8, 9):
            hard_case_details[idx] = details

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
        if idx in hard_case_details:
            print(
                "      token norms    : "
                + ", ".join(
                    f"{label}={value:.3f}"
                    for label, value in zip(
                        SEMANTIC_TOKEN_LABELS,
                        details["token_norms"],
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
            f"({entity_pass / entity_total:.1%})"
            f"  [N/A={entity_na}]"
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
    print("v0.9 semantic integration : semantic 26/30, strict 25/30")
    print("v0.6 semantic gated       : semantic 25/30, strict 24/30")
    print("Boundary v1               : semantic 25/30, strict 25/30")

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
    print("Hard-case semantic-token diagnostics")
    print("------------------------------------")
    for idx in (5, 8, 9):
        if idx not in hard_case_details:
            continue
        print(
            f"G{idx:02d}: "
            + ", ".join(
                f"{label}={value:.3f}"
                for label, value in zip(
                    SEMANTIC_TOKEN_LABELS,
                    hard_case_details[idx]["token_norms"],
                )
            )
        )


if __name__ == "__main__":
    main()
