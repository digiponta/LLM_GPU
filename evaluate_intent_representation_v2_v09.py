# evaluate_intent_representation_v2_v09.py
#
# Reuse the unchanged 30-case v0.9 development benchmark with
# Intent Representation v2 generation.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chat_intent_representation_v2_v09 import AI_PREFIX, USER_PREFIX, generate_reply
from evaluate_partial_intent_v09 import (
    CASES,
    dimension_match,
    semantic_match,
)
from intent_conditioning_v09 import load_intent_head
from intent_representation_v2_v09 import load_representation_v2_checkpoint
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INTENT_HEAD = "model/model-gpu-v0.8-intent-head.pt"
DEFAULT_CHECKPOINT = "model/model-gpu-v0.9-intent-representation-v2.pt"


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate v0.9 Intent Representation v2."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT_HEAD)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


def main():
    args = parse_args()
    for filename in (
        args.tokenizer,
        args.model,
        args.intent_head,
        args.checkpoint,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)

    intent_model, base_checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    intent_head, _head_checkpoint, labels = load_intent_head(
        args.intent_head,
        intent_model,
        device,
    )
    model, projection, checkpoint = load_representation_v2_checkpoint(
        args.checkpoint,
        device,
        labels,
    )

    print()
    print("==============================================")
    print(" LLM_GPU v0.9 Intent Representation v2 Test")
    print("==============================================")
    print("Device             :", device)
    print("Base checkpoint    :", base_checkpoint.get("loss"))
    print("Representation loss:", checkpoint.get("loss"))
    print("Intent dims        :", len(labels))
    print("Semantic dims      :", model.d_model)
    print("Fused dims         :", len(labels) + model.d_model)
    print("Intent alpha       :", projection.alpha)
    print("Inject after       :", projection.inject_after)
    print("Held-out cases     :", len(CASES))
    print()

    legacy_pass = semantic_pass = entity_pass = entity_total = 0
    entity_na = fluent_pass = strict_pass = 0
    per_intent = {}

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        prompt = f"{USER_PREFIX}{prompt_text}\n{AI_PREFIX}"

        reply, _, _ = generate_reply(
            generation_model=model,
            intent_model=intent_model,
            tokenizer=tokenizer,
            intent_head=intent_head,
            projection=projection,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.05,
        )

        legacy_ok, _, _ = semantic_match(
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
        if dims["missing_content"]:
            print("      missing-content: " + ", ".join(dims["missing_content"]))
        if dims["missing_entity"]:
            print("      missing-entity : " + ", ".join(dims["missing_entity"]))
        if dims["conflicts"]:
            print("      conflict       : " + ", ".join(dims["conflicts"]))
        if dims["fluency_issues"]:
            print("      fluency-issue  : " + ", ".join(dims["fluency_issues"]))
        if legacy_ok != dims["strict_ok"]:
            print("      legacy-rule    : " + ("PASS" if legacy_ok else "MISS"))

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


if __name__ == "__main__":
    main()
