# evaluate_semantic_aware_lm_head_v010.py

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from chat_semantic_generation_v062 import AI_PREFIX, USER_PREFIX, generate_reply
from evaluate_partial_intent_v09 import CASES, dimension_match, semantic_match
from semantic_aware_lm_head_v010 import load_semantic_aware_lm_head_checkpoint
from semantic_consistency_v09 import (
    gather_prompt_hidden,
    semantic_target,
    split_semantic_vector,
)
from semantic_encoder_adapter_v01 import CONCEPT_LABELS
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
from semantic_generation_integration_v09 import (
    forward_semantic_conditioned,
    infer_semantic_condition,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_CHECKPOINT = "model/model-gpu-v0.9.1-semantic-aware-lm-head-v010.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


@torch.no_grad()
def consistency_probe(
    text,
    tokenizer,
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    projection,
    consistency_head,
):
    ids = tokenizer.encode(f"{USER_PREFIX}{text}\n{AI_PREFIX}", add_bos=True)
    ids = ids[-generation_model.context_length:]
    device = next(generation_model.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_index = torch.tensor([x.size(1)-1], device=device)

    adapted, concept_prob, attribute_prob, hierarchy_prob = infer_semantic_condition(
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        x,
        prompt_index,
    )
    bias = projection(adapted, concept_prob, attribute_prob, hierarchy_prob)
    hidden = forward_semantic_conditioned(
        generation_model, x, bias, projection.inject_after
    )
    prompt_hidden = gather_prompt_hidden(hidden, prompt_index)
    pred = torch.sigmoid(consistency_head(prompt_hidden))
    target = semantic_target(concept_prob, attribute_prob, hierarchy_prob)

    pc, _pa, _ph = split_semantic_vector(pred[0])
    tc, _ta, _th = split_semantic_vector(target[0])
    return pc.tolist(), tc.tolist()


def main():
    args = parse_args()
    for filename in (
        args.tokenizer,
        args.base_model,
        args.semantic_adapter,
        args.checkpoint,
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
        _base_checkpoint,
        semantic_checkpoint,
    ) = load_frozen_semantic_path_v08(
        args.base_model, args.semantic_adapter, device
    )
    generation_model, projection, consistency_head, checkpoint = (
        load_semantic_aware_lm_head_checkpoint(args.checkpoint, device)
    )

    print()
    print("====================================================")
    print(" Semantic-Aware LM Head v0.10 Evaluation")
    print("====================================================")
    print("Device                  :", device)
    print("Semantic adapter loss   :", semantic_checkpoint.get("loss"))
    print("Checkpoint val loss     :", checkpoint.get("loss"))
    print("Checkpoint LM loss      :", checkpoint.get("lm_loss"))
    print("Checkpoint semantic loss:", checkpoint.get("consistency_loss"))
    print("LM Head LR              :", checkpoint.get("lm_head_learning_rate"))
    print()

    semantic_pass = strict_pass = fluent_pass = legacy_pass = 0
    entity_pass = entity_total = entity_na = 0
    hard = {}

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        prompt = f"{USER_PREFIX}{prompt_text}\n{AI_PREFIX}"

        reply, _, _details = generate_reply(
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
            pred, target = consistency_probe(
                prompt_text,
                tokenizer,
                generation_model,
                semantic_model,
                semantic_adapter,
                semantic_heads,
                hierarchy_head,
                projection,
                consistency_head,
            )
            hard[idx] = (pred, target)

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
            pred, target = hard[idx]
            pred_top = CONCEPT_LABELS[
                max(range(len(pred)), key=lambda j: pred[j])
            ]
            target_top = CONCEPT_LABELS[
                max(range(len(target)), key=lambda j: target[j])
            ]
            print(
                f"      target-concept={target_top} "
                f"predicted-concept={pred_top}"
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
    print("v0.9 semantic consistency : semantic 26/30, strict 26/30")
    print("v0.8 additive integration : semantic 25/30, strict 25/30")
    print()
    print("Hard-case concepts")
    print("------------------")
    for idx in (5, 8, 9, 28):
        pred, target = hard[idx]
        print(f"G{idx:02d}:")
        for label, t, p in zip(CONCEPT_LABELS, target, pred):
            print(f"  {label:16s} target={t:.4f} pred={p:.4f}")
        print()


if __name__ == "__main__":
    main()
