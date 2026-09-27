# evaluate_entity_target_logit_alignment_v0121.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from chat_semantic_generation_v062 import AI_PREFIX, USER_PREFIX
from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from entity_target_logit_alignment_v0121 import (
    ENTITY_TARGETS,
    load_checkpoint as load_entity_checkpoint,
)
from evaluate_partial_intent_v09 import CASES, dimension_match, semantic_match
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
from semantic_generation_integration_v09 import (
    forward_semantic_conditioned,
    infer_semantic_condition,
)
from semantic_lexical_logit_alignment_v012 import load_frozen_v011
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_NAME_BINDING = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
DEFAULT_V011 = "model/model-gpu-v0.9.1-lexical-generation-v011.pt"
DEFAULT_ENTITY = "model/model-gpu-v0.9.1-entity-target-logit-v0121.pt"

ORIGINAL_G05 = "コンピュータの中心で多様な命令を処理する装置は何ですか。"
NAME_REQUEST_G05 = "コンピュータの中心で多様な命令を処理する装置の名前は何ですか。"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--name-binding", default=DEFAULT_NAME_BINDING)
    p.add_argument("--v011-checkpoint", default=DEFAULT_V011)
    p.add_argument("--entity-checkpoint", default=DEFAULT_ENTITY)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


@torch.no_grad()
def prompt_parts(
    text,
    tokenizer,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    v011_projection,
):
    device = next(semantic_model.parameters()).device
    prompt = f"{USER_PREFIX}{text}\n{AI_PREFIX}"
    ids = tokenizer.encode(prompt, add_bos=True)
    ids = ids[-semantic_model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_index = torch.tensor([x.size(1)-1], dtype=torch.long, device=device)

    adapted, concept_prob, attribute_prob, hierarchy_prob = infer_semantic_condition(
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        x,
        prompt_index,
    )
    lexical_identity = name_binding(adapted)
    condition = torch.cat(
        [
            adapted,
            concept_prob,
            attribute_prob,
            hierarchy_prob,
            lexical_identity,
        ],
        dim=-1,
    )
    semantic_bias = v011_projection(
        adapted,
        concept_prob,
        attribute_prob,
        hierarchy_prob,
        lexical_identity,
    )
    return prompt, condition, semantic_bias


@torch.no_grad()
def generate_reply(
    text,
    tokenizer,
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    v011_projection,
    entity_adapter,
    max_new_tokens=96,
):
    prompt, condition, semantic_bias = prompt_parts(
        text,
        tokenizer,
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        name_binding,
        v011_projection,
    )
    direct_bias = entity_adapter(condition)[0]

    generated = list(tokenizer.encode(prompt, add_bos=True))
    response_ids: List[int] = []
    device = next(generation_model.parameters()).device

    for _ in range(max_new_tokens):
        context = generated[-generation_model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        hidden = forward_semantic_conditioned(
            generation_model,
            x,
            semantic_bias,
            v011_projection.inject_after,
        )
        logits = generation_model.lm_head(hidden[:, -1, :])[0].clone()

        if not response_ids:
            logits = logits + direct_bias

        for token_id in set(response_ids):
            if logits[token_id] >= 0:
                logits[token_id] /= 1.05
            else:
                logits[token_id] *= 1.05

        next_id = int(torch.argmax(logits).item())
        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)
        response_ids.append(next_id)
        decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
        if "\n" in decoded:
            break

    reply = tokenizer.decode(response_ids, skip_special_tokens=True)
    for marker in ("\n人:", "\nAI:", "\n"):
        if marker in reply:
            reply = reply.split(marker, 1)[0]
    return reply.strip(), direct_bias, semantic_bias, prompt


@torch.no_grad()
def first_token_rank_diagnostics(
    text,
    tokenizer,
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    v011_projection,
    entity_adapter,
):
    prompt, condition, semantic_bias = prompt_parts(
        text,
        tokenizer,
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        name_binding,
        v011_projection,
    )
    device = next(generation_model.parameters()).device
    ids = tokenizer.encode(prompt, add_bos=True)
    ids = ids[-generation_model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = forward_semantic_conditioned(
        generation_model,
        x,
        semantic_bias,
        v011_projection.inject_after,
    )
    base_logits = generation_model.lm_head(hidden[:, -1, :])[0]
    bias = entity_adapter(condition)[0]
    combined = base_logits + bias

    entity_rows = []
    for label, entity in ENTITY_TARGETS.items():
        ids = tokenizer.encode(entity)
        first_id = int(ids[0])
        base_rank = int((base_logits > base_logits[first_id]).sum().item()) + 1
        combined_rank = int((combined > combined[first_id]).sum().item()) + 1
        entity_rows.append(
            (
                label,
                entity,
                ids,
                first_id,
                float(base_logits[first_id].item()),
                float(bias[first_id].item()),
                base_rank,
                combined_rank,
            )
        )
    return entity_rows


def cpu_name_hit(reply):
    low = reply.lower()
    return (
        "cpu" in low
        or "central processing unit" in low
        or "中央処理装置" in reply
        or "中央演算処理装置" in reply
    )


def main():
    args = parse_args()
    for filename in (
        args.tokenizer,
        args.base_model,
        args.semantic_adapter,
        args.name_binding,
        args.v011_checkpoint,
        args.entity_checkpoint,
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
        args.base_model,
        args.semantic_adapter,
        device,
    )
    name_binding, name_checkpoint = load_cpu_name_binding_checkpoint(
        args.name_binding,
        device,
    )
    generation_model, v011_projection, v011_checkpoint = load_frozen_v011(
        args.v011_checkpoint,
        device,
    )
    entity_adapter, entity_checkpoint = load_entity_checkpoint(
        args.entity_checkpoint,
        device,
    )

    print()
    print("====================================================")
    print(" Entity-Target Logit Alignment v0.12.1 Evaluation")
    print("====================================================")
    print("Device               :", device)
    print("Base checkpoint loss :", base_checkpoint.get("loss"))
    print("Semantic adapter loss:", semantic_checkpoint.get("loss"))
    print("Name binding loss    :", name_checkpoint.get("loss"))
    print("v0.11 val loss       :", v011_checkpoint.get("loss"))
    print("Entity adapter loss  :", entity_checkpoint.get("loss"))
    print("Beta                 :", entity_adapter.beta)
    print()

    print("Entity tokenizer diagnostics")
    print("----------------------------")
    for label, entity in ENTITY_TARGETS.items():
        ids = tokenizer.encode(entity)
        pieces = [tokenizer.decode([i], skip_special_tokens=True) for i in ids]
        print(f"{label:16s} -> {entity:12s} ids={ids} pieces={pieces}")
    print()

    semantic_pass = strict_pass = fluent_pass = legacy_pass = 0
    entity_pass = entity_total = entity_na = 0

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        reply, _bias, _sem_bias, _prompt = generate_reply(
            prompt_text,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            entity_adapter,
            max_new_tokens=args.max_new_tokens,
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

    print("G05 entity-rank diagnostics")
    print("---------------------------")
    for label, text in (
        ("Original G05", ORIGINAL_G05),
        ("Name-request G05", NAME_REQUEST_G05),
    ):
        reply, _bias, _sem_bias, _prompt = generate_reply(
            text,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            entity_adapter,
            max_new_tokens=args.max_new_tokens,
        )
        rows = first_token_rank_diagnostics(
            text,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            entity_adapter,
        )
        print(label)
        print("  Prompt:", text)
        print("  Reply :", reply)
        print("  CPU name in answer:", "PASS" if cpu_name_hit(reply) else "MISS")
        for (
            concept,
            entity,
            ids,
            first_id,
            base_logit,
            bias_logit,
            base_rank,
            combined_rank,
        ) in rows:
            print(
                f"  {concept:16s} {entity:12s} first_id={first_id:4d} "
                f"base_rank={base_rank:4d} -> combined_rank={combined_rank:4d} "
                f"base={base_logit:+.4f} bias={bias_logit:+.4f}"
            )
        print()


if __name__ == "__main__":
    main()
