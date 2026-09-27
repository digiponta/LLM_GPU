# evaluate_lexical_generation_alignment_v011.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from chat_semantic_generation_v062 import AI_PREFIX, USER_PREFIX
from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from evaluate_partial_intent_v09 import CASES, dimension_match, semantic_match
from lexical_generation_alignment_v011 import load_checkpoint
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
from semantic_generation_integration_v09 import (
    forward_semantic_conditioned,
    infer_semantic_condition,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_NAME_BINDING = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
DEFAULT_CHECKPOINT = "model/model-gpu-v0.9.1-lexical-generation-v011.pt"


@torch.no_grad()
def lexical_condition(
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    projection,
    tokenizer,
    prompt,
):
    device = next(semantic_model.parameters()).device
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
    bias = projection(
        adapted,
        concept_prob,
        attribute_prob,
        hierarchy_prob,
        lexical_identity,
    )
    return bias, lexical_identity


@torch.no_grad()
def generate_reply(
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    projection,
    tokenizer,
    prompt,
    max_new_tokens=96,
):
    semantic_bias, lexical_identity = lexical_condition(
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        name_binding,
        projection,
        tokenizer,
        prompt,
    )

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
            projection.inject_after,
        )
        logits = generation_model.lm_head(hidden[:, -1, :])[0].clone()

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
    return reply.strip(), lexical_identity[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--name-binding", default=DEFAULT_NAME_BINDING)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


def main():
    args = parse_args()
    for filename in (
        args.tokenizer,
        args.base_model,
        args.semantic_adapter,
        args.name_binding,
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
    generation_model, projection, checkpoint = load_checkpoint(
        args.checkpoint,
        device,
    )

    # Build lexical CPU/GPU anchors for hard-case diagnostics.
    def lexical_vec(text):
        prompt = f"{USER_PREFIX}{text}\n{AI_PREFIX}"
        _bias, z = lexical_condition(
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            projection,
            tokenizer,
            prompt,
        )
        return z[0]

    cpu_anchor = F.normalize(
        torch.stack([
            lexical_vec("CPU"),
            lexical_vec("Central Processing Unit"),
            lexical_vec("中央処理装置"),
            lexical_vec("中央演算処理装置"),
        ]).mean(dim=0),
        dim=0,
    )
    gpu_anchor = F.normalize(
        torch.stack([
            lexical_vec("GPU"),
            lexical_vec("Graphics Processing Unit"),
        ]).mean(dim=0),
        dim=0,
    )

    print()
    print("====================================================")
    print(" Lexical Identity -> Generation Alignment v0.11 Eval")
    print("====================================================")
    print("Device               :", device)
    print("Base checkpoint loss :", base_checkpoint.get("loss"))
    print("Semantic adapter loss:", semantic_checkpoint.get("loss"))
    print("Name binding loss    :", name_checkpoint.get("loss"))
    print("Integration val loss :", checkpoint.get("loss"))
    print("Condition dim        :", projection.input_dim)
    print("Lexical dim          :", projection.lexical_dim)
    print()

    semantic_pass = strict_pass = fluent_pass = legacy_pass = 0
    entity_pass = entity_total = entity_na = 0
    hard = {}

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        prompt = f"{USER_PREFIX}{prompt_text}\n{AI_PREFIX}"
        reply, lexical_identity = generate_reply(
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            projection,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
        )

        legacy_ok, _m, _c = semantic_match(
            reply,
            case["required_all"],
            case["forbidden"],
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

        if idx in (4,5,6,8,9,27,28):
            hard[idx] = (
                float(torch.dot(lexical_identity, cpu_anchor).item()),
                float(torch.dot(lexical_identity, gpu_anchor).item()),
            )

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
            cpu_s, gpu_s = hard[idx]
            print(
                f"      lexical identity: CPU={cpu_s:.4f} "
                f"GPU={gpu_s:.4f} margin={cpu_s-gpu_s:+.4f}"
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
    print("v0.8 semantic-only additive : semantic 25/30, strict 25/30")
    print("v0.9 consistency            : semantic 26/30, strict 26/30")
    print()
    print("Primary hard cases")
    print("------------------")
    for idx in (4,5,6,8,9,27,28):
        if idx not in hard:
            continue
        cpu_s, gpu_s = hard[idx]
        print(
            f"G{idx:02d}: CPU={cpu_s:.6f} GPU={gpu_s:.6f} "
            f"margin={cpu_s-gpu_s:+.6f}"
        )


if __name__ == "__main__":
    main()
