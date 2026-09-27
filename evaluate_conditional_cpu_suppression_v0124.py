# evaluate_conditional_cpu_suppression_v0124.py
#
# LLM_GPU v0.9.1
# v0.12.4 Conditional CPU-Semantic Top-Competitor Suppression
#
# No training.
#
# Keep current-best v0.12.2 direct entity correction at gamma=1.0, then apply:
#
#   final_logits =
#       base_logits
#       + entity_gate * direct_bias
#       - lambda * P(tech_cpu) * blocker_mask
#
# Blocker suppression is therefore strong only when the frozen semantic head
# itself says the prompt is CPU-like.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from chat_semantic_generation_v062 import AI_PREFIX, USER_PREFIX
from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from entity_contrastive_logit_alignment_v0122 import load_checkpoint as load_v0122
from evaluate_partial_intent_v09 import CASES, dimension_match, semantic_match
from semantic_encoder_adapter_v01 import CONCEPT_LABELS
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
DEFAULT_V0122 = "model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt"

ORIGINAL_G05 = "コンピュータの中心で多様な命令を処理する装置は何ですか。"
NAME_REQUEST_G05 = "コンピュータの中心で多様な命令を処理する装置の名前は何ですか。"

# Identified by the v0.12.2 G05 Top-K diagnostic.
BLOCKER_TOKENS = ("多数の", "特集", "エ", "学習", "読み")
LAMBDAS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
DIRECT_GAIN = 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--name-binding", default=DEFAULT_NAME_BINDING)
    p.add_argument("--v011-checkpoint", default=DEFAULT_V011)
    p.add_argument("--v0122-checkpoint", default=DEFAULT_V0122)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


@torch.no_grad()
def prompt_state_with_cpu_prob(
    text,
    tokenizer,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    generation_model,
    v011_projection,
):
    device = next(generation_model.parameters()).device
    prompt = f"{USER_PREFIX}{text}\n{AI_PREFIX}"
    ids = tokenizer.encode(prompt, add_bos=True)
    ids = ids[-generation_model.context_length:]
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
    hidden = forward_semantic_conditioned(
        generation_model,
        x,
        semantic_bias,
        v011_projection.inject_after,
    )
    base_logits = generation_model.lm_head(hidden[:, -1, :])[0]

    cpu_index = CONCEPT_LABELS.index("tech_cpu")
    cpu_prob = float(concept_prob[0, cpu_index].item())
    concept_values = {
        label: float(concept_prob[0, i].item())
        for i, label in enumerate(CONCEPT_LABELS)
    }
    return prompt, condition, semantic_bias, base_logits, cpu_prob, concept_values


def blocker_ids(tokenizer):
    result = {}
    for token in BLOCKER_TOKENS:
        ids = tokenizer.encode(token)
        if not ids:
            raise RuntimeError(f"Tokenizer produced no ids for blocker {token!r}")
        # These blockers were observed as single first-token vocabulary entries.
        result[token] = int(ids[0])
    return result


@torch.no_grad()
def first_token_state(
    text,
    suppression_lambda,
    tokenizer,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    generation_model,
    v011_projection,
    adapter,
    gate,
    blockers,
):
    (
        prompt,
        condition,
        semantic_bias,
        base_logits,
        cpu_prob,
        concept_values,
    ) = prompt_state_with_cpu_prob(
        text,
        tokenizer,
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        name_binding,
        generation_model,
        v011_projection,
    )

    direct_bias = adapter(condition)[0]
    entity_gate = float(gate(condition)[0].item())
    effective_direct = DIRECT_GAIN * entity_gate * direct_bias

    suppression = torch.zeros_like(base_logits)
    amount = float(suppression_lambda) * cpu_prob
    for token_id in blockers.values():
        suppression[token_id] = amount

    final = base_logits + effective_direct - suppression
    return (
        prompt,
        semantic_bias,
        base_logits,
        final,
        cpu_prob,
        concept_values,
        entity_gate,
        effective_direct,
        suppression,
    )


@torch.no_grad()
def generate_reply(
    text,
    suppression_lambda,
    tokenizer,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    generation_model,
    v011_projection,
    adapter,
    gate,
    blockers,
    max_new_tokens=96,
):
    (
        prompt,
        semantic_bias,
        _base_logits,
        _final,
        cpu_prob,
        _concept_values,
        entity_gate,
        effective_direct,
        suppression,
    ) = first_token_state(
        text,
        suppression_lambda,
        tokenizer,
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        name_binding,
        generation_model,
        v011_projection,
        adapter,
        gate,
        blockers,
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
            v011_projection.inject_after,
        )
        logits = generation_model.lm_head(hidden[:, -1, :])[0].clone()

        if not response_ids:
            logits = logits + effective_direct - suppression

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
    return reply.strip(), cpu_prob, entity_gate


def rank_of(logits, token_id):
    return int((logits > logits[token_id]).sum().item()) + 1


def cpu_name_hit(reply):
    low = reply.lower()
    return (
        "cpu" in low
        or "central processing unit" in low
        or "中央処理装置" in reply
        or "中央演算処理装置" in reply
    )


@torch.no_grad()
def diagnostic_row(
    text,
    suppression_lambda,
    tokenizer,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    generation_model,
    v011_projection,
    adapter,
    gate,
    blockers,
):
    (
        _prompt,
        _semantic_bias,
        base_logits,
        final,
        cpu_prob,
        concept_values,
        entity_gate,
        _effective_direct,
        suppression,
    ) = first_token_state(
        text,
        suppression_lambda,
        tokenizer,
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        name_binding,
        generation_model,
        v011_projection,
        adapter,
        gate,
        blockers,
    )

    cpu_id = int(tokenizer.encode("CPU")[0])
    gpu_id = int(tokenizer.encode("GPU")[0])
    topv, topi = torch.topk(final, k=5)

    return {
        "cpu_prob": cpu_prob,
        "concept_values": concept_values,
        "entity_gate": entity_gate,
        "cpu_rank": rank_of(final, cpu_id),
        "gpu_rank": rank_of(final, gpu_id),
        "cpu_final": float(final[cpu_id].item()),
        "gpu_final": float(final[gpu_id].item()),
        "margin": float((final[cpu_id] - final[gpu_id]).item()),
        "top": [
            (
                int(token_id),
                tokenizer.decode([int(token_id)], skip_special_tokens=True),
                float(value),
            )
            for value, token_id in zip(topv.tolist(), topi.tolist())
        ],
        "suppression_amount": float(suppression_lambda) * cpu_prob,
        "blocker_values": {
            token: {
                "id": token_id,
                "base": float(base_logits[token_id].item()),
                "suppression": float(suppression[token_id].item()),
                "final": float(final[token_id].item()),
            }
            for token, token_id in blockers.items()
        },
    }


def evaluate_lambda(
    suppression_lambda,
    tokenizer,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    generation_model,
    v011_projection,
    adapter,
    gate,
    blockers,
    max_new_tokens,
):
    semantic_pass = strict_pass = fluent_pass = legacy_pass = 0
    rows = []

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        reply, cpu_prob, entity_gate = generate_reply(
            prompt_text,
            suppression_lambda,
            tokenizer,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            generation_model,
            v011_projection,
            adapter,
            gate,
            blockers,
            max_new_tokens,
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
        semantic_pass += int(dims["semantic_ok"])
        strict_pass += int(dims["strict_ok"])
        fluent_pass += int(dims["fluent_ok"])
        legacy_pass += int(legacy_ok)
        rows.append({
            "index": idx,
            "reply": reply,
            "strict": bool(dims["strict_ok"]),
            "cpu_prob": cpu_prob,
            "entity_gate": entity_gate,
        })

    return {
        "semantic": semantic_pass,
        "strict": strict_pass,
        "fluency": fluent_pass,
        "legacy": legacy_pass,
        "rows": rows,
    }


def row_by_index(result, index):
    return next(x for x in result["rows"] if x["index"] == index)


def main():
    args = parse_args()
    for filename in (
        args.tokenizer,
        args.base_model,
        args.semantic_adapter,
        args.name_binding,
        args.v011_checkpoint,
        args.v0122_checkpoint,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    blockers = blocker_ids(tokenizer)

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
    adapter, gate, checkpoint = load_v0122(
        args.v0122_checkpoint,
        device,
    )

    print()
    print("====================================================")
    print(" v0.12.4 Conditional CPU-Semantic Suppression Sweep")
    print("====================================================")
    print("Device                 :", device)
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Semantic adapter loss  :", semantic_checkpoint.get("loss"))
    print("Name binding loss      :", name_checkpoint.get("loss"))
    print("v0.11 integration loss :", v011_checkpoint.get("loss"))
    print("v0.12.2 loss           :", checkpoint.get("loss"))
    print("Training               : none")
    print("Direct gain            :", DIRECT_GAIN)
    print("CPU condition          : P(tech_cpu) from frozen semantic head")
    print("Suppression formula    : lambda * P(tech_cpu)")
    print("Lambdas                :", ", ".join(str(x) for x in LAMBDAS))
    print("Blockers:")
    for token, token_id in blockers.items():
        print(f"  {token!r:10s} -> id={token_id}")
    print()

    for suppression_lambda in LAMBDAS:
        result = evaluate_lambda(
            suppression_lambda,
            tokenizer,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            generation_model,
            v011_projection,
            adapter,
            gate,
            blockers,
            args.max_new_tokens,
        )

        original = diagnostic_row(
            ORIGINAL_G05,
            suppression_lambda,
            tokenizer,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            generation_model,
            v011_projection,
            adapter,
            gate,
            blockers,
        )
        name_req = diagnostic_row(
            NAME_REQUEST_G05,
            suppression_lambda,
            tokenizer,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            generation_model,
            v011_projection,
            adapter,
            gate,
            blockers,
        )
        original_reply, _, _ = generate_reply(
            ORIGINAL_G05,
            suppression_lambda,
            tokenizer,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            generation_model,
            v011_projection,
            adapter,
            gate,
            blockers,
            args.max_new_tokens,
        )
        name_reply, _, _ = generate_reply(
            NAME_REQUEST_G05,
            suppression_lambda,
            tokenizer,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            generation_model,
            v011_projection,
            adapter,
            gate,
            blockers,
            args.max_new_tokens,
        )

        print(f"lambda={suppression_lambda:.1f}")
        print("-" * 60)
        print(
            f"30-case semantic={result['semantic']}/30 "
            f"strict={result['strict']}/30 "
            f"fluency={result['fluency']}/30 "
            f"legacy={result['legacy']}/30"
        )
        print(
            f"Original G05 P(cpu)={original['cpu_prob']:.6f} "
            f"suppress={original['suppression_amount']:.4f} "
            f"CPU-rank={original['cpu_rank']} GPU-rank={original['gpu_rank']} "
            f"margin={original['margin']:+.4f} "
            f"CPU-name={'PASS' if cpu_name_hit(original_reply) else 'MISS'}"
        )
        print("  reply:", original_reply)
        print("  top5:", " | ".join(
            f"{rank+1}:{token!r}={value:+.3f}"
            for rank, (_tid, token, value) in enumerate(original["top"])
        ))
        print(
            f"Name G05     P(cpu)={name_req['cpu_prob']:.6f} "
            f"suppress={name_req['suppression_amount']:.4f} "
            f"CPU-rank={name_req['cpu_rank']} GPU-rank={name_req['gpu_rank']} "
            f"margin={name_req['margin']:+.4f} "
            f"CPU-name={'PASS' if cpu_name_hit(name_reply) else 'MISS'}"
        )
        print("  reply:", name_reply)
        print("  top5:", " | ".join(
            f"{rank+1}:{token!r}={value:+.3f}"
            for rank, (_tid, token, value) in enumerate(name_req["top"])
        ))

        hard = [row_by_index(result, i) for i in (8, 9, 10, 27, 28)]
        print(
            "Hard cases: "
            + " ".join(
                f"G{idx:02d}={'PASS' if row['strict'] else 'MISS'}"
                f"(Pcpu={row['cpu_prob']:.3f})"
                for idx, row in zip((8, 9, 10, 27, 28), hard)
            )
        )
        for idx, row in zip((8, 9, 10, 27, 28), hard):
            print(f"  G{idx:02d}: {row['reply']}")
        print()


if __name__ == "__main__":
    main()
