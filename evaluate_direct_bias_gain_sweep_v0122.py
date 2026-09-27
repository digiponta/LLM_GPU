# evaluate_direct_bias_gain_sweep_v0122.py
#
# LLM_GPU v0.9.1
# v0.12.2 Direct-Bias Gain Sweep
#
# No training.
#
# final first-token logits:
#   base_logits + gamma * gate * direct_bias
#
# Sweeps gamma while keeping v0.12.2 weights fixed.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from chat_semantic_generation_v062 import AI_PREFIX, USER_PREFIX
from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from entity_contrastive_logit_alignment_v0122 import load_checkpoint as load_v0122
from evaluate_entity_contrastive_logit_alignment_v0122 import prompt_state
from evaluate_partial_intent_v09 import CASES, dimension_match, semantic_match
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
from semantic_generation_integration_v09 import forward_semantic_conditioned
from semantic_lexical_logit_alignment_v012 import load_frozen_v011
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_NAME_BINDING = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
DEFAULT_V011 = "model/model-gpu-v0.9.1-lexical-generation-v011.pt"
DEFAULT_V0122 = "model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt"

GAINS = (0.50, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00)

ORIGINAL_G05 = "コンピュータの中心で多様な命令を処理する装置は何ですか。"
NAME_REQUEST_G05 = "コンピュータの中心で多様な命令を処理する装置の名前は何ですか。"


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
def gain_state(
    text,
    gamma,
    tokenizer,
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    v011_projection,
    adapter,
    gate,
):
    prompt, condition, semantic_bias, base_logits = prompt_state(
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
    bias = adapter(condition)[0]
    gate_prob = float(gate(condition)[0].item())
    effective = gamma * gate_prob * bias
    combined = base_logits + effective
    return prompt, semantic_bias, base_logits, combined, gate_prob, bias, effective


@torch.no_grad()
def generate_reply(
    text,
    gamma,
    tokenizer,
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    v011_projection,
    adapter,
    gate,
    max_new_tokens=96,
):
    (
        prompt,
        semantic_bias,
        _base_logits,
        _combined,
        gate_prob,
        _bias,
        effective,
    ) = gain_state(
        text,
        gamma,
        tokenizer,
        generation_model,
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        name_binding,
        v011_projection,
        adapter,
        gate,
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
            logits = logits + effective

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

    return reply.strip(), gate_prob


@torch.no_grad()
def g05_diagnostics(
    text,
    gamma,
    tokenizer,
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    v011_projection,
    adapter,
    gate,
):
    (
        _prompt,
        _semantic_bias,
        base_logits,
        combined,
        gate_prob,
        _bias,
        effective,
    ) = gain_state(
        text,
        gamma,
        tokenizer,
        generation_model,
        semantic_model,
        semantic_adapter,
        semantic_heads,
        hierarchy_head,
        name_binding,
        v011_projection,
        adapter,
        gate,
    )

    cpu_id = int(tokenizer.encode("CPU")[0])
    gpu_id = int(tokenizer.encode("GPU")[0])

    return {
        "gate": gate_prob,
        "cpu_base": float(base_logits[cpu_id].item()),
        "gpu_base": float(base_logits[gpu_id].item()),
        "cpu_bias": float(effective[cpu_id].item()),
        "gpu_bias": float(effective[gpu_id].item()),
        "cpu_final": float(combined[cpu_id].item()),
        "gpu_final": float(combined[gpu_id].item()),
        "margin": float((combined[cpu_id] - combined[gpu_id]).item()),
        "cpu_rank": int((combined > combined[cpu_id]).sum().item()) + 1,
        "gpu_rank": int((combined > combined[gpu_id]).sum().item()) + 1,
    }


def cpu_name_hit(reply):
    low = reply.lower()
    return (
        "cpu" in low
        or "central processing unit" in low
        or "中央処理装置" in reply
        or "中央演算処理装置" in reply
    )


def evaluate_gain(
    gamma,
    tokenizer,
    generation_model,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    v011_projection,
    adapter,
    gate,
    max_new_tokens,
):
    semantic_pass = strict_pass = fluent_pass = legacy_pass = 0
    entity_pass = entity_total = entity_na = 0
    rows = []

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        reply, gate_prob = generate_reply(
            prompt_text,
            gamma,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
            max_new_tokens=max_new_tokens,
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

        rows.append({
            "index": idx,
            "intent": str(case["intent"]),
            "prompt": prompt_text,
            "reply": reply,
            "semantic": bool(dims["semantic_ok"]),
            "entity": dims["entity_ok"],
            "fluency": bool(dims["fluent_ok"]),
            "strict": bool(dims["strict_ok"]),
            "gate": gate_prob,
        })

    n = len(CASES)
    return {
        "semantic": semantic_pass,
        "strict": strict_pass,
        "fluency": fluent_pass,
        "legacy": legacy_pass,
        "entity_pass": entity_pass,
        "entity_total": entity_total,
        "entity_na": entity_na,
        "rows": rows,
        "n": n,
    }


def row_by_index(result, index):
    return next(row for row in result["rows"] if row["index"] == index)


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
    print(" v0.12.2 Direct-Bias Gain Sweep")
    print("====================================================")
    print("Device                 :", device)
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Semantic adapter loss  :", semantic_checkpoint.get("loss"))
    print("Name binding loss      :", name_checkpoint.get("loss"))
    print("v0.11 integration loss :", v011_checkpoint.get("loss"))
    print("v0.12.2 loss           :", checkpoint.get("loss"))
    print("Training               : none")
    print("Formula                : base + gamma * gate * direct_bias")
    print("Gains                  :", ", ".join(str(x) for x in GAINS))
    print()

    results = {}

    for gamma in GAINS:
        result = evaluate_gain(
            gamma,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
            args.max_new_tokens,
        )
        results[gamma] = result

        original = g05_diagnostics(
            ORIGINAL_G05,
            gamma,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
        )
        name_req = g05_diagnostics(
            NAME_REQUEST_G05,
            gamma,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
        )
        original_reply, _ = generate_reply(
            ORIGINAL_G05,
            gamma,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
            args.max_new_tokens,
        )
        name_reply, _ = generate_reply(
            NAME_REQUEST_G05,
            gamma,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
            args.max_new_tokens,
        )

        g08 = row_by_index(result, 8)
        g09 = row_by_index(result, 9)
        g10 = row_by_index(result, 10)
        g27 = row_by_index(result, 27)
        g28 = row_by_index(result, 28)

        print(f"gamma={gamma:.2f}")
        print("-" * 42)
        print(
            f"30-case: semantic={result['semantic']:2d}/30 "
            f"strict={result['strict']:2d}/30 "
            f"fluency={result['fluency']:2d}/30 "
            f"legacy={result['legacy']:2d}/30"
        )
        print(
            f"Original G05 margin={original['margin']:+.6f} "
            f"CPU-rank={original['cpu_rank']} GPU-rank={original['gpu_rank']} "
            f"CPU-name={'PASS' if cpu_name_hit(original_reply) else 'MISS'}"
        )
        print("  reply:", original_reply)
        print(
            f"Name G05     margin={name_req['margin']:+.6f} "
            f"CPU-rank={name_req['cpu_rank']} GPU-rank={name_req['gpu_rank']} "
            f"CPU-name={'PASS' if cpu_name_hit(name_reply) else 'MISS'}"
        )
        print("  reply:", name_reply)
        print(
            "Hard cases: "
            f"G08={'PASS' if g08['strict'] else 'MISS'} "
            f"G09={'PASS' if g09['strict'] else 'MISS'} "
            f"G10={'PASS' if g10['strict'] else 'MISS'} "
            f"G27={'PASS' if g27['strict'] else 'MISS'} "
            f"G28={'PASS' if g28['strict'] else 'MISS'}"
        )
        print("  G08:", g08["reply"])
        print("  G09:", g09["reply"])
        print("  G10:", g10["reply"])
        print("  G27:", g27["reply"])
        print("  G28:", g28["reply"])
        print()

    print("Sweep summary")
    print("-------------")
    print(
        "gamma  sem strict flu  G05margin G05cpu  NameMargin NameCPU "
        "G08 G09 G10 G27 G28"
    )
    for gamma in GAINS:
        result = results[gamma]
        original = g05_diagnostics(
            ORIGINAL_G05,
            gamma,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
        )
        name_req = g05_diagnostics(
            NAME_REQUEST_G05,
            gamma,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
        )
        original_reply, _ = generate_reply(
            ORIGINAL_G05,
            gamma,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
            args.max_new_tokens,
        )
        name_reply, _ = generate_reply(
            NAME_REQUEST_G05,
            gamma,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            adapter,
            gate,
            args.max_new_tokens,
        )
        hard = [row_by_index(result, i)["strict"] for i in (8, 9, 10, 27, 28)]
        print(
            f"{gamma:4.2f}  "
            f"{result['semantic']:2d}   {result['strict']:2d}   {result['fluency']:2d}  "
            f"{original['margin']:+8.4f}   {'P' if cpu_name_hit(original_reply) else 'M'}     "
            f"{name_req['margin']:+8.4f}   {'P' if cpu_name_hit(name_reply) else 'M'}      "
            + " ".join("P" if x else "M" for x in hard)
        )


if __name__ == "__main__":
    main()
