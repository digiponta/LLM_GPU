# evaluate_cpu_gpu_local_margin_v0123.py

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cpu_gpu_local_margin_v0123 import load_checkpoint as load_v0123
from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from entity_contrastive_logit_alignment_v0122 import load_checkpoint as load_v0122
from entity_target_logit_alignment_v0121 import ENTITY_TARGETS
from evaluate_entity_contrastive_logit_alignment_v0122 import (
    entity_diagnostics,
    generate_reply,
)
from evaluate_partial_intent_v09 import CASES, dimension_match, semantic_match
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
from semantic_lexical_logit_alignment_v012 import load_frozen_v011
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_NAME_BINDING = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
DEFAULT_V011 = "model/model-gpu-v0.9.1-lexical-generation-v011.pt"
DEFAULT_V0122 = "model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt"
DEFAULT_V0123 = "model/model-gpu-v0.9.1-cpu-gpu-local-margin-v0123.pt"

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
    p.add_argument("--v0123-checkpoint", default=DEFAULT_V0123)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


def margin_from_rows(rows):
    by_label = {row["label"]: row for row in rows}
    cpu = by_label["tech_cpu"]
    gpu = by_label["tech_gpu"]
    return cpu, gpu, cpu["final"] - gpu["final"]


def main():
    args = parse_args()
    for filename in (
        args.tokenizer,
        args.base_model,
        args.semantic_adapter,
        args.name_binding,
        args.v011_checkpoint,
        args.v0122_checkpoint,
        args.v0123_checkpoint,
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
    old_adapter, old_gate, old_checkpoint = load_v0122(
        args.v0122_checkpoint,
        device,
    )
    adapter, gate, checkpoint, source_checkpoint = load_v0123(
        args.v0123_checkpoint,
        device,
    )

    print()
    print("====================================================")
    print(" CPU-GPU Local Margin Refinement v0.12.3 Eval")
    print("====================================================")
    print("Device               :", device)
    print("Base checkpoint loss :", base_checkpoint.get("loss"))
    print("Semantic adapter loss:", semantic_checkpoint.get("loss"))
    print("Name binding loss    :", name_checkpoint.get("loss"))
    print("v0.11 val loss       :", v011_checkpoint.get("loss"))
    print("v0.12.2 source loss  :", old_checkpoint.get("loss"))
    print("v0.12.3 val loss     :", checkpoint.get("loss"))
    print("Target margin        :", checkpoint.get("margin"))
    print("Gate                 : frozen from v0.12.2")
    print()

    semantic_pass = strict_pass = fluent_pass = legacy_pass = 0
    entity_pass = entity_total = entity_na = 0

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        reply, gate_prob, _bias, _effective = generate_reply(
            prompt_text,
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
            + f" | gate={gate_prob:.4f}"
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

    print("G05 local-margin comparison")
    print("---------------------------")
    for label, text in (
        ("Original G05", ORIGINAL_G05),
        ("Name-request G05", NAME_REQUEST_G05),
    ):
        old_gate_value, old_rows = entity_diagnostics(
            text,
            tokenizer,
            generation_model,
            semantic_model,
            semantic_adapter,
            semantic_heads,
            hierarchy_head,
            name_binding,
            v011_projection,
            old_adapter,
            old_gate,
        )
        new_gate_value, new_rows = entity_diagnostics(
            text,
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
        old_cpu, old_gpu, old_margin = margin_from_rows(old_rows)
        new_cpu, new_gpu, new_margin = margin_from_rows(new_rows)
        reply, _gp, _bias, _effective = generate_reply(
            text,
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
            max_new_tokens=args.max_new_tokens,
        )

        print(label)
        print("  Prompt:", text)
        print("  Reply :", reply)
        print(f"  Gate  : {new_gate_value:.6f} (v0.12.2={old_gate_value:.6f})")
        print(
            f"  v0.12.2 CPU={old_cpu['final']:+.4f} GPU={old_gpu['final']:+.4f} "
            f"margin={old_margin:+.4f}"
        )
        print(
            f"  v0.12.3 CPU={new_cpu['final']:+.4f} GPU={new_gpu['final']:+.4f} "
            f"margin={new_margin:+.4f}"
        )
        print(
            f"  margin delta = {new_margin-old_margin:+.4f}"
        )
        print(
            f"  CPU rank {new_cpu['base_rank']}->{new_cpu['final_rank']} | "
            f"GPU rank {new_gpu['base_rank']}->{new_gpu['final_rank']}"
        )
        print()

    print("Entity tokenization")
    print("-------------------")
    for concept, entity in ENTITY_TARGETS.items():
        ids = tokenizer.encode(entity)
        pieces = [tokenizer.decode([i], skip_special_tokens=True) for i in ids]
        print(f"{concept:16s} -> {entity:12s} ids={ids} pieces={pieces}")


if __name__ == "__main__":
    main()
