# evaluate_g05_name_request_v0111.py
#
# v0.11.1 G05 Name-Request Evaluation
#
# No training. Compare the original G05 prompt with a minimally modified
# prompt that explicitly asks for the device name.

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from chat_semantic_generation_v062 import AI_PREFIX, USER_PREFIX
from cpu_name_binding_v0102 import load_cpu_name_binding_checkpoint
from evaluate_lexical_generation_alignment_v011 import (
    generate_reply,
    lexical_condition,
)
from lexical_generation_alignment_v011 import load_checkpoint
from semantic_generation_integration_v08 import load_frozen_semantic_path_v08
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_NAME_BINDING = "model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt"
DEFAULT_CHECKPOINT = "model/model-gpu-v0.9.1-lexical-generation-v011.pt"

ORIGINAL_G05 = "コンピュータの中心で多様な命令を処理する装置は何ですか。"
NAME_REQUEST_G05 = "コンピュータの中心で多様な命令を処理する装置の名前は何ですか。"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--semantic-adapter", default=DEFAULT_SEMANTIC_ADAPTER)
    p.add_argument("--name-binding", default=DEFAULT_NAME_BINDING)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


@torch.no_grad()
def lexical_vector(
    text,
    semantic_model,
    semantic_adapter,
    semantic_heads,
    hierarchy_head,
    name_binding,
    projection,
    tokenizer,
):
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


def cpu_name_hit(reply):
    normalized = reply.lower()
    return (
        "cpu" in normalized
        or "中央処理装置" in reply
        or "中央演算処理装置" in reply
        or "central processing unit" in normalized
    )


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
    generation_model, projection, integration_checkpoint = load_checkpoint(
        args.checkpoint,
        device,
    )

    cpu_anchor = F.normalize(
        torch.stack([
            lexical_vector(
                x,
                semantic_model,
                semantic_adapter,
                semantic_heads,
                hierarchy_head,
                name_binding,
                projection,
                tokenizer,
            )
            for x in (
                "CPU",
                "Central Processing Unit",
                "中央処理装置",
                "中央演算処理装置",
            )
        ]).mean(dim=0),
        dim=0,
    )
    gpu_anchor = F.normalize(
        torch.stack([
            lexical_vector(
                x,
                semantic_model,
                semantic_adapter,
                semantic_heads,
                hierarchy_head,
                name_binding,
                projection,
                tokenizer,
            )
            for x in (
                "GPU",
                "Graphics Processing Unit",
            )
        ]).mean(dim=0),
        dim=0,
    )

    print()
    print("====================================================")
    print(" G05 Name-Request Evaluation v0.11.1")
    print("====================================================")
    print("Training              : none")
    print("Generation checkpoint :", args.checkpoint)
    print("Base checkpoint loss  :", base_checkpoint.get("loss"))
    print("Semantic adapter loss :", semantic_checkpoint.get("loss"))
    print("Name binding loss     :", name_checkpoint.get("loss"))
    print("Integration val loss  :", integration_checkpoint.get("loss"))
    print()

    rows = [
        ("Original G05", ORIGINAL_G05),
        ("Name-request G05", NAME_REQUEST_G05),
    ]

    results = []
    for label, text in rows:
        prompt = f"{USER_PREFIX}{text}\n{AI_PREFIX}"
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
        cpu_s = float(torch.dot(lexical_identity, cpu_anchor).item())
        gpu_s = float(torch.dot(lexical_identity, gpu_anchor).item())
        hit = cpu_name_hit(reply)
        results.append((label, text, reply, cpu_s, gpu_s, hit))

        print(label)
        print("-" * len(label))
        print("Prompt :", text)
        print("Reply  :", reply)
        print(f"CPU lexical similarity : {cpu_s:.6f}")
        print(f"GPU lexical similarity : {gpu_s:.6f}")
        print(f"CPU-GPU margin         : {cpu_s-gpu_s:+.6f}")
        print("CPU name in answer     :", "PASS" if hit else "MISS")
        print()

    old = results[0]
    new = results[1]

    print("Direct comparison")
    print("-----------------")
    print(
        f"Lexical CPU margin: original={old[3]-old[4]:+.6f} "
        f"name-request={new[3]-new[4]:+.6f} "
        f"delta={(new[3]-new[4])-(old[3]-old[4]):+.6f}"
    )
    print(
        "Generated CPU name: "
        f"original={'PASS' if old[5] else 'MISS'} "
        f"name-request={'PASS' if new[5] else 'MISS'}"
    )
    print()

    if (not old[5]) and new[5]:
        print(
            "Result: adding an explicit name request changes G05 generation "
            "to lexical/entity identification."
        )
    elif old[5] == new[5]:
        print(
            "Result: answer-type wording alone does not change the CPU-name "
            "generation outcome."
        )
    else:
        print(
            "Result: prompt wording changes generation, but not in the expected "
            "direction; inspect both replies."
        )


if __name__ == "__main__":
    main()
