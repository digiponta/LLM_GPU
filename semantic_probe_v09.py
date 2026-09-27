# semantic_probe_v09.py
#
# LLM_GPU v0.9 Semantic Probe Diagnostic.
#
# This script performs no training. It probes the frozen v0.8 pairwise-best
# encoder used by the intent path and asks:
#
#   1. Which technical concept centroid is each held-out prompt closest to?
#   2. What is the top-1 / top-2 cosine margin?
#   3. For hard cases G05 and G08, does the prompt representation already
#      contain the expected concept/category signal?
#   4. Does the frozen intent head agree with the centroid probe?
#
# The purpose is to separate encoder-side representation failure from
# semantic-to-generation mapping failure.

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from augment_sft_v07 import (
    AUGMENT_BANK,
    REVERSE_DEFINITION_ROWS,
    TARGETED_BOUNDARY_ROWS,
    TECHNICAL_CONCEPTS,
    technical_minimal_pairs,
)
from evaluate_partial_intent_v09 import CASES
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INTENT_HEAD = "model/model-gpu-v0.8-intent-head.pt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "

TECH_LABELS = [
    "tech_gpu",
    "tech_cpu",
    "tech_llm",
    "tech_transformer",
    "tech_cuda",
    "tech_python",
]

DISPLAY = {
    "tech_gpu": "GPU",
    "tech_cpu": "CPU",
    "tech_llm": "LLM",
    "tech_transformer": "Transformer",
    "tech_cuda": "CUDA",
    "tech_python": "Python",
}

TARGET_CASES = {
    "G05": "コンピュータの中心で多様な命令を処理する装置は何ですか。",
    "G08": "Attentionを中心に使う代表的な構造は何ですか。",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Probe frozen v0.8 semantic representations."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT_HEAD)
    p.add_argument(
        "--all-tech-cases",
        action="store_true",
        help="Also print probe results for every technical held-out case.",
    )
    return p.parse_args()


def encode_prompt(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
) -> torch.Tensor:
    device = next(model.parameters()).device
    prompt = f"{USER_PREFIX}{text}\n{AI_PREFIX}"
    ids = tokenizer.encode(prompt, add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        hidden = model.forward_hidden(x)
    return F.normalize(hidden[:, -1, :], dim=-1)[0]


def mean_unit(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(list(vectors), dim=0)
    return F.normalize(stacked.mean(dim=0), dim=-1)


def collect_reference_prompts() -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = defaultdict(list)

    # Core augmentation-bank prompts.
    for label in TECH_LABELS:
        bank = AUGMENT_BANK.get(label, {})
        refs[label].extend(str(x) for x in bank.get("prompts", []))

    # Shared minimal-pair prompts.
    for prompt, _answer, label in technical_minimal_pairs():
        if label in TECH_LABELS:
            refs[label].append(prompt)

    # Reverse-definition prompts.
    for prompt, _answer, label in REVERSE_DEFINITION_ROWS:
        if label in TECH_LABELS:
            refs[label].append(prompt)

    # Boundary v1 technical rows only.
    for prompt, _answer, label in TARGETED_BOUNDARY_ROWS:
        if label in TECH_LABELS:
            refs[label].append(prompt)

    # Deduplicate while preserving order.
    for label in refs:
        refs[label] = list(dict.fromkeys(refs[label]))

    return refs


def build_centroids(
    model: LanguageModel,
    tokenizer: Tokenizer,
    refs: Dict[str, List[str]],
):
    centroids = {}
    vectors = {}

    for label in TECH_LABELS:
        vecs = [encode_prompt(model, tokenizer, p) for p in refs[label]]
        vectors[label] = vecs
        centroids[label] = mean_unit(vecs)

    return centroids, vectors


def cosine_scores(
    vector: torch.Tensor,
    centroids: Dict[str, torch.Tensor],
):
    scores = {
        label: float(torch.dot(vector, centroid).item())
        for label, centroid in centroids.items()
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    margin = ranked[0][1] - ranked[1][1]
    return scores, ranked, margin


def intent_scores(
    vector: torch.Tensor,
    intent_head,
    labels: Sequence[str],
):
    with torch.no_grad():
        logits = intent_head(vector.unsqueeze(0))
        probs = torch.sigmoid(logits)[0]
    pairs = [
        (label, float(probs[i].item()))
        for i, label in enumerate(labels)
        if label in TECH_LABELS
    ]
    return sorted(pairs, key=lambda kv: kv[1], reverse=True)


def build_attribute_direction(
    model: LanguageModel,
    tokenizer: Tokenizer,
    positive_prompts: Sequence[str],
    negative_prompts: Sequence[str],
):
    positive = mean_unit([
        encode_prompt(model, tokenizer, p)
        for p in positive_prompts
    ])
    negative = mean_unit([
        encode_prompt(model, tokenizer, p)
        for p in negative_prompts
    ])
    return F.normalize(positive - negative, dim=-1), positive, negative


def projection_score(
    vector: torch.Tensor,
    direction: torch.Tensor,
):
    return float(torch.dot(vector, direction).item())


def print_case_probe(
    name: str,
    prompt: str,
    expected: str,
    model,
    tokenizer,
    centroids,
    intent_head,
    intent_labels,
):
    vector = encode_prompt(model, tokenizer, prompt)
    scores, ranked, margin = cosine_scores(vector, centroids)
    intents = intent_scores(vector, intent_head, intent_labels)

    print(f"{name}: {prompt}")
    print(f"  expected concept : {DISPLAY.get(expected, expected)}")
    print(
        "  nearest centroid : "
        f"{DISPLAY.get(ranked[0][0], ranked[0][0])} "
        f"({ranked[0][1]:.6f})"
    )
    print(
        "  runner-up        : "
        f"{DISPLAY.get(ranked[1][0], ranked[1][0])} "
        f"({ranked[1][1]:.6f})"
    )
    print(f"  top1-top2 margin : {margin:.6f}")
    print(
        "  expected cosine  : "
        f"{scores.get(expected, float('nan')):.6f}"
    )
    print("  all centroids     :")
    for label, score in ranked:
        print(f"    {DISPLAY[label]:12s} {score:.6f}")

    print("  intent-head tech probabilities:")
    for label, score in intents:
        marker = "*" if label == expected else " "
        print(f"   {marker} {DISPLAY[label]:12s} {score:.6f}")

    return vector, ranked


def main():
    args = parse_args()

    for filename in (args.tokenizer, args.model, args.intent_head):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    intent_head, _head_checkpoint, intent_labels = load_intent_head(
        args.intent_head,
        model,
        device,
    )

    model.eval()
    intent_head.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in intent_head.parameters():
        p.requires_grad_(False)

    refs = collect_reference_prompts()
    centroids, vectors = build_centroids(model, tokenizer, refs)

    print()
    print("====================================================")
    print(" LLM_GPU v0.9 Semantic Probe Diagnostic")
    print("====================================================")
    print("Device           :", device)
    if device.type == "cuda":
        print("GPU              :", torch.cuda.get_device_name(0))
    print("Checkpoint loss  :", checkpoint.get("loss"))
    print("Encoder          : frozen v0.8 pairwise-best")
    print("Hidden dimension :", model.d_model)
    print("Concepts         :", ", ".join(DISPLAY[x] for x in TECH_LABELS))
    print()
    print("Reference prompt counts")
    print("-----------------------")
    for label in TECH_LABELS:
        print(f"{DISPLAY[label]:12s}: {len(refs[label])}")
    print()

    # Hard-case concept probes.
    g05_vec, g05_ranked = print_case_probe(
        "G05",
        TARGET_CASES["G05"],
        "tech_cpu",
        model,
        tokenizer,
        centroids,
        intent_head,
        intent_labels,
    )
    print()
    g08_vec, g08_ranked = print_case_probe(
        "G08",
        TARGET_CASES["G08"],
        "tech_transformer",
        model,
        tokenizer,
        centroids,
        intent_head,
        intent_labels,
    )
    print()

    # CPU-vs-GPU property axis.
    cpu_property_positive = [
        "多様な命令を実行する汎用的な演算装置は何ですか。",
        "複雑な命令実行や制御を担当する中心的な装置は何ですか。",
        "幅広い処理を順序立てて実行する装置を答えてください。",
        "汎用処理とシステム制御を担当する演算装置は何ですか。",
    ]
    cpu_property_negative = [
        "大量の同種計算を並列に処理する装置は何ですか。",
        "多数の演算を同時実行することが得意な装置は何ですか。",
        "並列計算で機械学習を高速化する装置は何ですか。",
        "大量の数値計算を同時に処理する演算装置は何ですか。",
    ]
    cpu_axis, _, _ = build_attribute_direction(
        model,
        tokenizer,
        cpu_property_positive,
        cpu_property_negative,
    )

    # Transformer structure/category axis.
    structure_positive = [
        "Attentionを中心に文脈を処理するモデル構造は何ですか。",
        "Self-Attentionを使うニューラルネットワーク構造を答えてください。",
        "Attentionを主要機構とするモデルアーキテクチャは何ですか。",
        "系列をAttentionで処理する代表的なモデル構造は何ですか。",
    ]
    structure_negative = [
        "NVIDIA GPUを汎用計算に使う技術は何ですか。",
        "読みやすい文法を持つ汎用プログラミング言語は何ですか。",
        "大量の計算を並列に処理する演算装置は何ですか。",
        "文章を学習して生成する言語モデルは何ですか。",
    ]
    structure_axis, _, _ = build_attribute_direction(
        model,
        tokenizer,
        structure_positive,
        structure_negative,
    )

    print("Attribute-direction probes")
    print("--------------------------")
    print(
        "G05 CPU-general-vs-GPU-parallel axis : "
        f"{projection_score(g05_vec, cpu_axis):.6f}"
    )
    print(
        "G08 structure/model-category axis     : "
        f"{projection_score(g08_vec, structure_axis):.6f}"
    )
    print()

    # Anchor scores make the sign/magnitude interpretable.
    print("Attribute anchors")
    print("-----------------")
    for name, prompt, direction in [
        (
            "CPU positive",
            "汎用処理や制御を担当するCPUの特徴を説明してください。",
            cpu_axis,
        ),
        (
            "GPU negative",
            "大量並列計算を得意とするGPUの特徴を説明してください。",
            cpu_axis,
        ),
        (
            "Transformer structure positive",
            "TransformerはAttentionを使うモデル構造です。",
            structure_axis,
        ),
        (
            "CUDA/Python category contrast",
            "CUDAはGPU計算技術で、Pythonはプログラミング言語です。",
            structure_axis,
        ),
    ]:
        v = encode_prompt(model, tokenizer, prompt)
        print(f"{name:31s}: {projection_score(v, direction):.6f}")
    print()

    # Optional probe of all held-out technical cases.
    if args.all_tech_cases:
        print("All technical held-out cases")
        print("----------------------------")
        for idx, case in enumerate(CASES, start=1):
            intent = str(case["intent"])
            expected = {
                "gpu": "tech_gpu",
                "cpu": "tech_cpu",
                "llm": "tech_llm",
                "transformer": "tech_transformer",
                "cuda": "tech_cuda",
                "python": "tech_python",
            }.get(intent)
            if expected is None:
                continue
            prompt = str(case["prompt"])
            vector = encode_prompt(model, tokenizer, prompt)
            _, ranked, margin = cosine_scores(vector, centroids)
            ok = ranked[0][0] == expected
            print(
                f"G{idx:02d} {intent:12s} "
                f"nearest={DISPLAY[ranked[0][0]]:12s} "
                f"expected={DISPLAY[expected]:12s} "
                f"margin={margin:.6f} "
                f"{'PASS' if ok else 'MISS'}"
            )
        print()

    # Interpretation hints based only on observed probe results.
    print("Diagnostic interpretation")
    print("-------------------------")
    g05_encoder_ok = g05_ranked[0][0] == "tech_cpu"
    g08_encoder_ok = g08_ranked[0][0] == "tech_transformer"

    if g05_encoder_ok:
        print(
            "G05: nearest centroid is CPU -> encoder concept signal exists; "
            "focus next on semantic-to-generation mapping."
        )
    else:
        print(
            "G05: nearest centroid is not CPU -> encoder representation is "
            "already confused before generation."
        )

    if g08_encoder_ok:
        print(
            "G08: nearest centroid is Transformer -> concept identity is "
            "present in the frozen encoder."
        )
    else:
        print(
            "G08: nearest centroid is not Transformer -> Transformer identity "
            "is weak before generation."
        )

    print(
        "Use the attribute-axis score together with its printed anchors: "
        "a target score near the positive anchor supports presence of the "
        "corresponding property/category signal."
    )


if __name__ == "__main__":
    main()
