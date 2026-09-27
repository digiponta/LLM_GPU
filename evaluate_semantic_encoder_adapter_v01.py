# evaluate_semantic_encoder_adapter_v01.py
#
# Evaluate Semantic Encoder Adapter v0.1 without generation fine-tuning.
#
# Reports before/after centroid geometry, supervised concept probabilities,
# attribute probabilities, and representation drift for G05/G08.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic_encoder_adapter_v01 import (
    ATTRIBUTE_LABELS,
    CONCEPT_LABELS,
    load_semantic_adapter_checkpoint,
)
from semantic_probe_v09 import (
    DISPLAY,
    TARGET_CASES,
    collect_reference_prompts,
    cosine_scores,
    mean_unit,
)
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_ADAPTER = "model/model-gpu-v0.9-semantic-adapter-v01.pt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Semantic Encoder Adapter v0.1."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    return p.parse_args()


@torch.no_grad()
def encode_raw(model, tokenizer, text):
    device = next(model.parameters()).device
    prompt = f"{USER_PREFIX}{text}\n{AI_PREFIX}"
    ids = tokenizer.encode(prompt, add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = model.forward_hidden(x)
    return hidden[:, -1, :][0]


@torch.no_grad()
def encode_adapted(model, tokenizer, adapter, text):
    raw = encode_raw(model, tokenizer, text)
    return adapter(raw.unsqueeze(0))[0]


def build_centroids(model, tokenizer, adapter=None):
    refs = collect_reference_prompts()
    centroids = {}

    for label in CONCEPT_LABELS:
        vectors = []
        for prompt in refs[label]:
            if adapter is None:
                v = encode_raw(model, tokenizer, prompt)
            else:
                v = encode_adapted(model, tokenizer, adapter, prompt)
            vectors.append(F.normalize(v, dim=-1))
        centroids[label] = mean_unit(vectors)

    return centroids


def probability_table(logits, labels: Sequence[str]):
    probs = torch.softmax(logits, dim=-1)
    ranked = sorted(
        [
            (label, float(probs[i].item()))
            for i, label in enumerate(labels)
        ],
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked


def attribute_table(logits):
    probs = torch.sigmoid(logits)
    return [
        (label, float(probs[i].item()))
        for i, label in enumerate(ATTRIBUTE_LABELS)
    ]


def print_case(
    case_name,
    prompt,
    expected_label,
    model,
    tokenizer,
    adapter,
    heads,
    raw_centroids,
    adapted_centroids,
):
    raw = encode_raw(model, tokenizer, prompt)
    adapted = adapter(raw.unsqueeze(0))[0]

    raw_n = F.normalize(raw, dim=-1)
    adapted_n = F.normalize(adapted, dim=-1)

    _raw_scores, raw_ranked, raw_margin = cosine_scores(
        raw_n,
        raw_centroids,
    )
    _adapted_scores, adapted_ranked, adapted_margin = cosine_scores(
        adapted_n,
        adapted_centroids,
    )

    concept_logits, attribute_logits = heads(adapted.unsqueeze(0))
    concept_ranked = probability_table(
        concept_logits[0],
        CONCEPT_LABELS,
    )
    attrs = attribute_table(attribute_logits[0])

    drift = 1.0 - float(
        F.cosine_similarity(
            raw.unsqueeze(0),
            adapted.unsqueeze(0),
            dim=-1,
        ).item()
    )

    print(f"{case_name}: {prompt}")
    print(f"  expected              : {DISPLAY[expected_label]}")
    print(
        "  raw nearest           : "
        f"{DISPLAY[raw_ranked[0][0]]} ({raw_ranked[0][1]:.6f})"
    )
    print(
        "  raw runner-up         : "
        f"{DISPLAY[raw_ranked[1][0]]} ({raw_ranked[1][1]:.6f})"
    )
    print(f"  raw margin            : {raw_margin:.6f}")
    print(
        "  adapted nearest       : "
        f"{DISPLAY[adapted_ranked[0][0]]} ({adapted_ranked[0][1]:.6f})"
    )
    print(
        "  adapted runner-up     : "
        f"{DISPLAY[adapted_ranked[1][0]]} ({adapted_ranked[1][1]:.6f})"
    )
    print(f"  adapted margin        : {adapted_margin:.6f}")
    print(f"  cosine drift          : {drift:.6f}")
    print("  supervised concept probabilities:")
    for label, score in concept_ranked:
        marker = "*" if label == expected_label else " "
        print(f"   {marker} {DISPLAY[label]:12s} {score:.6f}")
    print("  supervised attribute probabilities:")
    for label, score in attrs:
        print(f"     {label:30s} {score:.6f}")

    return {
        "raw_top": raw_ranked[0][0],
        "adapted_top": adapted_ranked[0][0],
        "concept_top": concept_ranked[0][0],
        "raw_margin": raw_margin,
        "adapted_margin": adapted_margin,
        "drift": drift,
    }


def main():
    args = parse_args()

    for filename in (args.tokenizer, args.model, args.adapter):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    adapter, heads, adapter_checkpoint = load_semantic_adapter_checkpoint(
        args.adapter,
        device,
    )

    model.eval()
    adapter.eval()
    heads.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    raw_centroids = build_centroids(model, tokenizer, adapter=None)
    adapted_centroids = build_centroids(model, tokenizer, adapter=adapter)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.1 Evaluation")
    print("====================================================")
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", base_checkpoint.get("loss"))
    print("Adapter val loss    :", adapter_checkpoint.get("loss"))
    print("Adapter hidden      :", adapter.hidden_dim)
    print("Residual scale      :", adapter.residual_scale)
    print("Concept classes     :", len(CONCEPT_LABELS))
    print("Attribute targets   :", len(ATTRIBUTE_LABELS))
    print()

    g05 = print_case(
        "G05",
        TARGET_CASES["G05"],
        "tech_cpu",
        model,
        tokenizer,
        adapter,
        heads,
        raw_centroids,
        adapted_centroids,
    )
    print()
    g08 = print_case(
        "G08",
        TARGET_CASES["G08"],
        "tech_transformer",
        model,
        tokenizer,
        adapter,
        heads,
        raw_centroids,
        adapted_centroids,
    )
    print()

    print("Diagnostic summary")
    print("------------------")
    print(
        "G05 centroid binding : "
        f"{DISPLAY[g05['raw_top']]} -> {DISPLAY[g05['adapted_top']]}"
    )
    print(
        "G05 concept head     : "
        f"{DISPLAY[g05['concept_top']]}"
    )
    print(
        "G08 centroid binding : "
        f"{DISPLAY[g08['raw_top']]} -> {DISPLAY[g08['adapted_top']]}"
    )
    print(
        "G08 concept head     : "
        f"{DISPLAY[g08['concept_top']]}"
    )

    if g05["adapted_top"] == "tech_cpu":
        print(
            "G05: adapter corrected centroid binding to CPU. "
            "Generation integration is justified."
        )
    else:
        print(
            "G05: centroid binding is still not CPU. "
            "Strengthen adapter supervision before generation integration."
        )

    if g08["adapted_top"] == "tech_transformer":
        print(
            "G08: Transformer identity remains present after adaptation; "
            "inspect property_attention_structure probability above."
        )
    else:
        print(
            "G08: Transformer identity regressed after adaptation; "
            "do not integrate this adapter into generation yet."
        )


if __name__ == "__main__":
    main()
