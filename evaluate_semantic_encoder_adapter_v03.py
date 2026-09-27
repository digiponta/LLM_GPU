# evaluate_semantic_encoder_adapter_v03.py
#
# Evaluate Semantic Encoder Adapter v0.3 geometry before generation integration.

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v03 import (
    ATTRIBUTE_LABELS,
    CONCEPT_LABELS,
    load_semantic_adapter_v03_checkpoint,
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
DEFAULT_ADAPTER = "model/model-gpu-v0.9-semantic-adapter-v03.pt"
USER_PREFIX = "人: "
AI_PREFIX = "AI: "


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Semantic Encoder Adapter v0.3."
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
            v = (
                encode_raw(model, tokenizer, prompt)
                if adapter is None
                else encode_adapted(model, tokenizer, adapter, prompt)
            )
            vectors.append(F.normalize(v, dim=-1))
        centroids[label] = mean_unit(vectors)
    return centroids


def concept_probabilities(head, adapted):
    probs = torch.softmax(head(adapted.unsqueeze(0))[0], dim=-1)[0]
    return sorted(
        [
            (label, float(probs[i].item()))
            for i, label in enumerate(CONCEPT_LABELS)
        ],
        key=lambda kv: kv[1],
        reverse=True,
    )


def attribute_probabilities(head, adapted):
    probs = torch.sigmoid(head(adapted.unsqueeze(0))[1][0])
    return [
        (label, float(probs[i].item()))
        for i, label in enumerate(ATTRIBUTE_LABELS)
    ]


def print_case(
    name,
    prompt,
    expected,
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

    _rs, raw_ranked, raw_margin = cosine_scores(raw_n, raw_centroids)
    _as, adapted_ranked, adapted_margin = cosine_scores(
        adapted_n,
        adapted_centroids,
    )

    drift = 1.0 - float(
        F.cosine_similarity(
            raw.unsqueeze(0),
            adapted.unsqueeze(0),
            dim=-1,
        ).item()
    )

    concept_ranked = concept_probabilities(heads, adapted)
    attrs = attribute_probabilities(heads, adapted)

    print(f"{name}: {prompt}")
    print(f"  expected              : {DISPLAY[expected]}")
    print(
        f"  raw nearest           : {DISPLAY[raw_ranked[0][0]]} "
        f"({raw_ranked[0][1]:.6f})"
    )
    print(
        f"  raw runner-up         : {DISPLAY[raw_ranked[1][0]]} "
        f"({raw_ranked[1][1]:.6f})"
    )
    print(f"  raw margin            : {raw_margin:.6f}")
    print(
        f"  adapted nearest       : {DISPLAY[adapted_ranked[0][0]]} "
        f"({adapted_ranked[0][1]:.6f})"
    )
    print(
        f"  adapted runner-up     : {DISPLAY[adapted_ranked[1][0]]} "
        f"({adapted_ranked[1][1]:.6f})"
    )
    print(f"  adapted margin        : {adapted_margin:.6f}")
    print(f"  cosine drift          : {drift:.6f}")
    print("  supervised concept probabilities:")
    for label, score in concept_ranked:
        marker = "*" if label == expected else " "
        print(f"   {marker} {DISPLAY[label]:12s} {score:.6f}")
    print("  supervised attribute probabilities:")
    for label, score in attrs:
        print(f"     {label:30s} {score:.6f}")

    return {
        "raw_top": raw_ranked[0][0],
        "adapted_top": adapted_ranked[0][0],
        "raw_margin": raw_margin,
        "adapted_margin": adapted_margin,
        "drift": drift,
    }


def heldout_technical_accuracy(
    model,
    tokenizer,
    centroids,
    adapter=None,
):
    intent_to_label = {
        "gpu": "tech_gpu",
        "cpu": "tech_cpu",
        "llm": "tech_llm",
        "transformer": "tech_transformer",
        "cuda": "tech_cuda",
        "python": "tech_python",
    }

    correct = total = 0
    rows = []

    for idx, case in enumerate(CASES, start=1):
        expected = intent_to_label.get(str(case["intent"]))
        if expected is None:
            continue

        prompt = str(case["prompt"])
        vector = (
            encode_raw(model, tokenizer, prompt)
            if adapter is None
            else encode_adapted(model, tokenizer, adapter, prompt)
        )
        vector = F.normalize(vector, dim=-1)
        _scores, ranked, margin = cosine_scores(vector, centroids)
        ok = ranked[0][0] == expected
        correct += int(ok)
        total += 1
        rows.append((
            idx,
            str(case["intent"]),
            expected,
            ranked[0][0],
            margin,
            ok,
        ))

    return correct, total, rows


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
    adapter, heads, checkpoint = load_semantic_adapter_v03_checkpoint(
        args.adapter,
        device,
    )

    model.eval()
    adapter.eval()
    heads.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    raw_centroids = build_centroids(model, tokenizer, None)
    adapted_centroids = build_centroids(model, tokenizer, adapter)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.3 Evaluation")
    print("====================================================")
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", base_checkpoint.get("loss"))
    print("Adapter val loss    :", checkpoint.get("loss"))
    print("Adapter hidden      :", adapter.hidden_dim)
    print("Residual scale      :", adapter.residual_scale)
    print("Centroid margin     :", checkpoint.get("centroid_margin"))
    print("Centroid weight     :", checkpoint.get("centroid_margin_weight"))
    print("Pairwise margin     :", checkpoint.get("pairwise_margin"))
    print("Pairwise weight     :", checkpoint.get("pairwise_weight"))
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

    raw_correct, total, raw_rows = heldout_technical_accuracy(
        model,
        tokenizer,
        raw_centroids,
        None,
    )
    adapted_correct, _total2, adapted_rows = heldout_technical_accuracy(
        model,
        tokenizer,
        adapted_centroids,
        adapter,
    )

    print("Technical held-out centroid accuracy")
    print("------------------------------------")
    print(f"Raw     : {raw_correct}/{total} ({raw_correct / total:.1%})")
    print(
        f"Adapted : {adapted_correct}/{total} "
        f"({adapted_correct / total:.1%})"
    )
    print()

    print("Technical held-out cases")
    print("------------------------")
    for raw_row, adapted_row in zip(raw_rows, adapted_rows):
        idx, intent, expected, raw_top, raw_margin, _ = raw_row
        _idx2, _intent2, _expected2, adapted_top, adapted_margin, ok = adapted_row
        print(
            f"G{idx:02d} {intent:12s} "
            f"{DISPLAY[raw_top]:12s} -> {DISPLAY[adapted_top]:12s} "
            f"expected={DISPLAY[expected]:12s} "
            f"margin={adapted_margin:.6f} "
            f"{'PASS' if ok else 'MISS'}"
        )
    print()

    print("Acronym / full-name probe")
    print("-------------------------")
    acronym_cases = [
        ("CPU full name", "Central Processing Unitという正式名称に対応する略称は何でしょうか。", "tech_cpu"),
        ("CPU meaning", "Central Processing Unitが担う代表的な役割を名称とともに考えてください。", "tech_cpu"),
        ("GPU full name", "Graphics Processing Unitという正式名称の略称を答えてください。", "tech_gpu"),
        ("GPU meaning", "Graphics Processing Unitの現在の計算上の強みは何でしょうか。", "tech_gpu"),
    ]
    for name, prompt, expected in acronym_cases:
        vector = F.normalize(
            encode_adapted(model, tokenizer, adapter, prompt),
            dim=-1,
        )
        _scores, ranked, margin = cosine_scores(vector, adapted_centroids)
        print(
            f"{name:14s}: nearest={DISPLAY[ranked[0][0]]:12s} "
            f"expected={DISPLAY[expected]:12s} "
            f"margin={margin:.6f} "
            f"{'PASS' if ranked[0][0] == expected else 'MISS'}"
        )
    print()

    print("Diagnostic summary")
    print("------------------")
    print(
        "G05 centroid binding : "
        f"{DISPLAY[g05['raw_top']]} -> {DISPLAY[g05['adapted_top']]}"
    )
    print(
        "G08 centroid binding : "
        f"{DISPLAY[g08['raw_top']]} -> {DISPLAY[g08['adapted_top']]}"
    )
    print(
        "Held-out tech        : "
        f"{raw_correct}/{total} -> {adapted_correct}/{total}"
    )

    if (
        g05["adapted_top"] == "tech_cpu"
        and g08["adapted_top"] == "tech_transformer"
        and adapted_correct >= raw_correct
    ):
        print(
            "Result: semantic geometry meets the primary integration gate. "
            "Generation integration can be tested next."
        )
    else:
        print(
            "Result: semantic geometry does not yet meet the integration gate. "
            "Do not connect this adapter to generation yet."
        )


if __name__ == "__main__":
    main()
