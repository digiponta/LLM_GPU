# evaluate_alpha_sweep_v09.py
#
# Automatic alpha sweep for v0.9 soft Intent-Conditioned Generation.
# Reuses ONE trained projection checkpoint and changes only inference-time alpha.
# The same fixed 30 generalization prompts and scoring rules are used for every
# alpha value so the effect of conditioning strength can be compared directly.

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from chat_v09 import AI_PREFIX, USER_PREFIX, generate_reply
from evaluate_generalization_v09 import (
    CASES,
    dimension_match,
    semantic_match,
)
from intent_conditioning_v09 import load_intent_head, load_projection
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INTENT_HEAD = "model/model-gpu-v0.8-intent-head.pt"
DEFAULT_PROJECTION = "model/model-gpu-v0.9-soft-intent.pt"
DEFAULT_ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep inference-time alpha for v0.9 soft intent conditioning."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT_HEAD)
    p.add_argument("--projection", default=DEFAULT_PROJECTION)
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Inference-time alpha values to evaluate.",
    )
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument(
        "--show-cases",
        action="store_true",
        help="Print all 30 replies for every alpha.",
    )
    return p.parse_args()


@torch.no_grad()
def evaluate_alpha(
    alpha: float,
    model,
    tokenizer,
    intent_head,
    projection,
    max_new_tokens: int,
    show_cases: bool,
) -> Dict[str, object]:
    projection.alpha = float(alpha)

    count = len(CASES)
    semantic_pass = 0
    entity_pass = 0
    entity_total = 0
    entity_na = 0
    fluent_pass = 0
    strict_pass = 0
    legacy_pass = 0
    per_intent = defaultdict(lambda: {"semantic": 0, "strict": 0, "n": 0})
    case_results: List[Dict[str, object]] = []

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        prompt = f"{USER_PREFIX}{prompt_text}\n{AI_PREFIX}"

        reply, _, _ = generate_reply(
            model=model,
            tokenizer=tokenizer,
            intent_head=intent_head,
            projection=projection,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.05,
        )

        legacy_ok, _, _ = semantic_match(
            reply,
            case["required_all"],
            case["forbidden"],
        )
        intent = str(case["intent"])
        dims = dimension_match(
            reply,
            prompt_text,
            intent,
            case["required_all"],
            case["forbidden"],
        )

        semantic_pass += int(dims["semantic_ok"])
        fluent_pass += int(dims["fluent_ok"])
        strict_pass += int(dims["strict_ok"])
        legacy_pass += int(legacy_ok)

        if dims["entity_ok"] is None:
            entity_na += 1
        else:
            entity_total += 1
            entity_pass += int(dims["entity_ok"])

        stats = per_intent[intent]
        stats["semantic"] += int(dims["semantic_ok"])
        stats["strict"] += int(dims["strict_ok"])
        stats["n"] += 1

        case_results.append({
            "index": idx,
            "intent": intent,
            "prompt": prompt_text,
            "reply": reply,
            "semantic_ok": bool(dims["semantic_ok"]),
            "strict_ok": bool(dims["strict_ok"]),
        })

        if show_cases:
            print(
                f"[A={alpha:g} G{idx:02d}] {intent:14s} "
                f"semantic={'PASS' if dims['semantic_ok'] else 'MISS'} "
                f"strict={'PASS' if dims['strict_ok'] else 'MISS'}"
            )
            print(f"      AI: {reply}")

    return {
        "alpha": float(alpha),
        "semantic": semantic_pass,
        "entity_pass": entity_pass,
        "entity_total": entity_total,
        "entity_na": entity_na,
        "fluency": fluent_pass,
        "strict": strict_pass,
        "legacy": legacy_pass,
        "count": count,
        "per_intent": dict(per_intent),
        "cases": case_results,
    }


def pct(value: int, total: int) -> str:
    if total <= 0:
        return "N/A"
    return f"{value}/{total} ({value / total:.1%})"


def main() -> None:
    args = parse_args()

    for filename in (
        args.tokenizer,
        args.model,
        args.intent_head,
        args.projection,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    if not args.alphas:
        raise ValueError("--alphas must contain at least one value.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.model, device=device
    )
    intent_head, head_checkpoint, labels = load_intent_head(
        args.intent_head, model, device
    )
    projection, projection_checkpoint = load_projection(
        args.projection, model, labels, device
    )

    trained_alpha = float(projection_checkpoint.get("alpha", projection.alpha))

    print()
    print("====================================")
    print(" LLM_GPU v0.9 Alpha Sweep")
    print("====================================")
    print("Device           :", device)
    if device.type == "cuda":
        print("GPU              :", torch.cuda.get_device_name(0))
    print("Base loss        :", checkpoint.get("loss"))
    print("Intent-head loss :", head_checkpoint.get("loss"))
    print("Projection loss  :", projection_checkpoint.get("loss"))
    print("Trained alpha    :", trained_alpha)
    print("Sweep alphas     :", ", ".join(f"{x:g}" for x in args.alphas))
    print("Held-out cases   :", len(CASES))
    print()

    results = []
    for alpha in args.alphas:
        results.append(
            evaluate_alpha(
                alpha=float(alpha),
                model=model,
                tokenizer=tokenizer,
                intent_head=intent_head,
                projection=projection,
                max_new_tokens=args.max_new_tokens,
                show_cases=args.show_cases,
            )
        )

    print("Overall comparison")
    print("------------------")
    print(
        f"{'alpha':>6}  {'semantic':>17}  {'strict':>17}  "
        f"{'entity':>17}  {'fluency':>17}  {'legacy':>17}"
    )
    for r in results:
        print(
            f"{r['alpha']:6.2f}  "
            f"{pct(r['semantic'], r['count']):>17}  "
            f"{pct(r['strict'], r['count']):>17}  "
            f"{pct(r['entity_pass'], r['entity_total']):>17}  "
            f"{pct(r['fluency'], r['count']):>17}  "
            f"{pct(r['legacy'], r['count']):>17}"
        )

    baseline = results[0]
    if baseline["alpha"] != 0.0:
        zero = next((r for r in results if r["alpha"] == 0.0), None)
        if zero is not None:
            baseline = zero

    print()
    print("Changes versus alpha=0.0")
    print("------------------------")
    base_cases = {
        int(c["index"]): c
        for c in baseline["cases"]
    }
    for r in results:
        if r is baseline:
            continue
        improved = []
        regressed = []
        reply_changed = []
        for case in r["cases"]:
            idx = int(case["index"])
            base = base_cases[idx]
            if (not base["strict_ok"]) and case["strict_ok"]:
                improved.append(f"G{idx:02d}")
            if base["strict_ok"] and (not case["strict_ok"]):
                regressed.append(f"G{idx:02d}")
            if base["reply"] != case["reply"]:
                reply_changed.append(f"G{idx:02d}")

        print(
            f"alpha={r['alpha']:g}: "
            f"strict delta={r['strict'] - baseline['strict']:+d}, "
            f"improved={','.join(improved) if improved else '-'}, "
            f"regressed={','.join(regressed) if regressed else '-'}, "
            f"reply-changed={','.join(reply_changed) if reply_changed else '-'}"
        )

    all_intents = sorted({
        intent
        for r in results
        for intent in r["per_intent"]
    })

    print()
    print("Per-intent strict comparison")
    print("----------------------------")
    header = "intent".ljust(16) + "".join(
        f"a={r['alpha']:g}".rjust(12)
        for r in results
    )
    print(header)
    for intent in all_intents:
        row = intent.ljust(16)
        for r in results:
            stats = r["per_intent"].get(intent, {"strict": 0, "n": 0})
            cell = (
                f"{stats['strict']}/{stats['n']}"
                if stats["n"]
                else "-"
            )
            row += cell.rjust(12)
        print(row)

    best_strict = max(r["strict"] for r in results)
    best_semantic = max(r["semantic"] for r in results)
    best_strict_alphas = [
        r["alpha"] for r in results if r["strict"] == best_strict
    ]
    best_semantic_alphas = [
        r["alpha"] for r in results if r["semantic"] == best_semantic
    ]

    print()
    print("Best observed")
    print("-------------")
    print(
        "Strict   : "
        + pct(best_strict, len(CASES))
        + " at alpha="
        + ", ".join(f"{x:g}" for x in best_strict_alphas)
    )
    print(
        "Semantic : "
        + pct(best_semantic, len(CASES))
        + " at alpha="
        + ", ".join(f"{x:g}" for x in best_semantic_alphas)
    )
    print()
    print(
        "Note: this sweep changes inference-time alpha only. "
        "The projection weights are not retrained for each alpha."
    )


if __name__ == "__main__":
    main()
