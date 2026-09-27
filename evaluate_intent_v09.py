# evaluate_intent_v09.py
#
# Diagnostic evaluator for the v0.9 intent-conditioned generation experiment.
# It does NOT change generation. It measures whether the auxiliary multi-label
# intent head already recognizes the fixed 30 generalization prompts.
#
# Compare:
#   expected intent tags
#   predicted intent tags + probabilities
#   current LM-generated answer
#
# This separates intent-recognition failures from generation failures.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import torch
import torch.nn as nn

from chat import AI_PREFIX, USER_PREFIX, generate_reply
from evaluate_generalization_v07 import CASES
from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat.pt"
DEFAULT_INTENT_HEAD = "model/model-gpu-v0.8-intent-head.pt"


# Expected tags are diagnostic targets for the fixed 30 prompts.
# Only tags that are semantically necessary for the prompt are listed.
# Extra predicted tags are reported separately rather than automatically
# treated as a generation failure.
EXPECTED_TAGS: Dict[str, Set[str]] = {
    # GPU / CPU
    "大量の同時計算に向いているのはGPUですか。何が得意ですか。":
        {"tech_gpu", "property_parallel"},
    "GPUの強みを一文で述べてください。":
        {"tech_gpu"},
    "機械学習でGPUがよく使われる理由は何ですか。":
        {"tech_gpu", "property_parallel"},
    "CPUの得意分野を一文で説明してください。":
        {"tech_cpu"},
    "コンピュータの中心で多様な命令を処理する装置は何ですか。":
        {"tech_cpu", "property_general"},
    "CPUの特徴をGPUと混同しないように説明してください。":
        {"tech_cpu", "tech_gpu", "relation_distinction"},

    # Technical concepts
    "文章を学習して文章を生成するモデルは何ですか。":
        {"tech_llm", "property_language"},
    "Attentionを中心に使う代表的な構造は何ですか。":
        {"tech_transformer", "property_attention"},
    "NVIDIA製GPUで汎用計算を行う仕組みを何と呼びますか。":
        {"tech_cuda", "tech_gpu", "property_general"},
    "読みやすさで知られる汎用言語を一つ挙げてください。":
        {"tech_python", "property_language"},

    # Conversational control
    "回答は一言か二言にしてください。":
        {"control_short"},
    "長い説明はいりません。":
        {"control_short"},
    "別件について話したいです。":
        {"control_topic"},
    "今のテーマはやめて別のテーマへ移りたいです。":
        {"control_topic"},
    "今の説明では理解できませんでした。":
        {"control_repeat"},
    "別の言い方でもう一度お願いします。":
        {"control_repeat"},
    "この会話はここで終わりにしましょう。":
        {"control_end"},
    "続きは次回にします。":
        {"control_end"},

    # Debug / research
    "コード実行に失敗しました。最初に何を見ますか。":
        {"debug_error"},
    "不具合の原因を切り分けたいです。":
        {"debug_error"},
    "二つの実験を公平に比べるにはどうしますか。":
        {"research_compare"},
    "モデルAとBの差を検証したいです。":
        {"research_compare"},

    # General / conversation
    "日本で政府の中心となる都市はどこですか。":
        {"capital"},
    "やあ、調子はどうですか。":
        {"greeting"},
    "少し休みたいくらい疲れています。":
        {"fatigue"},
    "助かりました、感謝します。":
        {"thanks"},

    # Contrast
    "並列計算向きなのはCPUとGPUのどちらですか。":
        {
            "tech_cpu", "tech_gpu", "relation_compare",
            "relation_distinction", "property_parallel",
        },
    "汎用処理を主に担当するのはCPUとGPUのどちらですか。":
        {
            "tech_cpu", "tech_gpu", "relation_compare",
            "relation_distinction", "property_general",
        },
    "CUDAはハードウェアそのものですか。":
        {"tech_cuda", "relation_distinction"},
    "LLMとTransformerは同じ意味ですか。":
        {
            "tech_llm", "tech_transformer",
            "relation_compare", "relation_distinction",
        },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose v0.9 multi-label intent recognition."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT_HEAD)
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold stored in the intent-head checkpoint.",
    )
    p.add_argument("--top-k-tags", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


def build_intent_head(d_model: int, num_labels: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, num_labels),
    )


@torch.no_grad()
def predict_intent(
    model: LanguageModel,
    tokenizer: Tokenizer,
    intent_head: nn.Module,
    labels: Sequence[str],
    prompt_text: str,
    threshold: float,
    top_k: int,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    prompt = f"{USER_PREFIX}{prompt_text}\n{AI_PREFIX}"
    ids = tokenizer.encode(prompt, add_bos=True)

    if len(ids) > model.context_length:
        ids = ids[-model.context_length:]

    device = next(model.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)

    hidden = model.forward_hidden(x)
    prompt_repr = hidden[:, -1, :]
    logits = intent_head(prompt_repr)[0]
    probs = torch.sigmoid(logits)

    predicted = [
        label
        for label, prob in zip(labels, probs.tolist())
        if prob >= threshold
    ]

    k = max(1, min(int(top_k), len(labels)))
    values, indices = torch.topk(probs, k)
    ranked = [
        (labels[int(index)], float(value))
        for value, index in zip(values.tolist(), indices.tolist())
    ]
    return predicted, ranked


def format_tags(tags: Sequence[str]) -> str:
    return ", ".join(tags) if tags else "(none)"


def main() -> None:
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

    head_checkpoint = torch.load(args.intent_head, map_location=device)
    labels = list(head_checkpoint["labels"])
    head_d_model = int(head_checkpoint.get("d_model", model.d_model))

    if head_d_model != model.d_model:
        raise ValueError(
            "Intent-head/model d_model mismatch: "
            f"{head_d_model} != {model.d_model}"
        )

    intent_head = build_intent_head(model.d_model, len(labels)).to(device)
    intent_head.load_state_dict(head_checkpoint["state_dict"])
    intent_head.eval()
    model.eval()

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(head_checkpoint.get("threshold", 0.5))
    )

    missing_targets = sorted({
        tag
        for expected in EXPECTED_TAGS.values()
        for tag in expected
        if tag not in labels
    })
    if missing_targets:
        raise ValueError(
            "Expected tags missing from intent-head labels: "
            + ", ".join(missing_targets)
        )

    print()
    print("======================================")
    print(" LLM_GPU v0.9 Intent Head Diagnostic")
    print("======================================")
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Model loss         :", checkpoint.get("loss"))
    print("Intent-head loss   :", head_checkpoint.get("loss"))
    print("Intent labels      :", len(labels))
    print("Threshold          :", threshold)
    print("Generalization set :", len(CASES))
    print()

    total_expected = 0
    total_predicted = 0
    true_positive = 0
    exact_cases = 0
    full_recall_cases = 0

    per_intent: Dict[str, List[int]] = {}

    for idx, case in enumerate(CASES, start=1):
        prompt_text = str(case["prompt"])
        intent = str(case["intent"])
        expected = EXPECTED_TAGS.get(prompt_text)
        if expected is None:
            raise KeyError(f"No EXPECTED_TAGS entry for: {prompt_text}")

        predicted, ranked = predict_intent(
            model,
            tokenizer,
            intent_head,
            labels,
            prompt_text,
            threshold,
            args.top_k_tags,
        )
        predicted_set = set(predicted)

        tp = len(expected & predicted_set)
        expected_count = len(expected)
        predicted_count = len(predicted_set)

        total_expected += expected_count
        total_predicted += predicted_count
        true_positive += tp

        full_recall = expected.issubset(predicted_set)
        exact_match = expected == predicted_set
        full_recall_cases += int(full_recall)
        exact_cases += int(exact_match)

        stats = per_intent.setdefault(intent, [0, 0])
        stats[0] += int(full_recall)
        stats[1] += 1

        prompt = f"{USER_PREFIX}{prompt_text}\n{AI_PREFIX}"
        reply, _ = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.05,
        )

        missing = sorted(expected - predicted_set)
        extras = sorted(predicted_set - expected)

        print(f"[I{idx:02d}] {intent:14s} 人: {prompt_text}")
        print("      expected : " + format_tags(sorted(expected)))
        print("      predicted: " + format_tags(sorted(predicted_set)))
        print(
            "      top      : "
            + ", ".join(f"{name}={prob:.3f}" for name, prob in ranked)
        )
        print(
            "      expected-recall="
            + ("PASS" if full_recall else "MISS")
            + " | exact="
            + ("PASS" if exact_match else "MISS")
        )
        if missing:
            print("      missing  : " + ", ".join(missing))
        if extras:
            print("      extras   : " + ", ".join(extras))
        print("      AI       : " + reply)

    precision = (
        true_positive / total_predicted
        if total_predicted
        else 0.0
    )
    recall = (
        true_positive / total_expected
        if total_expected
        else 0.0
    )
    micro_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    print()
    print("Summary")
    print("-------")
    print(
        f"Expected-tag case recall : {full_recall_cases}/{len(CASES)} "
        f"({full_recall_cases / len(CASES):.1%})"
    )
    print(
        f"Exact tag-set match      : {exact_cases}/{len(CASES)} "
        f"({exact_cases / len(CASES):.1%})"
    )
    print(f"Micro precision          : {precision:.1%}")
    print(f"Micro recall             : {recall:.1%}")
    print(f"Micro F1                 : {micro_f1:.1%}")

    print()
    print("Per-intent expected-tag recall")
    print("------------------------------")
    for intent in sorted(per_intent):
        ok, n = per_intent[intent]
        print(f"{intent:14s}: {ok}/{n} ({ok / n:.1%})")

    print()
    print("Interpretation")
    print("--------------")
    print(
        "If expected-tag recall is high but the generated answer is wrong, "
        "the likely bottleneck is the missing intent-to-generation path."
    )
    print(
        "If both intent tags and generation are wrong, improve the intent "
        "representation/training before adding intent-conditioned generation."
    )


if __name__ == "__main__":
    main()
