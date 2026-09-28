# evaluate_calibrated_multi_concept_binding_v0119.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from technical_intent_calibrator_v0119 import (
    TECH_LABELS,
    extract_technical_logits,
    load_checkpoint as load_calibrator,
)
from multi_concept_safe_binding_v0118 import (
    CONCEPTS,
    CANONICAL,
    load_checkpoint as load_binding,
)
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from evaluate_generalization_v07 import CASES, dimension_match
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_ROLE = "model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt"
DEFAULT_BINDING = "model/model-gpu-v0.11.8-multi-concept-safe-binding.pt"
DEFAULT_CALIBRATOR = "model/model-gpu-v0.11.9-technical-intent-calibrator.pt"

DIRECT = {
    5: "CPU",
    7: "LLM",
    8: "Transformer",
    9: "CUDA",
    10: "Python",
    27: "GPU",
    28: "CPU",
}


def token_text(tok, token_id: int) -> str:
    try:
        return tok.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")
    except Exception:
        return ""


@torch.no_grad()
def generate(
    model,
    tokenizer,
    intent_head,
    intent_labels,
    role_head,
    binding,
    binding_ck,
    calibrator,
    prompt_text,
):
    prompt = f"人: {prompt_text}\nAI: "
    generated = tokenizer.encode(prompt, add_bos=True)
    response = []
    device = next(model.parameters()).device

    cpu_idx = intent_labels.index("tech_cpu")
    gpu_idx = intent_labels.index("tech_gpu")

    confidence_threshold = float(binding_ck.get("technical_confidence_threshold", 0.70))
    pair_intent_margin = float(binding_ck.get("cpu_gpu_intent_margin", 0.10))
    pair_role_margin = float(binding_ck.get("cpu_gpu_role_margin", 0.05))

    debug = None

    for step in range(96):
        context = generated[-model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        hidden = model.forward_hidden(x)
        h = hidden[:, -1, :]
        logits = model.lm_head(h)[0].clone()

        if step == 0:
            intent_logits = intent_head(h)
            ip = torch.sigmoid(intent_logits)
            rp = torch.sigmoid(role_head(h))

            cpu = float(ip[0, cpu_idx].item())
            gpu = float(ip[0, gpu_idx].item())
            controller = float(rp[0, 0].item())
            executor = float(rp[0, 1].item())

            raw_tech_logits = extract_technical_logits(intent_logits, intent_labels)
            calibrated_logits = calibrator(raw_tech_logits)
            calibrated_prob = torch.softmax(calibrated_logits, dim=-1)[0]

            top_index = int(torch.argmax(calibrated_prob).item())
            top_label = TECH_LABELS[top_index]
            top_score = float(calibrated_prob[top_index].item())
            raw_top_index = int(torch.argmax(raw_tech_logits[0]).item())
            raw_top_label = TECH_LABELS[raw_top_index]

            concept_idx = CONCEPTS.index(top_label)
            canonical = CANONICAL[top_label]
            target_id = int(binding_ck["token_ids"][top_label])

            pair_gate = False
            concept_gate = False

            if top_label == "tech_cpu":
                pair_gate = (
                    cpu > gpu
                    and (cpu - gpu) >= pair_intent_margin
                    and controller > executor
                    and (controller - executor) >= pair_role_margin
                )
            elif top_label == "tech_gpu":
                pair_gate = (
                    gpu > cpu
                    and (gpu - cpu) >= pair_intent_margin
                    and executor > controller
                    and (executor - controller) >= pair_role_margin
                )
            else:
                concept_gate = (
                    top_score >= confidence_threshold
                    and canonical.lower() not in prompt_text.lower()
                )

            active = pair_gate or concept_gate

            base_top_id = int(torch.argmax(logits).item())
            base_target = float(logits[target_id].item())

            boost = 0.0
            if active:
                features = torch.cat([ip[0], rp[0]], dim=-1).unsqueeze(0)
                cidx = torch.tensor([concept_idx], dtype=torch.long, device=device)
                boost = float(binding(features, cidx)[0].item())
                logits[target_id] += boost

            after_top_id = int(torch.argmax(logits).item())
            after_target = float(logits[target_id].item())

            debug = {
                "cpu": cpu,
                "gpu": gpu,
                "controller": controller,
                "executor": executor,
                "raw_top_label": raw_top_label,
                "top_label": top_label,
                "top_score": top_score,
                "canonical": canonical,
                "active": active,
                "target_id": target_id,
                "boost": boost,
                "base_top_id": base_top_id,
                "after_top_id": after_top_id,
                "base_target": base_target,
                "after_target": after_target,
                "after_top_logit": float(logits[after_top_id].item()),
            }

        if response:
            for token_id in set(response):
                if logits[token_id] >= 0:
                    logits[token_id] /= 1.05
                else:
                    logits[token_id] *= 1.05

        next_id = int(torch.argmax(logits).item())
        if next_id == tokenizer.eos_id:
            break

        generated.append(next_id)
        response.append(next_id)

        decoded = tokenizer.decode(response, skip_special_tokens=True)
        if "\n" in decoded:
            break

    reply = tokenizer.decode(response, skip_special_tokens=True).split("\n", 1)[0].strip()
    return reply, debug


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--role-checkpoint", default=DEFAULT_ROLE)
    p.add_argument("--binding", default=DEFAULT_BINDING)
    p.add_argument("--calibrator", default=DEFAULT_CALIBRATOR)
    args = p.parse_args()

    for filename in (
        args.tokenizer,
        args.model,
        args.intent_head,
        args.role_checkpoint,
        args.binding,
        args.calibrator,
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ck = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()

    intent_head, intent_ck, intent_labels = load_intent_head(args.intent_head, model, device)
    role_head, _old_binding, role_ck = load_role_checkpoint(
        args.role_checkpoint, intent_labels, device
    )
    binding, binding_ck = load_binding(args.binding, intent_labels, device)
    calibrator, calibrator_ck = load_calibrator(args.calibrator, intent_labels, device)

    print("====================================================")
    print(" Semantic Intent Repair / Calibration v0.11.9 Evaluation")
    print("====================================================")
    print("Device:", device)
    print("Base loss:", base_ck.get("loss"))
    print("Binding loss:", binding_ck.get("loss"))
    print("Calibrator loss:", calibrator_ck.get("loss"))
    print("Binding confidence gate:", binding_ck.get("technical_confidence_threshold"))
    print("Technical labels:", ", ".join(TECH_LABELS))

    semantic = strict = fluency = 0
    direct_hits = 0

    for idx, case in enumerate(CASES, start=1):
        prompt = str(case["prompt"])
        reply, dbg = generate(
            model,
            tokenizer,
            intent_head,
            intent_labels,
            role_head,
            binding,
            binding_ck,
            calibrator,
            prompt,
        )

        dims = dimension_match(
            reply,
            prompt,
            str(case["intent"]),
            case["required_all"],
            case["forbidden"],
        )

        semantic += int(dims["semantic_ok"])
        strict += int(dims["strict_ok"])
        fluency += int(dims["fluent_ok"])

        print(f"[G{idx:02d}] {str(case['intent']):14s} 人: {prompt}")
        print("      AI:", reply)
        print(
            "      semantic-content="
            + ("PASS" if dims["semantic_ok"] else "MISS")
            + " | strict="
            + ("PASS" if dims["strict_ok"] else "MISS")
        )
        print(
            f"      raw-top={dbg['raw_top_label']} "
            f"cal-top={dbg['top_label']}:{dbg['top_score']:.3f} "
            f"concept={dbg['canonical']} "
            f"binding={'ON' if dbg['active'] else 'OFF'}"
        )

        if dbg["active"]:
            print(
                f"      target={token_text(tokenizer, dbg['target_id'])!r} "
                f"boost={dbg['boost']:+.3f} "
                f"base-winner={token_text(tokenizer, dbg['base_top_id'])!r} "
                f"after-winner={token_text(tokenizer, dbg['after_top_id'])!r}"
            )

        if idx in DIRECT:
            expected = DIRECT[idx]
            ok = reply.startswith(expected)
            direct_hits += int(ok)
            print(
                "      direct-entity="
                + ("PASS" if ok else "MISS")
                + " | expected="
                + expected
            )

    print()
    print("Summary")
    print("-------")
    print(f"Semantic-content rate : {semantic}/30 ({semantic/30:.1%})")
    print(f"Fluency rate          : {fluency}/30 ({fluency/30:.1%})")
    print(f"Strict composite rate : {strict}/30 ({strict/30:.1%})")
    print(f"Tracked direct rate   : {direct_hits}/{len(DIRECT)} ({direct_hits/len(DIRECT):.1%})")


if __name__ == "__main__":
    main()
