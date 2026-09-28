# evaluate_llm_scope_rescue_v01112.py
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from selective_intent_repair_v01110 import (
    TECH_LABELS,
    extract_technical_logits,
    load_checkpoint as load_repair,
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
DEFAULT_REPAIR = "model/model-gpu-v0.11.10-selective-intent-repair.pt"

REPAIR_MARGIN = 0.20
SCOPE_THRESHOLD = 0.50
BINDING_CONFIDENCE = 0.45
REPAIR_CONFIDENCE = 0.55
LLM_RESCUE_MARGIN = 0.20
LLM_RESCUE_CONFIDENCE = 0.70

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
def generate(model, tokenizer, intent_head, intent_labels, role_head, binding, binding_ck, repair, prompt_text):
    prompt = f"人: {prompt_text}\nAI: "
    generated = tokenizer.encode(prompt, add_bos=True)
    response = []
    device = next(model.parameters()).device

    cpu_idx = intent_labels.index("tech_cpu")
    gpu_idx = intent_labels.index("tech_gpu")
    pair_intent_margin = float(binding_ck.get("cpu_gpu_intent_margin", 0.10))
    pair_role_margin = float(binding_ck.get("cpu_gpu_role_margin", 0.05))
    debug = None

    for step in range(96):
        context = generated[-model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        h = model.forward_hidden(x)[:, -1, :]
        logits = model.lm_head(h)[0].clone()

        if step == 0:
            intent_logits = intent_head(h)
            ip = torch.sigmoid(intent_logits)
            rp = torch.sigmoid(role_head(h))

            raw_tech_logits = extract_technical_logits(intent_logits, intent_labels)
            repaired_logits, scope_logit = repair(raw_tech_logits)

            raw_scores = torch.sigmoid(raw_tech_logits)[0]
            repaired_probs = torch.softmax(repaired_logits, dim=-1)[0]
            scope_prob = float(torch.sigmoid(scope_logit)[0].item())

            raw_order = torch.argsort(raw_tech_logits[0], descending=True)
            raw_top_i = int(raw_order[0].item())
            raw_second_i = int(raw_order[1].item())
            raw_top_label = TECH_LABELS[raw_top_i]
            raw_margin = float((raw_tech_logits[0, raw_top_i] - raw_tech_logits[0, raw_second_i]).item())

            repaired_top_i = int(torch.argmax(repaired_probs).item())
            repaired_top_label = TECH_LABELS[repaired_top_i]
            repaired_top_conf = float(repaired_probs[repaired_top_i].item())

            chosen_i = raw_top_i
            repair_applied = False
            if (
                raw_margin <= REPAIR_MARGIN
                and repaired_top_i != raw_top_i
                and repaired_top_conf >= REPAIR_CONFIDENCE
            ):
                chosen_i = repaired_top_i
                repair_applied = True

            chosen_label = TECH_LABELS[chosen_i]
            canonical = CANONICAL[chosen_label]
            concept_idx = CONCEPTS.index(chosen_label)
            target_id = int(binding_ck["token_ids"][chosen_label])
            chosen_confidence = max(
                float(raw_scores[chosen_i].item()),
                float(repaired_probs[chosen_i].item()),
            )

            cpu = float(ip[0, cpu_idx].item())
            gpu = float(ip[0, gpu_idx].item())
            controller = float(rp[0, 0].item())
            executor = float(rp[0, 1].item())

            pair_gate = False
            concept_gate = False
            llm_scope_rescue = False

            if chosen_label == "tech_cpu":
                pair_gate = (
                    cpu > gpu
                    and (cpu - gpu) >= pair_intent_margin
                    and controller > executor
                    and (controller - executor) >= pair_role_margin
                )
            elif chosen_label == "tech_gpu":
                pair_gate = (
                    gpu > cpu
                    and (gpu - cpu) >= pair_intent_margin
                    and executor > controller
                    and (executor - controller) >= pair_role_margin
                )
            else:
                normal_scope = scope_prob >= SCOPE_THRESHOLD
                if chosen_label == "tech_llm":
                    llm_scope_rescue = (
                        raw_top_label == "tech_llm"
                        and repaired_top_label == "tech_llm"
                        and raw_margin <= LLM_RESCUE_MARGIN
                        and chosen_confidence >= LLM_RESCUE_CONFIDENCE
                    )
                concept_gate = (
                    (normal_scope or llm_scope_rescue)
                    and chosen_confidence >= BINDING_CONFIDENCE
                    and canonical.lower() not in prompt_text.lower()
                )

            active = pair_gate or concept_gate
            base_top_id = int(torch.argmax(logits).item())

            boost = 0.0
            if active:
                features = torch.cat([ip[0], rp[0]], dim=-1).unsqueeze(0)
                cidx = torch.tensor([concept_idx], dtype=torch.long, device=device)
                boost = float(binding(features, cidx)[0].item())
                logits[target_id] += boost

            after_top_id = int(torch.argmax(logits).item())
            debug = {
                "raw_top": raw_top_label,
                "raw_margin": raw_margin,
                "repair_top": repaired_top_label,
                "repair_conf": repaired_top_conf,
                "chosen": chosen_label,
                "chosen_conf": chosen_confidence,
                "scope": scope_prob,
                "repair_applied": repair_applied,
                "llm_rescue": llm_scope_rescue,
                "active": active,
                "target_id": target_id,
                "boost": boost,
                "base_top_id": base_top_id,
                "after_top_id": after_top_id,
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
        if "\n" in tokenizer.decode(response, skip_special_tokens=True):
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
    p.add_argument("--repair", default=DEFAULT_REPAIR)
    args = p.parse_args()

    for filename in (args.tokenizer, args.model, args.intent_head, args.role_checkpoint, args.binding, args.repair):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ck = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    intent_head, intent_ck, intent_labels = load_intent_head(args.intent_head, model, device)
    role_head, _old_binding, role_ck = load_role_checkpoint(args.role_checkpoint, intent_labels, device)
    binding, binding_ck = load_binding(args.binding, intent_labels, device)
    repair, repair_ck = load_repair(args.repair, intent_labels, device)

    print("====================================================")
    print(" LLM-specific Scope Rescue v0.11.12")
    print("====================================================")
    print("Device:", device)
    print("Fixed thresholds: margin=0.20 scope=0.50 binding_conf=0.45")
    print("LLM rescue: raw=LLM, repair=LLM, raw_margin<=0.20, confidence>=0.70")
    print()

    semantic = strict = fluency = direct_hits = 0
    guard_ok = True

    for idx, case in enumerate(CASES, start=1):
        prompt = str(case["prompt"])
        reply, dbg = generate(
            model, tokenizer, intent_head, intent_labels, role_head,
            binding, binding_ck, repair, prompt
        )
        dims = dimension_match(
            reply, prompt, str(case["intent"]), case["required_all"], case["forbidden"]
        )
        semantic += int(dims["semantic_ok"])
        strict += int(dims["strict_ok"])
        fluency += int(dims["fluent_ok"])

        print(f"[G{idx:02d}] {str(case['intent']):14s} 人: {prompt}")
        print("      AI:", reply)
        print(
            "      semantic-content=" + ("PASS" if dims["semantic_ok"] else "MISS")
            + " | strict=" + ("PASS" if dims["strict_ok"] else "MISS")
        )
        print(
            f"      raw={dbg['raw_top']} margin={dbg['raw_margin']:.3f} "
            f"repair={dbg['repair_top']}:{dbg['repair_conf']:.3f} "
            f"chosen={dbg['chosen']}:{dbg['chosen_conf']:.3f} "
            f"scope={dbg['scope']:.3f} rescue={'YES' if dbg['llm_rescue'] else 'NO'} "
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
            print("      direct-entity=" + ("PASS" if ok else "MISS") + " | expected=" + expected)

        if idx == 21 and dbg["active"]:
            guard_ok = False
        if idx == 30 and dbg["active"]:
            guard_ok = False

    print()
    print("Summary")
    print("-------")
    print(f"Semantic-content rate : {semantic}/30 ({semantic/30:.1%})")
    print(f"Fluency rate          : {fluency}/30 ({fluency/30:.1%})")
    print(f"Strict composite rate : {strict}/30 ({strict/30:.1%})")
    print(f"Tracked direct rate   : {direct_hits}/{len(DIRECT)} ({direct_hits/len(DIRECT):.1%})")
    print(f"G21/G30 safety        : {'PASS' if guard_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
