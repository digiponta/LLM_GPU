# evaluate_canonical_entity_prefix_binding_v01114.py
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from selective_intent_repair_v01110 import (
    TECH_LABELS, extract_technical_logits, load_checkpoint as load_repair,
)
from multi_concept_safe_binding_v0118 import (
    CONCEPTS, CANONICAL, load_checkpoint as load_binding,
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
LLM_FIRST_TOKEN_MULTIPLIER = 2.00
PREFIX_MARGIN = 0.50

DIRECT = {
    5: "CPU", 7: "LLM", 8: "Transformer", 9: "CUDA",
    10: "Python", 27: "GPU", 28: "CPU",
}


def token_text(tok, token_id: int) -> str:
    try:
        return tok.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")
    except Exception:
        return ""


@torch.no_grad()
def generate(model, tok, intent_head, intent_labels, role_head, binding, binding_ck, repair, prompt_text):
    prompt = f"人: {prompt_text}\nAI: "
    generated = tok.encode(prompt, add_bos=True)
    response = []
    device = next(model.parameters()).device
    cpu_idx = intent_labels.index("tech_cpu")
    gpu_idx = intent_labels.index("tech_gpu")
    pair_intent_margin = float(binding_ck.get("cpu_gpu_intent_margin", 0.10))
    pair_role_margin = float(binding_ck.get("cpu_gpu_role_margin", 0.05))

    debug = None
    prefix_ids = []
    binding_active = False
    first_step_boost = 0.0
    llm_rescue = False

    for step in range(96):
        x = torch.tensor([generated[-model.context_length:]], dtype=torch.long, device=device)
        h = model.forward_hidden(x)[:, -1, :]
        logits = model.lm_head(h)[0].clone()

        if step == 0:
            intent_logits = intent_head(h)
            ip = torch.sigmoid(intent_logits)
            rp = torch.sigmoid(role_head(h))
            raw = extract_technical_logits(intent_logits, intent_labels)
            repaired, scope_logit = repair(raw)

            raw_scores = torch.sigmoid(raw)[0]
            repaired_probs = torch.softmax(repaired, dim=-1)[0]
            scope = float(torch.sigmoid(scope_logit)[0].item())

            order = torch.argsort(raw[0], descending=True)
            top = int(order[0].item())
            second = int(order[1].item())
            raw_top = TECH_LABELS[top]
            raw_margin = float((raw[0, top] - raw[0, second]).item())

            rep_top = int(torch.argmax(repaired_probs).item())
            rep_label = TECH_LABELS[rep_top]
            rep_conf = float(repaired_probs[rep_top].item())

            chosen = top
            repair_applied = False
            if raw_margin <= REPAIR_MARGIN and rep_top != top and rep_conf >= REPAIR_CONFIDENCE:
                chosen = rep_top
                repair_applied = True

            chosen_label = TECH_LABELS[chosen]
            canonical = CANONICAL[chosen_label]
            cidx = CONCEPTS.index(chosen_label)
            prefix_ids = tok.encode(canonical)
            if not prefix_ids:
                raise RuntimeError("Empty canonical tokenization: " + canonical)

            chosen_conf = max(float(raw_scores[chosen].item()), float(repaired_probs[chosen].item()))
            cpu = float(ip[0, cpu_idx].item())
            gpu = float(ip[0, gpu_idx].item())
            controller = float(rp[0, 0].item())
            executor = float(rp[0, 1].item())

            pair_gate = False
            concept_gate = False
            if chosen_label == "tech_cpu":
                pair_gate = cpu > gpu and (cpu-gpu) >= pair_intent_margin and controller > executor and (controller-executor) >= pair_role_margin
            elif chosen_label == "tech_gpu":
                pair_gate = gpu > cpu and (gpu-cpu) >= pair_intent_margin and executor > controller and (executor-controller) >= pair_role_margin
            else:
                normal_scope = scope >= SCOPE_THRESHOLD
                if chosen_label == "tech_llm":
                    llm_rescue = (
                        raw_top == "tech_llm"
                        and rep_label == "tech_llm"
                        and raw_margin <= LLM_RESCUE_MARGIN
                        and chosen_conf >= LLM_RESCUE_CONFIDENCE
                    )
                concept_gate = (
                    (normal_scope or llm_rescue)
                    and chosen_conf >= BINDING_CONFIDENCE
                    and canonical.lower() not in prompt_text.lower()
                )

            binding_active = pair_gate or concept_gate
            base_top = int(torch.argmax(logits).item())

            if binding_active:
                feats = torch.cat([ip[0], rp[0]], dim=-1).unsqueeze(0)
                ci = torch.tensor([cidx], dtype=torch.long, device=device)
                learned = float(binding(feats, ci)[0].item())
                first_step_boost = learned * (LLM_FIRST_TOKEN_MULTIPLIER if llm_rescue and chosen_label == "tech_llm" else 1.0)
                logits[prefix_ids[0]] += first_step_boost

            after_top = int(torch.argmax(logits).item())
            debug = {
                "raw_top": raw_top,
                "raw_margin": raw_margin,
                "repair_top": rep_label,
                "repair_conf": rep_conf,
                "chosen": chosen_label,
                "chosen_conf": chosen_conf,
                "scope": scope,
                "repair_applied": repair_applied,
                "llm_rescue": llm_rescue,
                "active": binding_active,
                "canonical": canonical,
                "prefix_ids": list(prefix_ids),
                "first_boost": first_step_boost,
                "base_top": base_top,
                "after_top": after_top,
                "prefix_steps": [],
            }

        elif binding_active and step < len(prefix_ids):
            expected = int(prefix_ids[step])
            target_logit = float(logits[expected].item())
            masked = logits.clone()
            masked[expected] = -torch.inf
            competitor_logit = float(masked.max().item())
            needed = max(0.0, competitor_logit + PREFIX_MARGIN - target_logit)
            logits[expected] += needed
            debug["prefix_steps"].append({
                "step": step,
                "token_id": expected,
                "token": token_text(tok, expected),
                "boost": needed,
                "target_before": target_logit,
                "competitor": competitor_logit,
                "target_after": float(logits[expected].item()),
            })

        if response:
            for tid in set(response):
                if logits[tid] >= 0:
                    logits[tid] /= 1.05
                else:
                    logits[tid] *= 1.05

        nid = int(torch.argmax(logits).item())
        if nid == tok.eos_id:
            break
        generated.append(nid)
        response.append(nid)
        if "\n" in tok.decode(response, skip_special_tokens=True):
            break

    reply = tok.decode(response, skip_special_tokens=True).split("\n", 1)[0].strip()
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
    tok = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    intent_head, _, intent_labels = load_intent_head(args.intent_head, model, device)
    role_head, _, _ = load_role_checkpoint(args.role_checkpoint, intent_labels, device)
    binding, binding_ck = load_binding(args.binding, intent_labels, device)
    repair, _ = load_repair(args.repair, intent_labels, device)

    print("=" * 56)
    print(" Canonical Entity Prefix Binding v0.11.14")
    print("=" * 56)
    print("Device:", device)
    print("Fixed thresholds: margin=0.20 scope=0.50 confidence=0.45")
    print("LLM first-token multiplier: 2.00")
    print("Continuation prefix margin:", PREFIX_MARGIN)
    print()
    for label in CONCEPTS:
        ids = tok.encode(CANONICAL[label])
        pieces = [token_text(tok, i) for i in ids]
        print(f"{label:16s} -> {CANONICAL[label]:11s} ids={ids} pieces={pieces}")
    print()

    semantic = strict = fluency = direct_hits = 0
    guard_ok = True

    for idx, case in enumerate(CASES, start=1):
        prompt = str(case["prompt"])
        reply, dbg = generate(model, tok, intent_head, intent_labels, role_head, binding, binding_ck, repair, prompt)
        dims = dimension_match(reply, prompt, str(case["intent"]), case["required_all"], case["forbidden"])
        semantic += int(dims["semantic_ok"])
        strict += int(dims["strict_ok"])
        fluency += int(dims["fluent_ok"])

        print(f"[G{idx:02d}] {str(case['intent']):14s} 人: {prompt}")
        print("      AI:", reply)
        print("      semantic-content=" + ("PASS" if dims["semantic_ok"] else "MISS") + " | strict=" + ("PASS" if dims["strict_ok"] else "MISS"))
        print(
            f"      chosen={dbg['chosen']} scope={dbg['scope']:.3f} rescue={'YES' if dbg['llm_rescue'] else 'NO'} "
            f"binding={'ON' if dbg['active'] else 'OFF'} canonical={dbg['canonical']}"
        )
        if dbg["active"]:
            print(f"      first boost={dbg['first_boost']:+.3f} prefix_ids={dbg['prefix_ids']}")
            for item in dbg["prefix_steps"]:
                print(
                    f"      prefix step={item['step']} token={item['token']!r} boost={item['boost']:+.3f} "
                    f"target={item['target_before']:+.3f}->{item['target_after']:+.3f} competitor={item['competitor']:+.3f}"
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
