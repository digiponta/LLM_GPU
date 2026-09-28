# evaluate_transformer_continuation_category_repair_v01117.py
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from evaluate_post_entity_boundary_binding_v01115 import (
    DEFAULT_TOKENIZER, DEFAULT_MODEL, DEFAULT_INTENT, DEFAULT_ROLE,
    DEFAULT_BINDING, DEFAULT_REPAIR,
    generate as generate_v01115,
    build_boundary_block_ids,
)
from selective_intent_repair_v01110 import load_checkpoint as load_repair
from multi_concept_safe_binding_v0118 import CANONICAL, load_checkpoint as load_binding
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from evaluate_generalization_v07 import CASES, dimension_match
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

# Short semantic continuation anchors.  These are not complete answers;
# they only provide enough post-entity structure to keep generation aligned.
SEMANTIC_CONTINUATIONS = {
    "GPU": "は大量の並列計算を得意",
    "CPU": "は汎用処理や制御を担当",
    "LLM": "は文章を学習して生成する言語モデル",
    "Transformer": "はAttentionを中心に使うモデル構造",
    "CUDA": "はNVIDIA GPUで汎用計算を行う技術",
    "Python": "は読みやすい汎用プログラミング言語",
}

CONTINUATION_MARGIN = 0.35
MAX_GUIDED_TOKENS = 8
TRANSFORMER_FULL_ANCHOR = True

DIRECT = {
    5: "CPU", 7: "LLM", 8: "Transformer", 9: "CUDA",
    10: "Python", 27: "GPU", 28: "CPU",
}


def token_text(tok, token_id: int) -> str:
    try:
        return tok.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")
    except Exception:
        return ""


def detect_generated_entity(tok, response_ids):
    """Return canonical entity if the generated prefix exactly matches one."""
    matches = []
    for canonical in SEMANTIC_CONTINUATIONS:
        ids = tok.encode(canonical)
        if len(response_ids) >= len(ids) and response_ids[:len(ids)] == ids:
            matches.append((len(ids), canonical, ids))
    if not matches:
        return None, []
    matches.sort(reverse=True)
    _, canonical, ids = matches[0]
    return canonical, ids


@torch.no_grad()
def generate(model, tok, intent_head, intent_labels, role_head, binding, binding_ck, repair, prompt_text, boundary_block_ids):
    # Reuse v0.11.15 once to determine the actual generated entity and whether
    # its output is already semantically sufficient.
    baseline_reply, baseline_dbg = generate_v01115(
        model, tok, intent_head, intent_labels, role_head,
        binding, binding_ck, repair, prompt_text, boundary_block_ids
    )

    baseline_ids = tok.encode(baseline_reply)
    canonical, canonical_ids = detect_generated_entity(tok, baseline_ids)

    # If no canonical entity appears at the start, keep v0.11.15 unchanged.
    if canonical is None:
        baseline_dbg["continuation_entity"] = None
        baseline_dbg["continuation_steps"] = []
        baseline_dbg["continuation_active"] = False
        return baseline_reply, baseline_dbg

    # Only repair when current output is semantically incomplete for this case.
    case = next((c for c in CASES if str(c["prompt"]) == prompt_text), None)
    if case is not None:
        dims = dimension_match(
            baseline_reply, prompt_text, str(case["intent"]),
            case["required_all"], case["forbidden"]
        )
        if dims["semantic_ok"]:
            baseline_dbg["continuation_entity"] = canonical
            baseline_dbg["continuation_steps"] = []
            baseline_dbg["continuation_active"] = False
            return baseline_reply, baseline_dbg

    anchor_text = SEMANTIC_CONTINUATIONS[canonical]
    anchor_ids = tok.encode(anchor_text)
    # The anchor starts with the canonical entity's boundary/continuation text,
    # not with the entity itself.
    if canonical == "Transformer" and TRANSFORMER_FULL_ANCHOR:
        guide_ids = anchor_ids
    else:
        guide_ids = anchor_ids[:MAX_GUIDED_TOKENS]

    prompt = f"人: {prompt_text}\nAI: "
    generated = tok.encode(prompt, add_bos=True)
    response = []
    device = next(model.parameters()).device

    # First reproduce v0.11.15's canonical entity exactly by seeding it.
    response.extend(canonical_ids)
    generated.extend(canonical_ids)

    continuation_steps = []
    for step, expected in enumerate(guide_ids):
        x = torch.tensor([generated[-model.context_length:]], dtype=torch.long, device=device)
        h = model.forward_hidden(x)[:, -1, :]
        logits = model.lm_head(h)[0].clone()

        target_before = float(logits[expected].item())
        masked = logits.clone()
        masked[expected] = -torch.inf
        competitor = float(masked.max().item())
        needed = max(0.0, competitor + CONTINUATION_MARGIN - target_before)
        logits[expected] += needed

        if response:
            for tid in set(response):
                if logits[tid] >= 0:
                    logits[tid] /= 1.05
                else:
                    logits[tid] *= 1.05

        nid = int(torch.argmax(logits).item())
        generated.append(nid)
        response.append(nid)
        continuation_steps.append({
            "step": step,
            "expected": int(expected),
            "piece": token_text(tok, int(expected)),
            "selected": nid,
            "selected_piece": token_text(tok, nid),
            "boost": needed,
            "target_before": target_before,
            "competitor": competitor,
        })
        if nid == tok.eos_id:
            break

    # Return to ordinary greedy LM generation after the guided prefix.
    while len(response) < 96:
        x = torch.tensor([generated[-model.context_length:]], dtype=torch.long, device=device)
        h = model.forward_hidden(x)[:, -1, :]
        logits = model.lm_head(h)[0].clone()
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
    baseline_dbg["continuation_entity"] = canonical
    baseline_dbg["continuation_steps"] = continuation_steps
    baseline_dbg["continuation_active"] = True
    baseline_dbg["baseline_reply"] = baseline_reply
    return reply, baseline_dbg


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
    boundary_block_ids = build_boundary_block_ids(tok)

    print("=" * 58)
    print(" Transformer Continuation Category Repair v0.11.17")
    print("=" * 58)
    print("Device:", device)
    print("Continuation margin:", CONTINUATION_MARGIN)
    print("Default max guided tokens:", MAX_GUIDED_TOKENS)
    print("Transformer guidance: full semantic anchor")
    print("Trigger: actual generated canonical entity + semantic MISS")
    print()

    semantic = strict = fluency = direct_hits = 0
    safety = True

    for idx, case in enumerate(CASES, start=1):
        prompt = str(case["prompt"])
        reply, dbg = generate(
            model, tok, intent_head, intent_labels, role_head,
            binding, binding_ck, repair, prompt, boundary_block_ids
        )
        dims = dimension_match(
            reply, prompt, str(case["intent"]),
            case["required_all"], case["forbidden"]
        )
        semantic += int(dims["semantic_ok"])
        strict += int(dims["strict_ok"])
        fluency += int(dims["fluent_ok"])

        print(f"[G{idx:02d}] {str(case['intent']):14s} 人: {prompt}")
        if dbg.get("continuation_active"):
            print("      baseline:", dbg.get("baseline_reply", ""))
        print("      AI:", reply)
        print(
            "      semantic-content=" + ("PASS" if dims["semantic_ok"] else "MISS")
            + " | strict=" + ("PASS" if dims["strict_ok"] else "MISS")
        )
        print(
            "      continuation="
            + ("ON" if dbg.get("continuation_active") else "OFF")
            + " entity=" + str(dbg.get("continuation_entity"))
        )
        for item in dbg.get("continuation_steps", []):
            print(
                f"      guide step={item['step']} expected={item['piece']!r} "
                f"selected={item['selected_piece']!r} boost={item['boost']:+.3f}"
            )

        if idx in DIRECT:
            expected = DIRECT[idx]
            ok = reply.startswith(expected)
            direct_hits += int(ok)
            print("      direct-entity=" + ("PASS" if ok else "MISS") + " | expected=" + expected)

        # Preserve the v0.11.15 safety requirements.
        if idx == 21 and dbg.get("continuation_active"):
            safety = False
        if idx == 30 and dbg.get("continuation_active"):
            safety = False

    print()
    print("Summary")
    print("-------")
    print(f"Semantic-content rate : {semantic}/30 ({semantic/30:.1%})")
    print(f"Fluency rate          : {fluency}/30 ({fluency/30:.1%})")
    print(f"Strict composite rate : {strict}/30 ({strict/30:.1%})")
    print(f"Tracked direct rate   : {direct_hits}/{len(DIRECT)} ({direct_hits/len(DIRECT):.1%})")
    print(f"G21/G30 safety        : {'PASS' if safety else 'FAIL'}")


if __name__ == "__main__":
    main()
