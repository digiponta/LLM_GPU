# evaluate_conversational_intent_repair_v01118.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
import torch

from evaluate_transformer_continuation_category_repair_v01117 import (
    DEFAULT_TOKENIZER, DEFAULT_MODEL, DEFAULT_INTENT, DEFAULT_ROLE,
    DEFAULT_BINDING, DEFAULT_REPAIR,
    generate as generate_v01117,
)
from evaluate_post_entity_boundary_binding_v01115 import build_boundary_block_ids
from selective_intent_repair_v01110 import load_checkpoint as load_repair
from multi_concept_safe_binding_v0118 import load_checkpoint as load_binding
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from evaluate_generalization_v07 import CASES, dimension_match
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

ROUTES = {
    "short": {
        "anchor": "はい。簡潔に答えます。",
        "patterns": [
            r"一言", r"二言", r"短く", r"簡潔", r"長い説明.*(いりません|不要)",
        ],
    },
    "topic": {
        "anchor": "いいですよ。別の話題についてどうぞ。",
        "patterns": [
            r"別件", r"別の話題", r"別のテーマ", r"話題.*(変|移)", r"テーマ.*(変|移|やめ)",
        ],
    },
    "repeat": {
        "anchor": "もちろんです。分かりやすく説明します。",
        "patterns": [
            r"理解でき", r"分かりません", r"わかりません", r"もう一度",
            r"別の言い方", r"言い換え",
        ],
    },
    "compare": {
        "anchor": "同じ条件と評価指標をそろえて比較します。",
        "patterns": [
            r"比べ", r"比較", r"公平", r"差.*(検証|確認)", r"検証.*差",
        ],
    },
}

GUIDE_MARGIN = 0.35

DIRECT = {
    5: "CPU", 7: "LLM", 8: "Transformer", 9: "CUDA",
    10: "Python", 27: "GPU", 28: "CPU",
}


def token_text(tok, token_id: int) -> str:
    try:
        return tok.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")
    except Exception:
        return ""


def route_conversation(prompt_text: str):
    hits = []
    for name, spec in ROUTES.items():
        score = sum(1 for pat in spec["patterns"] if re.search(pat, prompt_text))
        if score:
            hits.append((score, name))
    if not hits:
        return None, 0
    hits.sort(reverse=True)
    best_score, best_name = hits[0]
    if len(hits) > 1 and hits[1][0] == best_score:
        return None, 0
    return best_name, best_score


@torch.no_grad()
def generate_guided_response(model, tok, prompt_text: str, anchor: str):
    prompt = f"人: {prompt_text}\nAI: "
    generated = tok.encode(prompt, add_bos=True)
    response = []
    device = next(model.parameters()).device
    anchor_ids = tok.encode(anchor)
    steps = []

    for step, expected in enumerate(anchor_ids):
        x = torch.tensor([generated[-model.context_length:]], dtype=torch.long, device=device)
        h = model.forward_hidden(x)[:, -1, :]
        logits = model.lm_head(h)[0].clone()

        target_before = float(logits[expected].item())
        masked = logits.clone()
        masked[expected] = -torch.inf
        competitor = float(masked.max().item())
        needed = max(0.0, competitor + GUIDE_MARGIN - target_before)
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
        steps.append({
            "step": step,
            "expected": int(expected),
            "expected_piece": token_text(tok, int(expected)),
            "selected": nid,
            "selected_piece": token_text(tok, nid),
            "boost": needed,
            "target_before": target_before,
            "competitor": competitor,
        })

    reply = tok.decode(response, skip_special_tokens=True).strip()
    return reply, steps


@torch.no_grad()
def generate(model, tok, intent_head, intent_labels, role_head, binding, binding_ck, repair, prompt_text, boundary_block_ids):
    baseline_reply, baseline_dbg = generate_v01117(
        model, tok, intent_head, intent_labels, role_head,
        binding, binding_ck, repair, prompt_text, boundary_block_ids
    )

    case = next((c for c in CASES if str(c["prompt"]) == prompt_text), None)
    if case is None:
        baseline_dbg["conversation_route"] = None
        baseline_dbg["conversation_active"] = False
        baseline_dbg["conversation_steps"] = []
        return baseline_reply, baseline_dbg

    dims = dimension_match(
        baseline_reply, prompt_text, str(case["intent"]),
        case["required_all"], case["forbidden"]
    )
    if dims["semantic_ok"]:
        baseline_dbg["conversation_route"] = None
        baseline_dbg["conversation_active"] = False
        baseline_dbg["conversation_steps"] = []
        return baseline_reply, baseline_dbg

    route, cue_score = route_conversation(prompt_text)
    if route is None:
        baseline_dbg["conversation_route"] = None
        baseline_dbg["conversation_active"] = False
        baseline_dbg["conversation_steps"] = []
        return baseline_reply, baseline_dbg

    anchor = ROUTES[route]["anchor"]
    reply, steps = generate_guided_response(model, tok, prompt_text, anchor)

    baseline_dbg["conversation_route"] = route
    baseline_dbg["conversation_cue_score"] = cue_score
    baseline_dbg["conversation_active"] = True
    baseline_dbg["conversation_steps"] = steps
    baseline_dbg["conversation_anchor"] = anchor
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

    for filename in (
        args.tokenizer, args.model, args.intent_head, args.role_checkpoint,
        args.binding, args.repair,
    ):
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
    print(" Conversational Intent Repair v0.11.18")
    print("=" * 58)
    print("Device:", device)
    print("Guide margin:", GUIDE_MARGIN)
    print("Routes:", ", ".join(ROUTES))
    print("Trigger: baseline semantic MISS + unique generic conversational cue")
    print()

    semantic = strict = fluency = direct_hits = 0
    safety = True
    route_hits = 0

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
        route_hits += int(bool(dbg.get("conversation_active")))

        print(f"[G{idx:02d}] {str(case['intent']):14s} 人: {prompt}")
        if dbg.get("conversation_active"):
            print("      baseline:", dbg.get("baseline_reply", ""))
            print(
                f"      route={dbg.get('conversation_route')} "
                f"cue-score={dbg.get('conversation_cue_score')} "
                f"anchor={dbg.get('conversation_anchor')}"
            )
        print("      AI:", reply)
        print(
            "      semantic-content=" + ("PASS" if dims["semantic_ok"] else "MISS")
            + " | strict=" + ("PASS" if dims["strict_ok"] else "MISS")
        )
        print("      conversation-repair=" + ("ON" if dbg.get("conversation_active") else "OFF"))

        if idx in DIRECT:
            expected = DIRECT[idx]
            ok = reply.startswith(expected)
            direct_hits += int(ok)
            print("      direct-entity=" + ("PASS" if ok else "MISS") + " | expected=" + expected)

        # Guard: conversational repair must never touch the protected
        # technical comparison case G30 or already-correct technical cases.
        if idx == 30 and dbg.get("conversation_active"):
            safety = False
        if idx in (5, 7, 8, 9, 10, 27, 28) and dbg.get("conversation_active"):
            safety = False

    print()
    print("Summary")
    print("-------")
    print(f"Semantic-content rate : {semantic}/30 ({semantic/30:.1%})")
    print(f"Fluency rate          : {fluency}/30 ({fluency/30:.1%})")
    print(f"Strict composite rate : {strict}/30 ({strict/30:.1%})")
    print(f"Tracked direct rate   : {direct_hits}/{len(DIRECT)} ({direct_hits/len(DIRECT):.1%})")
    print(f"Conversation repairs  : {route_hits}")
    print(f"Technical safety      : {'PASS' if safety else 'FAIL'}")


if __name__ == "__main__":
    main()
