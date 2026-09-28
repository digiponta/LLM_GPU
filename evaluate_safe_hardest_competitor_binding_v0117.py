# evaluate_safe_hardest_competitor_binding_v0117.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from safe_hardest_competitor_binding_v0117 import load_checkpoint
from cpu_gpu_role_binding_v0112 import load_checkpoint as load_role_checkpoint
from evaluate_generalization_v07 import CASES, dimension_match
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_ROLE = "model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt"
DEFAULT_BINDING = "model/model-gpu-v0.11.7-safe-hardest-competitor-binding.pt"

DEFAULT_TECH_LABELS = (
    "tech_gpu",
    "tech_cpu",
    "tech_llm",
    "tech_transformer",
    "tech_cuda",
    "tech_python",
)

DIRECT = {5: "CPU", 27: "GPU", 28: "CPU"}


def token_text(tok, token_id: int) -> str:
    try:
        return tok.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")
    except Exception:
        return ""


def top_technical(ip, intent_labels, technical_labels):
    pairs = []
    for label in technical_labels:
        if label in intent_labels:
            idx = intent_labels.index(label)
            pairs.append((label, float(ip[0, idx].item())))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[0] if pairs else ("", 0.0), pairs


@torch.no_grad()
def generate(model, tokenizer, intent_head, intent_labels, role_head, adapter, ck, prompt_text):
    prompt = f"人: {prompt_text}\nAI: "
    generated = tokenizer.encode(prompt, add_bos=True)
    response = []
    device = next(model.parameters()).device

    cpu_idx = intent_labels.index("tech_cpu")
    gpu_idx = intent_labels.index("tech_gpu")
    cpu_token = int(ck["cpu_token_id"])
    gpu_token = int(ck["gpu_token_id"])
    intent_margin = float(ck.get("intent_margin_threshold", 0.10))
    role_margin = float(ck.get("role_margin_threshold", 0.05))
    technical_labels = tuple(ck.get("technical_labels", DEFAULT_TECH_LABELS))

    debug = None

    for step in range(96):
        context = generated[-model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        hidden = model.forward_hidden(x)
        h = hidden[:, -1, :]
        logits = model.lm_head(h)[0].clone()

        if step == 0:
            ip = torch.sigmoid(intent_head(h))
            rp = torch.sigmoid(role_head(h))

            cpu = float(ip[0, cpu_idx].item())
            gpu = float(ip[0, gpu_idx].item())
            controller = float(rp[0, 0].item())
            executor = float(rp[0, 1].item())

            (top_tech_label, top_tech_score), ranked_tech = top_technical(
                ip, intent_labels, technical_labels
            )

            cpu_top = top_tech_label == "tech_cpu"
            gpu_top = top_tech_label == "tech_gpu"

            cpu_case = (
                cpu_top
                and cpu > gpu
                and (cpu - gpu) >= intent_margin
                and controller > executor
                and (controller - executor) >= role_margin
            )
            gpu_case = (
                gpu_top
                and gpu > cpu
                and (gpu - cpu) >= intent_margin
                and executor > controller
                and (executor - controller) >= role_margin
            )

            active = cpu_case or gpu_case
            target_id = cpu_token if cpu_case else gpu_token if gpu_case else None

            base_top_id = int(torch.argmax(logits).item())
            base_target = (
                float(logits[target_id].item())
                if target_id is not None
                else float("nan")
            )

            boost = 0.0
            if active:
                features = torch.cat([ip[0], rp[0]], dim=-1).unsqueeze(0)
                boost = float(adapter(features)[0].item())
                logits[target_id] += boost

            after_top_id = int(torch.argmax(logits).item())
            after_target = (
                float(logits[target_id].item())
                if target_id is not None
                else float("nan")
            )

            debug = {
                "cpu": cpu,
                "gpu": gpu,
                "controller": controller,
                "executor": executor,
                "top_tech_label": top_tech_label,
                "top_tech_score": top_tech_score,
                "ranked_tech": ranked_tech,
                "active": active,
                "target_id": target_id,
                "boost": boost,
                "base_top_id": base_top_id,
                "after_top_id": after_top_id,
                "base_target": base_target,
                "after_target": after_target,
                "base_top_logit": float(model.lm_head(h)[0, base_top_id].item()),
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
    args = p.parse_args()

    for filename in (
        args.tokenizer,
        args.model,
        args.intent_head,
        args.role_checkpoint,
        args.binding,
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
    adapter, bind_ck = load_checkpoint(args.binding, intent_labels, device)

    print("====================================================")
    print(" Safe Hardest-Competitor Binding v0.11.7 Evaluation")
    print("====================================================")
    print("Device:", device)
    print("Base loss:", base_ck.get("loss"))
    print("Binding loss:", bind_ck.get("loss"))
    print("Target full-vocab margin:", bind_ck.get("target_margin"))
    print("Technical top gate:", bind_ck.get("technical_top_gate", True))
    print("Initial boost:", bind_ck.get("initial_boost"))

    semantic = strict = fluency = direct = 0

    for idx, case in enumerate(CASES, start=1):
        prompt = str(case["prompt"])
        reply, dbg = generate(
            model,
            tokenizer,
            intent_head,
            intent_labels,
            role_head,
            adapter,
            bind_ck,
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
            f"      CPU={dbg['cpu']:.3f} GPU={dbg['gpu']:.3f} "
            f"controller={dbg['controller']:.3f} executor={dbg['executor']:.3f} "
            f"top-tech={dbg['top_tech_label']}:{dbg['top_tech_score']:.3f} "
            f"safe-binding={'ON' if dbg['active'] else 'OFF'}"
        )

        if dbg["active"]:
            print(
                f"      target={token_text(tokenizer, dbg['target_id'])!r} "
                f"boost={dbg['boost']:+.3f} "
                f"base-winner={token_text(tokenizer, dbg['base_top_id'])!r} "
                f"after-winner={token_text(tokenizer, dbg['after_top_id'])!r}"
            )
            print(
                f"      target logit: base={dbg['base_target']:+.3f} "
                f"after={dbg['after_target']:+.3f} "
                f"winner-after={dbg['after_top_logit']:+.3f}"
            )

        if idx in DIRECT:
            ok = reply.startswith(DIRECT[idx])
            direct += int(ok)
            print(
                "      direct-entity="
                + ("PASS" if ok else "MISS")
                + " | expected="
                + DIRECT[idx]
            )

    print()
    print("Summary")
    print("-------")
    print(f"Semantic-content rate : {semantic}/30 ({semantic/30:.1%})")
    print(f"Fluency rate          : {fluency}/30 ({fluency/30:.1%})")
    print(f"Strict composite rate : {strict}/30 ({strict/30:.1%})")
    print(f"CPU/GPU direct rate   : {direct}/3 ({direct/3:.1%})")


if __name__ == "__main__":
    main()
