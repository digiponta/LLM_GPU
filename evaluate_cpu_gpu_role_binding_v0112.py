# evaluate_cpu_gpu_role_binding_v0112.py
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from cpu_gpu_role_binding_v0112 import load_checkpoint
from evaluate_generalization_v07 import CASES, dimension_match
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_ROLE = "model/model-gpu-v0.11.2-cpu-gpu-role-binding.pt"

DIRECT = {5: "CPU", 27: "GPU", 28: "CPU"}


@torch.no_grad()
def generate(model, tokenizer, intent_head, intent_labels, role_head, binding, ck, prompt_text):
    prompt = f"人: {prompt_text}\nAI: "
    generated = tokenizer.encode(prompt, add_bos=True)
    response = []
    device = next(model.parameters()).device
    debug = None

    cpu_idx = intent_labels.index("tech_cpu")
    gpu_idx = intent_labels.index("tech_gpu")
    role_gate = float(ck.get("role_gate_threshold", 0.55))
    entity_margin = float(ck.get("entity_margin_threshold", 0.10))

    for step in range(96):
        context = generated[-model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        hidden = model.forward_hidden(x)
        h = hidden[:, -1, :]
        logits = model.lm_head(h)[0].clone()

        if step == 0:
            intent_prob = torch.sigmoid(intent_head(h))
            role_prob = torch.sigmoid(role_head(h))

            cpu = float(intent_prob[0, cpu_idx].item())
            gpu = float(intent_prob[0, gpu_idx].item())
            controller = float(role_prob[0, 0].item())
            executor = float(role_prob[0, 1].item())

            entity_clear = max(cpu, gpu) >= 0.50 and abs(cpu - gpu) >= entity_margin
            role_clear = max(controller, executor) >= role_gate
            role_matches = (
                (cpu > gpu and controller > executor)
                or (gpu > cpu and executor > controller)
            )
            active = entity_clear and role_clear and role_matches

            base_cpu = float(logits[int(ck["target_token_ids"]["CPU"])].item())
            base_gpu = float(logits[int(ck["target_token_ids"]["GPU"])].item())

            if active:
                bias = binding(intent_prob, role_prob)[0]
                logits = logits + bias
                bias_cpu = float(bias[int(ck["target_token_ids"]["CPU"])].item())
                bias_gpu = float(bias[int(ck["target_token_ids"]["GPU"])].item())
            else:
                bias_cpu = 0.0
                bias_gpu = 0.0

            after_cpu = float(logits[int(ck["target_token_ids"]["CPU"])].item())
            after_gpu = float(logits[int(ck["target_token_ids"]["GPU"])].item())

            debug = {
                "cpu": cpu,
                "gpu": gpu,
                "controller": controller,
                "executor": executor,
                "active": active,
                "base_gap": base_cpu - base_gpu,
                "bias_gap": bias_cpu - bias_gpu,
                "after_gap": after_cpu - after_gpu,
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
    p.add_argument("--role-binding", default=DEFAULT_ROLE)
    args = p.parse_args()

    for filename in (args.tokenizer, args.model, args.intent_head, args.role_binding):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_ck = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    intent_head, intent_ck, intent_labels = load_intent_head(args.intent_head, model, device)
    role_head, binding, role_ck = load_checkpoint(args.role_binding, intent_labels, device)

    print("====================================================")
    print(" CPU-GPU Role Binding v0.11.2 Evaluation")
    print("====================================================")
    print("Device:", device)
    print("Base loss:", base_ck.get("loss"))
    print("Role binding loss:", role_ck.get("loss"))
    print("Role gate:", role_ck.get("role_gate_threshold"))
    print("Entity margin gate:", role_ck.get("entity_margin_threshold"))

    semantic = strict = fluency = direct = 0

    for idx, case in enumerate(CASES, start=1):
        prompt = str(case["prompt"])
        reply, dbg = generate(
            model, tokenizer, intent_head, intent_labels, role_head, binding, role_ck, prompt
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
            f"role-binding={'ON' if dbg['active'] else 'OFF'}"
        )
        print(
            f"      CPU-GPU logit gap: base={dbg['base_gap']:+.3f} "
            f"bias={dbg['bias_gap']:+.3f} after={dbg['after_gap']:+.3f}"
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
