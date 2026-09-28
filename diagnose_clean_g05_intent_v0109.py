# diagnose_clean_g05_intent_v0109.py
#
# Focused diagnostic for the rebuilt clean-v0.8 G05 reverse-identification case.
# No training is performed.

from __future__ import annotations

from pathlib import Path

import torch

from chat import AI_PREFIX, USER_PREFIX, generate_reply
from evaluate_intent_v09 import build_intent_head, predict_intent
from model import LanguageModel
from tokenizer_bpe import Tokenizer


TOKENIZER = "model/tokenizer-v0.7-bpe.json"
MODEL = "model/model-gpu-v0.8-chat-clean.pt"
INTENT_HEAD = "model/model-gpu-v0.8-intent-head-clean.pt"

G05 = "コンピュータの中心で多様な命令を処理する装置は何ですか。"
EXPECTED = {"tech_cpu", "property_general"}


def main():
    for filename in (TOKENIZER, MODEL, INTENT_HEAD):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(TOKENIZER)
    model, model_checkpoint = LanguageModel.load_checkpoint(MODEL, device=device)

    head_checkpoint = torch.load(INTENT_HEAD, map_location=device)
    labels = list(head_checkpoint["labels"])
    head = build_intent_head(model.d_model, len(labels)).to(device)
    head.load_state_dict(head_checkpoint["state_dict"])
    head.eval()
    model.eval()

    threshold = float(head_checkpoint.get("threshold", 0.5))
    predicted, ranked = predict_intent(
        model,
        tokenizer,
        head,
        labels,
        G05,
        threshold,
        len(labels),
    )
    probs = dict(ranked)

    prompt = f"{USER_PREFIX}{G05}\n{AI_PREFIX}"
    reply, _ = generate_reply(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=96,
        temperature=0.0,
        top_k=1,
        repetition_penalty=1.05,
    )

    cpu = float(probs.get("tech_cpu", 0.0))
    gpu = float(probs.get("tech_gpu", 0.0))
    general = float(probs.get("property_general", 0.0))
    parallel = float(probs.get("property_parallel", 0.0))

    print()
    print("====================================================")
    print(" Clean v0.8 G05 Intent Diagnostic v0.10.9")
    print("====================================================")
    print("Device             :", device)
    print("Model loss         :", model_checkpoint.get("loss"))
    print("Intent-head loss   :", head_checkpoint.get("loss"))
    print("Threshold          :", threshold)
    print("Prompt             :", G05)
    print("Expected tags      :", ", ".join(sorted(EXPECTED)))
    print("Predicted tags     :", ", ".join(sorted(predicted)) if predicted else "(none)")
    print()
    print("Focused probabilities")
    print("---------------------")
    print(f"tech_cpu           : {cpu:.6f}")
    print(f"tech_gpu           : {gpu:.6f}")
    print(f"CPU-GPU margin     : {cpu - gpu:+.6f}")
    print(f"property_general   : {general:.6f}")
    print(f"property_parallel  : {parallel:.6f}")
    print(f"general-parallel   : {general - parallel:+.6f}")
    print()
    print("Top intent probabilities")
    print("------------------------")
    for label, prob in ranked[:10]:
        print(f"{label:22s}: {prob:.6f}")
    print()
    print("Generated answer")
    print("----------------")
    print(reply)
    print()
    print("Interpretation")
    print("--------------")
    if cpu > gpu and general > parallel:
        print("Intent representation favors CPU/general-purpose semantics.")
        if reply.startswith("CPU") or "CPU" in reply:
            print("Generation is also CPU-like: representation and generation agree.")
        else:
            print("Generation is not CPU-like: bottleneck is representation-to-generation binding.")
    elif gpu >= cpu:
        print("Intent representation itself favors or ties GPU: reverse-identification remains a representation problem.")
    else:
        print("Mixed intent evidence: inspect the full v0.10.9 intent report before changing generation.")


if __name__ == "__main__":
    main()
