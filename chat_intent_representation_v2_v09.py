# chat_intent_representation_v2_v09.py
#
# Inference helpers for LLM_GPU v0.9 Intent Representation v2.

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from intent_representation_v2_v09 import (
    forward_representation_v2_conditioned,
    infer_intent_representation_v2,
)


USER_PREFIX = "人: "
AI_PREFIX = "AI: "


@torch.no_grad()
def compute_conditioning_bias(
    intent_model,
    tokenizer,
    intent_head,
    projection,
    prompt,
):
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    context = prompt_ids[-intent_model.context_length:]
    device = next(intent_model.parameters()).device
    x = torch.tensor([context], dtype=torch.long, device=device)

    intent_prob, semantic_hidden = infer_intent_representation_v2(
        intent_model,
        intent_head,
        x,
        prompt_index=None,
    )
    bias = projection(intent_prob, semantic_hidden)
    return bias, intent_prob


@torch.no_grad()
def generate_reply(
    generation_model,
    intent_model,
    tokenizer,
    intent_head,
    projection,
    prompt,
    max_new_tokens=96,
    temperature=0.45,
    top_k=20,
    repetition_penalty=1.05,
):
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    generated = list(prompt_ids)
    response_ids: List[int] = []
    device = next(generation_model.parameters()).device

    generation_model.eval()
    intent_model.eval()
    intent_head.eval()
    projection.eval()

    conditioning_bias, intent_prob = compute_conditioning_bias(
        intent_model,
        tokenizer,
        intent_head,
        projection,
        prompt,
    )

    for _ in range(max_new_tokens):
        context = generated[-generation_model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)

        hidden = forward_representation_v2_conditioned(
            generation_model,
            x,
            conditioning_bias,
            projection.inject_after,
        )
        logits = generation_model.lm_head(hidden[:, -1, :])[0].clone()

        if repetition_penalty != 1.0:
            for token_id in set(response_ids):
                if logits[token_id] >= 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

        if temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_k is not None and 0 < top_k < logits.numel():
                values, indices = torch.topk(logits, top_k)
                probs = F.softmax(values, dim=-1)
                selected = torch.multinomial(probs, 1)
                next_id = int(indices[selected].item())
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())

        if next_id == tokenizer.eos_id:
            break

        generated.append(next_id)
        response_ids.append(next_id)

        decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
        if "\n" in decoded:
            break

    reply = tokenizer.decode(response_ids, skip_special_tokens=True)
    for marker in ("\n人:", "\nAI:", "\n"):
        if marker in reply:
            reply = reply.split(marker, 1)[0]

    return reply.strip(), len(response_ids), intent_prob[0].tolist()
