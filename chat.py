# chat.py
#
# Interactive chat interface for the current LLM_GPU conversational checkpoint.
# Defaults to the v0.8 cleaned chat model used by the v1.4/v1.5 experiments.
#
# v1.5.7 additions:
#   - conservative chat-level Unknown rejection
#   - multiple probe generations
#   - token-confidence / response-agreement checks
#   - malformed / repetitive output detection
#   - fallback response: "未学習です"
#
# Note:
# The v1.4.32/v1.5.1 semantic risk predictors are contrast-routing specific.
# They cannot be applied directly to arbitrary chat prompts.  This file uses
# a chat-level conservative rejection gate inspired by the same selective
# prediction principle.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer_bpe import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "
UNKNOWN_REPLY = "未学習です"


@dataclass
class GenerationResult:
    text: str
    token_count: int
    mean_confidence: float
    min_confidence: float
    mean_top2_margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the current LLM_GPU conversational model."
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--history-turns",
        type=int,
        default=3,
        help="Number of previous turns included in the prompt.",
    )

    # Conservative selective-prediction gate.
    parser.add_argument(
        "--unknown-rejection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable conservative chat-level Unknown rejection.",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=3,
        help="Number of generation probes used for agreement.",
    )
    parser.add_argument(
        "--probe-temperature",
        type=float,
        default=0.30,
        help="Sampling temperature for reliability probes.",
    )
    parser.add_argument(
        "--probe-top-k",
        type=int,
        default=10,
        help="Top-k used for reliability probes.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.18,
        help="Minimum mean selected-token probability.",
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=0.35,
        help="Minimum mean textual agreement between probe responses.",
    )
    parser.add_argument(
        "--semantic-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable question-answer semantic consistency checks.",
    )
    parser.add_argument(
        "--history-contamination-margin",
        type=float,
        default=0.05,
        help=(
            "Reject if the answer is closer to a previous user turn than "
            "the current turn by this cosine-similarity margin."
        ),
    )
    parser.add_argument(
        "--show-risk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show confidence/agreement diagnostics.",
    )
    return parser.parse_args()


def _slot_overlap(a: list[str], b: list[str]) -> bool:
    aa = {x.lower() for x in a}
    bb = {x.lower() for x in b}
    return bool(aa & bb)


def select_relevant_history(
    history: List[Tuple[str, str]],
    user_text: str,
    history_turns: int,
) -> List[Tuple[str, str]]:
    """v1.5.5 minimal-context policy.

    definition  : no raw history
    comparison  : no raw history
    greeting    : at most one previous greeting
    general     : at most one highly similar previous user turn
    """
    if history_turns <= 0 or not history:
        return []

    current_intent, _ = classify_intent_and_slots(user_text)

    if current_intent in ("definition", "comparison"):
        return []

    if current_intent == "greeting":
        for old_user, old_ai in reversed(history):
            old_intent, _ = classify_intent_and_slots(old_user)
            if old_intent == "greeting":
                return [(old_user, old_ai)]
        return []

    # General question: keep only one closely related prior user turn.
    # Character bigram similarity is deliberately conservative and cheap.
    best: Tuple[str, str] | None = None
    best_score = 0.0
    for old_user, old_ai in history:
        score = response_similarity(user_text, old_user)
        if score > best_score:
            best_score = score
            best = (old_user, old_ai)

    if best is not None and best_score >= 0.55:
        return [best]

    return []


def build_prompt(
    history: List[Tuple[str, str]],
    user_text: str,
    history_turns: int,
) -> tuple[str, List[Tuple[str, str]]]:
    chunks: List[str] = []
    selected = select_relevant_history(
        history=history,
        user_text=user_text,
        history_turns=history_turns,
    )

    for old_user, old_ai in selected:
        chunks.append(
            f"{USER_PREFIX}{old_user}\n"
            f"{AI_PREFIX}{old_ai}\n"
        )

    chunks.append(f"{USER_PREFIX}{user_text}\n{AI_PREFIX}")
    return "".join(chunks), selected


def _clean_reply(reply: str) -> str:
    for marker in ("\n人:", "\nAI:", "\n"):
        if marker in reply:
            reply = reply.split(marker, 1)[0]
    return reply.strip()


@torch.no_grad()
def generate_reply(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    seed: int | None = None,
) -> GenerationResult:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    generated = list(prompt_ids)
    response_ids: List[int] = []

    confidences: List[float] = []
    margins: List[float] = []

    model.eval()
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        context = generated[-model.context_length:]
        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device,
        )

        raw_logits = model(x)[0, -1, :].clone()

        if repetition_penalty != 1.0:
            for token_id in set(response_ids):
                if 0 <= token_id < raw_logits.numel():
                    if raw_logits[token_id] >= 0:
                        raw_logits[token_id] /= repetition_penalty
                    else:
                        raw_logits[token_id] *= repetition_penalty

        # Confidence is always measured on the untempered model distribution.
        raw_probs = F.softmax(raw_logits, dim=-1)
        top2_values, _ = torch.topk(raw_probs, k=2)

        if temperature <= 0:
            next_id = int(torch.argmax(raw_logits).item())
        else:
            logits = raw_logits / temperature
            if 0 < top_k < logits.numel():
                values, indices = torch.topk(logits, top_k)
                probs = F.softmax(values, dim=-1)
                selected = torch.multinomial(probs, 1)
                next_id = int(indices[selected].item())
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())

        if next_id == tokenizer.eos_id:
            break

        confidences.append(float(raw_probs[next_id].item()))
        margins.append(float((top2_values[0] - top2_values[1]).item()))

        generated.append(next_id)
        response_ids.append(next_id)

        decoded = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )
        if "\n" in decoded:
            break

    reply = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    )
    reply = _clean_reply(reply)

    mean_conf = (
        sum(confidences) / len(confidences)
        if confidences else 0.0
    )
    min_conf = min(confidences) if confidences else 0.0
    mean_margin = (
        sum(margins) / len(margins)
        if margins else 0.0
    )

    return GenerationResult(
        text=reply,
        token_count=len(response_ids),
        mean_confidence=mean_conf,
        min_confidence=min_conf,
        mean_top2_margin=mean_margin,
    )


def _normalize_for_similarity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[。、,.!?！？:：;；「」『』（）()\[\]{}]", "", text)
    return text


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    text = _normalize_for_similarity(text)
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text)-n+1)}


def response_similarity(a: str, b: str) -> float:
    aa = _char_ngrams(a)
    bb = _char_ngrams(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def mean_pairwise_agreement(results: List[GenerationResult]) -> float:
    if len(results) < 2:
        return 1.0
    values: List[float] = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            values.append(
                response_similarity(results[i].text, results[j].text)
            )
    return sum(values) / len(values) if values else 1.0


def malformed_or_unstable(text: str) -> bool:
    if not text:
        return True

    # Unicode replacement character is a strong malformed-output signal.
    if "�" in text:
        return True

    # Reject pathological character/token repetition.
    compact = _normalize_for_similarity(text)
    if len(compact) >= 8:
        for n in (1, 2, 3, 4):
            chunks = [
                compact[i:i+n]
                for i in range(0, len(compact)-n+1)
            ]
            if chunks:
                most = max(chunks.count(x) for x in set(chunks))
                if most / len(chunks) > 0.45:
                    return True

    return False


@torch.no_grad()
def semantic_vector(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    block_index: int = 4,
) -> torch.Tensor:
    """Mean pooled hidden representation at block5 (index 4)."""
    ids = tokenizer.encode(text, add_bos=True)
    if not ids:
        return torch.zeros(model.d_model, device=next(model.parameters()).device)

    ids = ids[-model.context_length:]
    device = next(model.parameters()).device
    x_ids = torch.tensor([ids], dtype=torch.long, device=device)

    x = model.embedding(x_ids)
    if model.position_embedding is not None:
        positions = torch.arange(x_ids.shape[1], device=device)
        x = x + model.position_embedding(positions).unsqueeze(0)

    last = min(block_index, len(model.blocks)-1)
    for i in range(last + 1):
        x = model.blocks[i](x)

    # Exclude BOS when possible.
    token_states = x[0, 1:, :] if x.shape[1] > 1 else x[0]
    v = token_states.mean(dim=0)
    return F.normalize(v, dim=0)


SLOT_ALIASES = {
    "ai": ("ai", "人工知能", "artificial intelligence"),
    "llm": ("llm", "大規模言語モデル", "large language model", "言語モデル"),
    "cpu": ("cpu", "中央処理装置", "central processing unit"),
    "gpu": ("gpu", "画像処理装置", "graphics processing unit"),
    "cuda": ("cuda",),
}

KNOWN_ENTITY_TOKENS = {
    "AI", "LLM", "CPU", "GPU", "CUDA",
}


def _clean_slot(slot: str) -> str:
    slot = slot.strip()
    slot = re.sub(r"(?:を|は|が|の)$", "", slot)
    return slot.strip()


def slot_aliases(slot: str) -> tuple[str, ...]:
    key = _clean_slot(slot).lower()
    return SLOT_ALIASES.get(key, (key,))


def slot_present(slot: str, answer: str) -> bool:
    answer_lower = answer.lower()
    return any(alias.lower() in answer_lower for alias in slot_aliases(slot))


def extract_suspicious_entities(text: str) -> list[str]:
    """Extract novel-looking identifiers whose identity should survive generation."""
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_+.#-]{1,39}", text)
    entities: list[str] = []
    for token in candidates:
        upper = token.upper()
        if upper in KNOWN_ENTITY_TOKENS:
            continue

        # Conservative: require digits/hyphen, or mixed-case identifier shape.
        suspicious = (
            any(ch.isdigit() for ch in token)
            or "-" in token
            or (
                any(ch.islower() for ch in token)
                and any(ch.isupper() for ch in token[1:])
                and len(token) >= 5
            )
        )
        if suspicious and token not in entities:
            entities.append(token)
    return entities


def extract_definition_focus(text: str) -> str | None:
    """Extract focus from Japanese definition prompts such as 'GPUとは'."""
    compact = text.strip()
    patterns = [
        r"^\s*([A-Za-z0-9_+.#\-]{2,40})\s*(?:とは|って|とは何|って何)",
        r"^\s*([^\s、。！？?]{1,40})\s*(?:とは|って)\s*[？?]?$",
    ]
    for p in patterns:
        m = re.search(p, compact, flags=re.IGNORECASE)
        if m:
            return _clean_slot(m.group(1))
    return None


def classify_intent_and_slots(question: str) -> tuple[str, list[str]]:
    q = question.strip()
    q_lower = q.lower()

    greeting_patterns = (
        "こんにちは", "こんばんは", "おはよう", "お疲れ", "はじめまして",
        "やあ", "hello", "hi",
    )
    if any(x in q_lower for x in greeting_patterns):
        return "greeting", []

    # Comparison: "AとBの違い", "AとBを比較", "AとBの差"
    m = re.search(
        r"^\s*([^\s、。！？?と]{1,30})\s*と\s*([^\s、。！？?]{1,30}?)"
        r"\s*(?:を|の)?(?:違い|差|比較|違う点)",
        q,
        flags=re.IGNORECASE,
    )
    if m:
        return "comparison", [
            _clean_slot(m.group(1)),
            _clean_slot(m.group(2)),
        ]

    focus = extract_definition_focus(q)
    if focus:
        return "definition", [focus]

    return "general", []


def greeting_consistent(answer: str) -> bool:
    a = answer.lower()
    greeting_terms = (
        "こんにちは", "こんばんは", "おはよう", "お疲れ", "はじめまして",
        "よろしく", "話しましょう", "何について", "hello", "hi",
    )
    return any(x in a for x in greeting_terms)


def slot_coverage_check(
    intent: str,
    slots: list[str],
    answer: str,
) -> tuple[bool, float, str]:
    if intent == "greeting":
        if greeting_consistent(answer):
            return True, 1.0, "greeting matched"
        return False, 0.0, "greeting intent mismatch"

    if not slots:
        return True, 1.0, "no required slots"

    hits = [slot for slot in slots if slot_present(slot, answer)]
    coverage = len(hits) / len(slots)

    if intent == "definition":
        if coverage < 1.0:
            return False, coverage, f"definition focus missing: {slots[0]}"
        return True, coverage, "definition slot covered"

    if intent == "comparison":
        if coverage < 1.0:
            missing = [s for s in slots if not slot_present(s, answer)]
            return (
                False,
                coverage,
                "comparison slot missing: " + ", ".join(missing),
            )

        comparison_cues = (
            "一方", "対して", "違", "比べ", "比較", "対照", "より",
            "得意", "汎用", "並列", "制御",
        )
        if not any(cue in answer for cue in comparison_cues):
            return False, coverage, "comparison relation missing"

        return True, coverage, "comparison slots covered"

    return True, coverage, "slots covered"


def focus_consistent(question: str, answer: str) -> tuple[bool, str]:
    focus = extract_definition_focus(question)
    if not focus:
        return True, "no explicit focus"

    if focus.lower() in answer.lower():
        return True, "focus preserved"

    # Strong mismatch for acronym/technical-term definition questions.
    if re.fullmatch(r"[A-Za-z0-9_+.#\-]{2,20}", focus):
        return False, f"definition focus missing: {focus}"

    return True, "focus not mandatory"


def semantic_consistency_check(
    model: LanguageModel,
    tokenizer: Tokenizer,
    current_question: str,
    answer: str,
    history: List[Tuple[str, str]],
    contamination_margin: float,
) -> tuple[bool, float, float, str, str, list[str], float]:
    intent, slots = classify_intent_and_slots(current_question)

    suspicious_entities = extract_suspicious_entities(current_question)
    missing_entities = [
        entity
        for entity in suspicious_entities
        if entity.lower() not in answer.lower()
    ]
    if missing_entities:
        return (
            False, 0.0, 0.0,
            "unknown entity missing: " + ", ".join(missing_entities),
            intent, slots, 0.0,
        )

    slots_ok, slot_coverage, slot_reason = slot_coverage_check(
        intent, slots, answer
    )
    if not slots_ok:
        return (
            False, 0.0, 0.0, slot_reason,
            intent, slots, slot_coverage,
        )

    qv = semantic_vector(model, tokenizer, current_question)
    av = semantic_vector(model, tokenizer, answer)
    current_sim = float(torch.dot(qv, av).item())

    # Greeting turns are intentionally short and semantically broad.
    # Slot/intent matching is more reliable than history similarity here.
    if intent == "greeting":
        return (
            True, current_sim, -1.0, slot_reason,
            intent, slots, slot_coverage,
        )

    previous_sims: List[float] = []
    for old_user, _ in history[-3:]:
        pv = semantic_vector(model, tokenizer, old_user)
        previous_sims.append(float(torch.dot(pv, av).item()))

    previous_best = max(previous_sims) if previous_sims else -1.0

    if previous_sims and previous_best > current_sim + contamination_margin:
        return (
            False,
            current_sim,
            previous_best,
            "history contamination",
            intent,
            slots,
            slot_coverage,
        )

    return (
        True,
        current_sim,
        previous_best,
        slot_reason,
        intent,
        slots,
        slot_coverage,
    )


def evaluate_unknown_gate(
    results: List[GenerationResult],
    min_confidence: float,
    min_agreement: float,
) -> Tuple[bool, float, float, str]:
    if not results:
        return False, 0.0, 0.0, "no generation"

    primary = results[0]

    if malformed_or_unstable(primary.text):
        return False, primary.mean_confidence, 0.0, "malformed/repetitive output"

    agreement = mean_pairwise_agreement(results)

    if primary.mean_confidence < min_confidence:
        return (
            False,
            primary.mean_confidence,
            agreement,
            "low token confidence",
        )

    if agreement < min_agreement:
        return (
            False,
            primary.mean_confidence,
            agreement,
            "low response agreement",
        )

    return True, primary.mean_confidence, agreement, "accepted"


def print_info(
    model: LanguageModel,
    tokenizer: Tokenizer,
    checkpoint: dict,
    device: torch.device,
    model_path: Path,
    tokenizer_path: Path,
    args: argparse.Namespace,
) -> None:
    print()
    print("==============================================")
    print(" LLM_GPU Chat - v1.5.7 Robust Slot Parsing / Unknown Entity Gate")
    print("==============================================")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Model           :", model_path)
    print("Tokenizer       :", tokenizer_path)
    print("Vocabulary size :", tokenizer.vocab_size)
    print("Parameters      :", f"{model.parameter_count:,}")
    print("Context length  :", model.context_length)
    print("d_model         :", model.d_model)
    print("Layers          :", model.num_layers)
    print("Attention heads :", model.num_heads)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Unknown reject  :", args.unknown_rejection)
    if args.unknown_rejection:
        print("Probe count     :", args.probe_count)
        print("Min confidence  :", args.min_confidence)
        print("Min agreement   :", args.min_agreement)
        print("Fallback        :", UNKNOWN_REPLY)
        print("Context policy  : minimal")
    print()


def main() -> None:
    args = parse_args()

    tokenizer_path = Path(args.tokenizer)
    model_path = Path(args.model)

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Chat model not found: {model_path}\n"
            "Expected the cleaned v0.8 conversational checkpoint."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(model_path),
        device=device,
    )

    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    print_info(
        model=model,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        device=device,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        args=args,
    )

    print("Commands:")
    print("  /reset  clear conversation history")
    print("  /info   show model/checkpoint information")
    print("  /exit   quit")
    print()

    history: List[Tuple[str, str]] = []

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue

        command = user_text.lower()

        if command in ("/exit", "exit", "quit"):
            break

        if command == "/reset":
            history.clear()
            print("[conversation history cleared]")
            print()
            continue

        if command == "/info":
            print_info(
                model=model,
                tokenizer=tokenizer,
                checkpoint=checkpoint,
                device=device,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                args=args,
            )
            continue

        prompt, selected_history = build_prompt(
            history=history,
            user_text=user_text,
            history_turns=args.history_turns,
        )

        start = time.perf_counter()

        # Primary response: greedy when rejection is enabled so that the
        # displayed candidate itself is deterministic and reproducible.
        primary_temperature = 0.0 if args.unknown_rejection else args.temperature
        results = [
            generate_reply(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=primary_temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                seed=0,
            )
        ]

        if args.unknown_rejection:
            for probe in range(max(0, args.probe_count - 1)):
                results.append(
                    generate_reply(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.probe_temperature,
                        top_k=args.probe_top_k,
                        repetition_penalty=args.repetition_penalty,
                        seed=1000 + probe,
                    )
                )

        primary = results[0]

        accepted = True
        confidence = primary.mean_confidence
        agreement = 1.0
        reason = "rejection disabled"
        qa_similarity = 0.0
        previous_similarity = -1.0
        intent = "general"
        slots: list[str] = []
        slot_coverage = 1.0

        if args.unknown_rejection:
            accepted, confidence, agreement, reason = evaluate_unknown_gate(
                results,
                min_confidence=args.min_confidence,
                min_agreement=args.min_agreement,
            )

        if args.semantic_consistency:
            # Intent/slot analysis is useful even when the generation gate
            # already rejected the response, so always compute metadata.
            (
                semantic_ok,
                qa_similarity,
                previous_similarity,
                semantic_reason,
                intent,
                slots,
                slot_coverage,
            ) = semantic_consistency_check(
                model=model,
                tokenizer=tokenizer,
                current_question=user_text,
                answer=primary.text,
                history=history,
                contamination_margin=args.history_contamination_margin,
            )
            if accepted and not semantic_ok:
                accepted = False
                reason = semantic_reason
            elif accepted:
                reason = semantic_reason

        reply = primary.text if accepted else UNKNOWN_REPLY
        new_tokens = sum(r.token_count for r in results)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        rate = new_tokens / elapsed if elapsed > 0 else 0.0

        if not reply:
            reply = UNKNOWN_REPLY if args.unknown_rejection else "(no response)"

        print(f"AI> {reply}")

        if args.show_risk and args.unknown_rejection:
            status = "KNOWN" if accepted else "UNKNOWN"
            semantic_part = ""
            if args.semantic_consistency:
                slot_text = "|".join(slots) if slots else "-"
                semantic_part = (
                    f", intent={intent}"
                    f", slots={slot_text}"
                    f", slot_cov={slot_coverage:.2f}"
                    f", qa_sim={qa_similarity:.3f}"
                    f", prev_sim={previous_similarity:.3f}"
                )
            print(
                f"[gate={status}, "
                f"confidence={confidence:.3f}, "
                f"agreement={agreement:.3f}"
                f"{semantic_part}, "
                f"context_turns={len(selected_history)}, "
                f"reason={reason}]"
            )

        print(
            f"[{new_tokens} generated probe tokens, "
            f"{elapsed:.2f}s, "
            f"{rate:.1f} tok/s]"
        )
        print()

        # v1.5.4 clean-history policy:
        # only accepted KNOWN turns may enter future generation context.
        # UNKNOWN/rejected turns are intentionally discarded.
        if accepted:
            history.append((user_text, reply))


if __name__ == "__main__":
    main()
