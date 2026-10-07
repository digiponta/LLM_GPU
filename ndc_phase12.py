from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import torch
import torch.nn.functional as F

from chat import semantic_vector
from ndc_taxonomy_phase12 import PHASE1, PHASE2, ANCHORS, phase2_children, validate_taxonomy


@dataclass
class NDCResult:
    text: str
    phase1_code: str
    phase1_label: str
    phase1_score: float
    phase1_margin: float
    phase2_code: str
    phase2_label: str
    phase2_score: float
    phase2_margin: float
    route_source: str = "semantic"
    matched_anchor: str = ""

    @property
    def confident(self) -> bool:
        if self.route_source == "anchor":
            return True
        return self.phase1_margin >= 0.01 and self.phase2_margin >= 0.01


def _normalize_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).lower()
    s = re.sub(r"[\s\u3000。、，,．.！？!?「」『』（）()\[\]{}]", "", s)
    return s


def lexical_ndc_hint(text: str):
    """Return (phase2_code, matched_anchor) for an explicit NDC anchor hit.

    Longest match wins so that '日本文学' is preferred over the broader '文学'.
    This is deterministic ontology routing, not learned prediction.
    """
    q = _normalize_text(text)
    hits = []
    for code, anchors in ANCHORS.items():
        for anchor in anchors:
            a = _normalize_text(anchor)
            if a and a in q:
                hits.append((len(a), code, anchor))
        label = PHASE2.get(code, "")
        a = _normalize_text(label)
        if a and a in q:
            hits.append((len(a), code, label))

    if not hits:
        return None

    hits.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, code, anchor = hits[0]
    return code, anchor


def _norm(v: torch.Tensor) -> torch.Tensor:
    if v.ndim != 1:
        v = v.reshape(-1)
    return F.normalize(v.float(), dim=0)


@torch.no_grad()
def _encode(model, tokenizer, text: str) -> torch.Tensor:
    return _norm(semantic_vector(model, tokenizer, text).detach())


@torch.no_grad()
def _centroid(model, tokenizer, texts: list[str]) -> torch.Tensor:
    zs = torch.stack([_encode(model, tokenizer, t) for t in texts])
    return _norm(zs.mean(dim=0))


@torch.no_grad()
def build_ndc_centroids(model, tokenizer):
    """Build semantic anchor centroids without changing model weights."""
    validate_taxonomy()

    phase2 = {}
    for code, label in PHASE2.items():
        texts = [label, f"{code} {label}"] + ANCHORS.get(code, [])
        phase2[code] = _centroid(model, tokenizer, texts)

    phase1 = {}
    for code, label in PHASE1.items():
        children = torch.stack([phase2[c] for c in phase2_children(code)])
        main = _centroid(model, tokenizer, [label, f"{code} {label}"])
        phase1[code] = _norm(0.5 * main + 0.5 * children.mean(dim=0))

    return phase1, phase2


def _rank(q: torch.Tensor, codes: list[str], table: dict[str, torch.Tensor]):
    scored = sorted(
        ((float(torch.dot(q, table[c]).item()), c) for c in codes),
        reverse=True,
    )
    best_score, best_code = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0
    return best_code, best_score, best_score - second_score


@torch.no_grad()
def classify_ndc_phase12(model, tokenizer, text: str, centroids=None) -> NDCResult:
    """Hybrid hierarchical NDC classification.

    1. Explicit known NDC terms use deterministic ontology/anchor routing.
    2. Otherwise fall back to semantic-vector hierarchical routing:
       Phase 1 (10 classes) -> Phase 2 (10 children).
    """
    hint = lexical_ndc_hint(text)
    if hint is not None:
        p2_code, anchor = hint
        p1_code = p2_code[0]
        return NDCResult(
            text=text,
            phase1_code=p1_code,
            phase1_label=PHASE1[p1_code],
            phase1_score=1.0,
            phase1_margin=1.0,
            phase2_code=p2_code,
            phase2_label=PHASE2[p2_code],
            phase2_score=1.0,
            phase2_margin=1.0,
            route_source="anchor",
            matched_anchor=anchor,
        )

    if centroids is None:
        centroids = build_ndc_centroids(model, tokenizer)
    p1_table, p2_table = centroids

    q = _encode(model, tokenizer, text)
    p1_code, p1_score, p1_margin = _rank(q, list(PHASE1), p1_table)
    child_codes = phase2_children(p1_code)
    p2_code, p2_score, p2_margin = _rank(q, child_codes, p2_table)

    return NDCResult(
        text=text,
        phase1_code=p1_code,
        phase1_label=PHASE1[p1_code],
        phase1_score=p1_score,
        phase1_margin=p1_margin,
        phase2_code=p2_code,
        phase2_label=PHASE2[p2_code],
        phase2_score=p2_score,
        phase2_margin=p2_margin,
        route_source="semantic",
        matched_anchor="",
    )
