from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def confident(self) -> bool:
        return self.phase1_margin >= 0.01 and self.phase2_margin >= 0.01


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
        # Blend official main-class name with the mean of its ten Phase-2 divisions.
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
    """Hierarchical NDC classification: Phase 1 (10 classes) -> Phase 2 (100 divisions)."""
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
    )
