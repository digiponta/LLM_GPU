# concept_eval_protocol_v1512.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class EvalResult:
    accuracy: float
    mean_margin: float
    centroids: torch.Tensor
    predictions: torch.Tensor
    margins: list[float]


def class_centroids(z: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build centroids ONLY from the training/reference split."""
    centroids = []
    for i in range(num_classes):
        members = z[y == i]
        if members.numel() == 0:
            raise ValueError(f"no training samples for class index {i}")
        centroids.append(F.normalize(members.mean(dim=0), dim=0))
    return torch.stack(centroids)


@torch.no_grad()
def evaluate_train_centroid_holdout(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    holdout_x: torch.Tensor,
    holdout_y: torch.Tensor,
    num_classes: int,
    projection=None,
) -> EvalResult:
    """Canonical v1.5.12+ evaluation protocol.

    1. Fit centroids from TRAIN only.
    2. Never use holdout vectors/labels to construct centroids.
    3. Apply those fixed train centroids to unseen HOLDOUT.
    """
    if projection is None:
        train_z = F.normalize(train_x, dim=-1)
        holdout_z = F.normalize(holdout_x, dim=-1)
    else:
        projection.eval()
        train_z = projection(train_x)
        holdout_z = projection(holdout_x)

    centroids = class_centroids(train_z, train_y, num_classes)
    sims = holdout_z @ centroids.T
    pred = sims.argmax(dim=1)
    accuracy = (pred == holdout_y).float().mean().item()

    margins: list[float] = []
    for i in range(len(holdout_y)):
        gold = int(holdout_y[i].item())
        gold_score = sims[i, gold]
        mask = torch.ones(num_classes, dtype=torch.bool, device=sims.device)
        mask[gold] = False
        best_other = sims[i, mask].max()
        margins.append(float((gold_score - best_other).item()))

    return EvalResult(
        accuracy=accuracy,
        mean_margin=sum(margins) / len(margins) if margins else 0.0,
        centroids=centroids,
        predictions=pred,
        margins=margins,
    )


@torch.no_grad()
def evaluate_train_split(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    projection=None,
) -> EvalResult:
    """Diagnostic only: train centroids evaluated on train data.

    Do not compare this metric to unseen holdout accuracy as if equivalent.
    """
    return evaluate_train_centroid_holdout(
        train_x=train_x,
        train_y=train_y,
        holdout_x=train_x,
        holdout_y=train_y,
        num_classes=num_classes,
        projection=projection,
    )
