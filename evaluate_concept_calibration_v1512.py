# evaluate_concept_calibration_v1512.py
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from chat import DEFAULT_MODEL, DEFAULT_TOKENIZER, semantic_vector
from train_concept_calibration_v1510 import CONCEPTS, HOLDOUT_SAMPLES, ConceptProjection


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--checkpoint", default="model/concept-calibration-v1512.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    base, _ = LanguageModel.load_checkpoint(args.model, device=device)
    base.eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    if ckpt.get("centroid_source") != "train":
        raise ValueError("checkpoint violates v1.5.12 protocol: centroid_source != train")
    if ckpt.get("holdout_used_for_centroid", True):
        raise ValueError("checkpoint violates v1.5.12 protocol: holdout used for centroid")

    projection = ConceptProjection(
        ckpt["input_dim"], ckpt["hidden_dim"], ckpt["output_dim"]
    ).to(device)
    projection.load_state_dict(ckpt["projection_state"])
    projection.eval()
    centroids = F.normalize(ckpt["centroids"].to(device), dim=-1)

    print("=" * 100)
    print(" LLM_GPU v1.5.12 Standardized Unseen-Holdout Evaluation")
    print("=" * 100)
    print("Evaluation protocol     :", ckpt["evaluation_protocol"])
    print("Centroid source         :", ckpt["centroid_source"].upper())
    print("Holdout used in fit     :", ckpt["holdout_used_for_centroid"])

    correct = 0
    margins = []
    for label_idx, label in enumerate(CONCEPTS):
        for text in HOLDOUT_SAMPLES[label]:
            raw = semantic_vector(base, tokenizer, text)
            z = projection(raw.unsqueeze(0))[0]
            sims = z @ centroids.T
            pred_idx = int(torch.argmax(sims).item())
            pred = CONCEPTS[pred_idx]

            gold_score = sims[label_idx]
            mask = torch.ones(len(CONCEPTS), dtype=torch.bool, device=device)
            mask[label_idx] = False
            margin = float((gold_score - sims[mask].max()).item())

            ok = pred == label
            correct += int(ok)
            margins.append(margin)
            print(
                f"{'PASS' if ok else 'FAIL'} "
                f"gold={label:<4s} pred={pred:<4s} "
                f"gold-margin={margin:.6f} :: {text}"
            )

    n = sum(len(v) for v in HOLDOUT_SAMPLES.values())
    print()
    print(f"Unseen holdout accuracy : {100.0*correct/n:.1f}%")
    print(f"Mean gold-class margin  : {sum(margins)/len(margins):.6f}")


if __name__ == "__main__":
    main()
