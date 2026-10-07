from __future__ import annotations

import argparse
import torch

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from chat import DEFAULT_MODEL, DEFAULT_TOKENIZER
from ndc_phase12 import build_ndc_centroids, classify_ndc_phase12
from ndc_taxonomy_phase12 import PHASE1, PHASE2


def main():
    ap = argparse.ArgumentParser(description="LLM_GPU v1.7.1 NDC Phase 1/2 classifier")
    ap.add_argument("text", nargs="*", help="text to classify")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--show-taxonomy", action="store_true")
    args = ap.parse_args()

    if args.show_taxonomy:
        print("Phase 1")
        for code, label in PHASE1.items():
            print(f"  {code}: {label}")
        print("\nPhase 2")
        for code, label in PHASE2.items():
            print(f"  {code}: {label}")
        return

    if not args.text:
        ap.error("text is required unless --show-taxonomy is used")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, ckpt = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()

    print("=" * 88)
    print(" LLM_GPU v1.7.1 NDC Phase 1/2 Semantic Classification")
    print("=" * 88)
    print("Device          :", device)
    print("Base loss       :", ckpt.get("loss"))
    print("Routing         : hierarchical 10 main classes -> 100 divisions")
    print("Model retraining: False")
    print()

    centroids = build_ndc_centroids(model, tokenizer)

    text = " ".join(args.text)
    r = classify_ndc_phase12(model, tokenizer, text, centroids)
    print("Input           :", r.text)
    print(f"Phase 1         : {r.phase1_code} {r.phase1_label}")
    print(f"  score/margin  : {r.phase1_score:.6f} / {r.phase1_margin:.6f}")
    print(f"Phase 2         : {r.phase2_code} {r.phase2_label}")
    print(f"  score/margin  : {r.phase2_score:.6f} / {r.phase2_margin:.6f}")
    print("Confidence      :", "ACCEPT" if r.confident else "REVIEW")


if __name__ == "__main__":
    main()
