# train_semantic_encoder_adapter_v061.py
#
# Semantic Encoder Adapter v0.6.1: Instruction Binding Refinement
#
# Continues from v0.6 and specifically strengthens:
#   heterogeneous_instruction > homogeneous_computation
# for diverse/mixed instruction prompts, without using the exact G05 prompt.

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v06 import (
    HIERARCHY_LABELS,
    load_semantic_adapter_v06_checkpoint,
)
from semantic_encoder_adapter_v061 import save_semantic_adapter_v061_checkpoint
from tokenizer_bpe import Tokenizer
from train_semantic_encoder_adapter_v03 import build_rows
from train_semantic_encoder_adapter_v01 import AdapterDataset


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INIT = "model/model-gpu-v0.9.1-semantic-adapter-v06.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-semantic-adapter-v061.pt"
SEED = 42


# target style: +1 means hetero > homo, -1 means homo > hetero.
REFINEMENT_ROWS: List[Tuple[str, int]] = [
    ("コンピュータの中心で多種類の命令を処理するプロセッサはCPUです。", +1),
    ("CPUは演算だけでなく、制御、分岐、メモリ操作など異なる命令を扱います。", +1),
    ("多様な命令を扱うとは、同じ計算を繰り返すことだけではありません。", +1),
    ("CPUの命令処理には、計算命令と非計算命令の両方が含まれます。", +1),
    ("CPUは算術命令に加えて、比較、ジャンプ、ロード、ストアも実行します。", +1),
    ("汎用プロセッサは異なる種類の命令を組み合わせて処理します。", +1),
    ("制御フロー、データ移動、メモリアクセスを含む多様な命令処理はCPU向きです。", +1),
    ("CPUの役割は数値計算だけではなく、プログラム全体の制御も含みます。", +1),
    ("異種命令を柔軟に実行することがCPUの特徴です。", +1),
    ("分岐やI/Oを含む命令列を扱うのは、同種演算の大量並列とは異なる仕事です。", +1),

    ("同じ演算を多数のデータへ繰り返し適用する処理はGPU向きです。", -1),
    ("多数の似た計算を並列実行するのがGPUの得意分野です。", -1),
    ("GPUは多様な制御命令より、同型の演算を大量並列に処理することを重視します。", -1),
    ("homogeneous computationを高スループットで実行するのがGPUの強みです。", -1),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Semantic Encoder Adapter v0.6.1."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--init-adapter", default=DEFAULT_INIT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--hierarchy-weight", type=float, default=0.75)
    p.add_argument("--direct-binding-weight", type=float, default=1.5)
    p.add_argument("--direct-binding-margin", type=float, default=0.30)
    p.add_argument("--hierarchy-preservation-weight", type=float, default=0.50)
    p.add_argument("--representation-preservation-weight", type=float, default=0.50)
    return p.parse_args()


@torch.no_grad()
def encode_prompt_hidden(model, tokenizer, prompt):
    device = next(model.parameters()).device
    text = f"人: {prompt}\nAI: "
    ids = tokenizer.encode(text, add_bos=True)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = model.forward_hidden(x)
    return hidden[:, -1, :][0].detach()


def exact_overlap():
    dev = {str(case["prompt"]) for case in CASES}
    return sorted(prompt for prompt, _ in REFINEMENT_ROWS if prompt in dev)


def binding_loss(logits, styles, margin):
    probs = torch.sigmoid(logits)
    hetero = HIERARCHY_LABELS.index("heterogeneous_instruction")
    homo = HIERARCHY_LABELS.index("homogeneous_computation")

    h = probs[:, hetero]
    m = probs[:, homo]

    cpu_mask = styles > 0
    gpu_mask = styles < 0

    losses = []
    if cpu_mask.any():
        losses.append(F.relu(margin + m[cpu_mask] - h[cpu_mask]).mean())
    if gpu_mask.any():
        losses.append(F.relu(margin + h[gpu_mask] - m[gpu_mask]).mean())

    return torch.stack(losses).mean()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model, args.init_adapter):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    overlaps = exact_overlap()
    if overlaps:
        raise RuntimeError("Exact overlap with fixed DEV prompts: " + repr(overlaps))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    base_model, base_checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    adapter, heads, hierarchy_head, init_checkpoint = (
        load_semantic_adapter_v06_checkpoint(args.init_adapter, device)
    )

    # Concept/attribute heads remain fixed in this refinement.
    heads.eval()
    for p in heads.parameters():
        p.requires_grad_(False)

    refinement_hidden = torch.stack(
        [
            encode_prompt_hidden(base_model, tokenizer, prompt)
            for prompt, _style in REFINEMENT_ROWS
        ],
        dim=0,
    ).to(device)
    refinement_styles = torch.tensor(
        [style for _prompt, style in REFINEMENT_ROWS],
        dtype=torch.long,
        device=device,
    )

    # Preserve the full v0.6 hierarchy behavior on the existing semantic bank.
    base_rows = build_rows()
    base_set = AdapterDataset(base_rows, base_model, tokenizer)
    preserve_hidden = torch.stack(
        [item[0] for item in base_set.items],
        dim=0,
    ).to(device)

    adapter.eval()
    hierarchy_head.eval()
    with torch.no_grad():
        start_refinement_adapted = adapter(refinement_hidden)
        start_refinement_logits = hierarchy_head(start_refinement_adapted)
        refinement_targets = torch.sigmoid(start_refinement_logits).detach()

        start_preserve_adapted = adapter(preserve_hidden).detach()
        start_preserve_hierarchy = torch.sigmoid(
            hierarchy_head(start_preserve_adapted)
        ).detach()

    parameters = list(adapter.parameters()) + list(hierarchy_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.6.1 Training")
    print("====================================================")
    print("Device                         :", device)
    if device.type == "cuda":
        print("GPU                            :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss           :", base_checkpoint.get("loss"))
    print("Initialization                 :", args.init_adapter)
    print("Initial adapter loss           :", init_checkpoint.get("loss"))
    print("Base encoder                   : frozen")
    print("Concept/attribute heads        : frozen")
    print("Adapter + hierarchy head       : trainable")
    print("Refinement rows                :", len(REFINEMENT_ROWS))
    print("Exact DEV overlap              :", len(overlaps))
    print("Learning rate                  :", args.lr)
    print("Direct binding margin          :", args.direct_binding_margin)
    print("Direct binding weight          :", args.direct_binding_weight)
    print("Hierarchy preservation weight  :", args.hierarchy_preservation_weight)
    print("Representation preservation    :", args.representation_preservation_weight)
    print()

    best_loss = float("inf")
    best_epoch = 0
    best_adapter = None
    best_hierarchy = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        hierarchy_head.train()

        optimizer.zero_grad(set_to_none=True)

        adapted = adapter(refinement_hidden)
        logits = hierarchy_head(adapted)

        # Soft preservation on all seven axes, plus hard direct binding.
        hierarchy_loss = F.binary_cross_entropy_with_logits(
            logits,
            refinement_targets,
        )
        direct = binding_loss(
            logits,
            refinement_styles,
            args.direct_binding_margin,
        )

        preserved_adapted = adapter(preserve_hidden)
        preserved_probs = torch.sigmoid(hierarchy_head(preserved_adapted))
        hierarchy_preservation = F.mse_loss(
            preserved_probs,
            start_preserve_hierarchy,
        )
        representation_preservation = (
            1.0
            - F.cosine_similarity(
                preserved_adapted,
                start_preserve_adapted,
                dim=-1,
            ).mean()
        )

        total = (
            args.hierarchy_weight * hierarchy_loss
            + args.direct_binding_weight * direct
            + args.hierarchy_preservation_weight * hierarchy_preservation
            + args.representation_preservation_weight
            * representation_preservation
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        value = float(total.item())
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| loss={value:.4f} "
            f"hier={hierarchy_loss.item():.4f} "
            f"direct={direct.item():.4f} "
            f"hpres={hierarchy_preservation.item():.5f} "
            f"rpres={representation_preservation.item():.5f}"
        )

        if value < best_loss - 1e-5:
            best_loss = value
            best_epoch = epoch
            bad_epochs = 0
            best_adapter = {
                k: v.detach().cpu().clone()
                for k, v in adapter.state_dict().items()
            }
            best_hierarchy = {
                k: v.detach().cpu().clone()
                for k, v in hierarchy_head.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    adapter.load_state_dict(best_adapter)
    hierarchy_head.load_state_dict(best_hierarchy)

    save_semantic_adapter_v061_checkpoint(
        args.output,
        adapter,
        heads,
        hierarchy_head,
        epoch=best_epoch,
        loss=best_loss,
        base_model=args.model,
        init_adapter=args.init_adapter,
        learning_rate=args.lr,
        hierarchy_weight=args.hierarchy_weight,
        direct_binding_weight=args.direct_binding_weight,
        direct_binding_margin=args.direct_binding_margin,
        hierarchy_preservation_weight=args.hierarchy_preservation_weight,
        representation_preservation_weight=args.representation_preservation_weight,
    )

    print()
    print("Semantic Encoder Adapter v0.6.1 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best loss        :", f"{best_loss:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
