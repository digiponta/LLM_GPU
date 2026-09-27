# train_semantic_encoder_adapter_v06.py
#
# Semantic Encoder Adapter v0.6: Instruction Semantics
#
# CPU branch:
#   general-purpose / control-oriented / heterogeneous-instruction
#
# GPU branch:
#   throughput-oriented / data-parallel / homogeneous-computation
#
# Continues from v0.5. Base v0.8 encoder remains frozen.

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from evaluate_partial_intent_v09 import CASES
from model import LanguageModel
from semantic_encoder_adapter_v06 import (
    HIERARCHY_LABELS,
    initialize_from_v05,
    save_semantic_adapter_v06_checkpoint,
)
from tokenizer_bpe import Tokenizer
from train_semantic_encoder_adapter_v01 import AdapterDataset, stratified_split
from train_semantic_encoder_adapter_v03 import build_rows


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_INIT = "model/model-gpu-v0.9-semantic-adapter-v05.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.9.1-semantic-adapter-v06.pt"
SEED = 42


# Targets:
# processor, general_purpose, control_oriented,
# throughput_oriented, data_parallel,
# heterogeneous_instruction, homogeneous_computation
INSTRUCTION_ROWS: List[Tuple[str, Tuple[int, ...]]] = [
    # CPU: instructions include far more than arithmetic.
    ("CPUは算術計算だけでなく、分岐、比較、メモリアクセス、制御命令も実行します。", (1,1,1,0,0,1,0)),
    ("CPUが扱う命令には演算のほか、ロード、ストア、ジャンプ、関数呼び出しがあります。", (1,1,1,0,0,1,0)),
    ("多様な種類の命令を柔軟に実行するプロセッサはCPUです。", (1,1,1,0,0,1,0)),
    ("CPUは条件分岐や割り込み、I/O制御など計算以外の命令も処理します。", (1,1,1,0,0,1,0)),
    ("CPUは異なる種類の命令を順序立てて実行するgeneral-purpose processorです。", (1,1,1,0,0,1,0)),
    ("コンピュータの中心で制御、メモリ操作、分岐を含む多様な命令を処理するのはCPUです。", (1,1,1,0,0,1,0)),
    ("命令実行は計算だけではなく、制御フローやデータ移動も含み、CPUがそれらを担当します。", (1,1,1,0,0,1,0)),
    ("CPUの仕事は数値演算だけではなく、OS制御、分岐、メモリアクセスなど多岐にわたります。", (1,1,1,0,0,1,0)),
    ("比較、ジャンプ、ロード、ストアのような異種命令を扱うのがCPUです。", (1,1,1,0,0,1,0)),
    ("CPUはheterogeneous instructionsを扱い、逐次的な制御にも向いています。", (1,1,1,0,0,1,0)),

    # GPU: repeated/similar computation over many data items.
    ("GPUは同じ種類の演算を大量のデータへ並列適用するのが得意です。", (1,0,0,1,1,0,1)),
    ("GPUは多数の同種計算を同時実行するthroughput-oriented processorです。", (1,0,0,1,1,0,1)),
    ("GPUは複雑な分岐制御より、同型の演算を大量並列に処理する用途に向きます。", (1,0,0,1,1,0,1)),
    ("行列演算のような同種計算を大量に並列実行するのがGPUの強みです。", (1,0,0,1,1,0,1)),
    ("GPUはhomogeneous computationを高スループットで実行する設計です。", (1,0,0,1,1,0,1)),
    ("同じ処理を多数のデータ要素へ適用するdata-parallel processorはGPUです。", (1,0,0,1,1,0,1)),
    ("GPUの強みは多様な命令制御ではなく、大量の似た計算の並列実行です。", (1,0,0,1,1,0,1)),
    ("GPUは多数の類似演算を並列化することで高いスループットを得ます。", (1,0,0,1,1,0,1)),

    # Explicit contrast rows.
    ("CPUは異種命令を柔軟に処理し、GPUは同種演算を大量並列に処理します。", (1,1,1,1,1,1,1)),
    ("CPUの多様な命令には分岐やメモリ操作も含まれ、GPUの得意な大量並列演算とは異なります。", (1,1,1,1,1,1,1)),
]


class HierarchyDataset(Dataset):
    def __init__(self, rows, model, tokenizer):
        self.items = []
        for prompt, target in rows:
            hidden = encode_prompt_hidden(model, tokenizer, prompt).cpu()
            self.items.append((
                hidden,
                torch.tensor(target, dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Semantic Encoder Adapter v0.6 instruction semantics."
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--init-adapter", default=DEFAULT_INIT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--validation-ratio", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--hierarchy-weight", type=float, default=1.0)
    p.add_argument("--instruction-contrast-weight", type=float, default=1.0)
    p.add_argument("--preservation-weight", type=float, default=0.25)
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


def exact_overlap_with_dev(rows):
    dev = {str(case["prompt"]) for case in CASES}
    base_overlap = sorted({
        prompt for prompt, _concept, _attrs in rows if prompt in dev
    })
    instruction_overlap = sorted({
        prompt for prompt, _target in INSTRUCTION_ROWS if prompt in dev
    })
    return sorted(set(base_overlap + instruction_overlap))


def instruction_contrast_loss(logits, targets):
    probs = torch.sigmoid(logits)

    hetero = HIERARCHY_LABELS.index("heterogeneous_instruction")
    homo = HIERARCHY_LABELS.index("homogeneous_computation")
    general = HIERARCHY_LABELS.index("general_purpose")
    control = HIERARCHY_LABELS.index("control_oriented")
    throughput = HIERARCHY_LABELS.index("throughput_oriented")
    parallel = HIERARCHY_LABELS.index("data_parallel")

    cpu_mask = (
        (targets[:, hetero] > 0.5)
        & (targets[:, general] > 0.5)
        & (targets[:, throughput] < 0.5)
    )
    gpu_mask = (
        (targets[:, homo] > 0.5)
        & (targets[:, parallel] > 0.5)
        & (targets[:, general] < 0.5)
    )

    losses = []
    if cpu_mask.any():
        cpu_positive = (
            probs[cpu_mask, hetero]
            + probs[cpu_mask, general]
            + probs[cpu_mask, control]
        ) / 3.0
        cpu_negative = (
            probs[cpu_mask, homo]
            + probs[cpu_mask, throughput]
            + probs[cpu_mask, parallel]
        ) / 3.0
        losses.append(F.relu(0.25 + cpu_negative - cpu_positive).mean())

    if gpu_mask.any():
        gpu_positive = (
            probs[gpu_mask, homo]
            + probs[gpu_mask, throughput]
            + probs[gpu_mask, parallel]
        ) / 3.0
        gpu_negative = (
            probs[gpu_mask, hetero]
            + probs[gpu_mask, general]
            + probs[gpu_mask, control]
        ) / 3.0
        losses.append(F.relu(0.25 + gpu_negative - gpu_positive).mean())

    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def combined_loss(
    adapter,
    heads,
    hierarchy_head,
    base_hidden,
    hierarchy_hidden,
    hierarchy_targets,
    args,
):
    # Preserve the previously learned concept/attribute behavior by keeping
    # their heads active on the original supervision rows.
    adapted = adapter(base_hidden)
    concept_logits, attribute_logits = heads(adapted)

    # Pseudo-targets are produced from the frozen starting heads before
    # optimization and passed in via attached attributes.
    concept_target = combined_loss.concept_target
    attribute_target = combined_loss.attribute_target

    concept_loss = F.cross_entropy(concept_logits, concept_target)
    attribute_loss = F.binary_cross_entropy_with_logits(
        attribute_logits,
        attribute_target,
    )

    hierarchy_adapted = adapter(hierarchy_hidden)
    hierarchy_logits = hierarchy_head(hierarchy_adapted)
    hierarchy_loss = F.binary_cross_entropy_with_logits(
        hierarchy_logits,
        hierarchy_targets,
    )
    contrast = instruction_contrast_loss(
        hierarchy_logits,
        hierarchy_targets,
    )

    preservation = 1.0 - F.cosine_similarity(
        adapted,
        base_hidden,
        dim=-1,
    ).mean()

    total = (
        0.5 * concept_loss
        + 0.25 * attribute_loss
        + args.hierarchy_weight * hierarchy_loss
        + args.instruction_contrast_weight * contrast
        + args.preservation_weight * preservation
    )

    return total, hierarchy_loss, contrast, preservation


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model, args.init_adapter):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)

    base_model, base_checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    adapter, heads, hierarchy_head, init_checkpoint = initialize_from_v05(
        args.init_adapter,
        device,
    )

    rows = build_rows()
    overlaps = exact_overlap_with_dev(rows)
    if overlaps:
        raise RuntimeError(
            "Exact overlap with fixed development prompts: " + repr(overlaps)
        )

    # Original semantic rows provide a preservation bank.
    base_set = AdapterDataset(rows, base_model, tokenizer)
    base_hidden = torch.stack(
        [item[0] for item in base_set.items],
        dim=0,
    ).to(device)

    # Freeze pseudo-targets from v0.5 before any update.
    adapter.eval()
    heads.eval()
    with torch.no_grad():
        start_adapted = adapter(base_hidden)
        start_concept, start_attribute = heads(start_adapted)
        combined_loss.concept_target = start_concept.argmax(dim=-1)
        combined_loss.attribute_target = torch.sigmoid(start_attribute)

    hierarchy_set = HierarchyDataset(
        INSTRUCTION_ROWS,
        base_model,
        tokenizer,
    )
    hierarchy_hidden = torch.stack(
        [item[0] for item in hierarchy_set.items],
        dim=0,
    ).to(device)
    hierarchy_targets = torch.stack(
        [item[1] for item in hierarchy_set.items],
        dim=0,
    ).to(device)

    parameters = (
        list(adapter.parameters())
        + list(heads.parameters())
        + list(hierarchy_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=0.01,
    )

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.6 Training")
    print("====================================================")
    print("Device                    :", device)
    if device.type == "cuda":
        print("GPU                       :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss      :", base_checkpoint.get("loss"))
    print("Initial adapter           :", args.init_adapter)
    print("Initial adapter loss      :", init_checkpoint.get("loss"))
    print("Base encoder              : frozen")
    print("Hierarchy labels          :", ", ".join(HIERARCHY_LABELS))
    print("Instruction rows          :", len(INSTRUCTION_ROWS))
    print("Exact DEV overlap         :", len(overlaps))
    print("Learning rate             :", args.lr)
    print("Hierarchy weight          :", args.hierarchy_weight)
    print("Instruction contrast      :", args.instruction_contrast_weight)
    print("Preservation weight       :", args.preservation_weight)
    print()

    best_loss = float("inf")
    best_epoch = 0
    best_adapter = best_heads = best_hierarchy = None
    bad_epochs = 0

    # Full-batch hierarchy training is deliberate: the supervision bank is tiny.
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        heads.train()
        hierarchy_head.train()

        optimizer.zero_grad(set_to_none=True)
        total, hierarchy_loss, contrast, preservation = combined_loss(
            adapter,
            heads,
            hierarchy_head,
            base_hidden,
            hierarchy_hidden,
            hierarchy_targets,
            args,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        value = float(total.item())
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"| loss={value:.4f} "
            f"hier={hierarchy_loss.item():.4f} "
            f"contrast={contrast.item():.4f} "
            f"pres={preservation.item():.4f}"
        )

        if value < best_loss - 1e-5:
            best_loss = value
            best_epoch = epoch
            bad_epochs = 0
            best_adapter = {
                k: v.detach().cpu().clone()
                for k, v in adapter.state_dict().items()
            }
            best_heads = {
                k: v.detach().cpu().clone()
                for k, v in heads.state_dict().items()
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
    heads.load_state_dict(best_heads)
    hierarchy_head.load_state_dict(best_hierarchy)

    save_semantic_adapter_v06_checkpoint(
        args.output,
        adapter,
        heads,
        hierarchy_head,
        epoch=best_epoch,
        loss=best_loss,
        base_model=args.model,
        learning_rate=args.lr,
        hierarchy_weight=args.hierarchy_weight,
        instruction_contrast_weight=args.instruction_contrast_weight,
        preservation_weight=args.preservation_weight,
    )

    print()
    print("Semantic Encoder Adapter v0.6 training completed.")
    print("Best epoch       :", best_epoch)
    print("Best loss        :", f"{best_loss:.6f}")
    print("Checkpoint saved :", args.output)


if __name__ == "__main__":
    main()
