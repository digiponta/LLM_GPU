# train_selective_intent_repair_v01110.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from selective_intent_repair_v01110 import (
    TECH_LABELS,
    SelectiveIntentRepair,
    extract_technical_logits,
    save_checkpoint,
)
from evaluate_generalization_v07 import CASES
from intent_conditioning_v09 import load_intent_head
from model import LanguageModel
from tokenizer_bpe import Tokenizer

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-gpu-v0.8-chat-clean.pt"
DEFAULT_INTENT = "model/model-gpu-v0.8-intent-head-clean.pt"
DEFAULT_OUTPUT = "model/model-gpu-v0.11.10-selective-intent-repair.pt"

SEED = 42

POSITIVE_ROWS = [
    ("大量の同種計算を並列処理する装置は何ですか。", "tech_gpu"),
    ("機械学習で大量の並列計算を行う演算装置を答えてください。", "tech_gpu"),
    ("汎用命令の実行とシステム制御を担当する装置は何ですか。", "tech_cpu"),
    ("コンピュータの中心で幅広い命令を処理する演算装置は何ですか。", "tech_cpu"),
    ("大量の文章を学習して文章を生成する大規模言語モデルは何ですか。", "tech_llm"),
    ("自然言語を学習して文章を生成するモデルの種類は何ですか。", "tech_llm"),
    ("Self-Attentionを中心に使う代表的なモデル構造は何ですか。", "tech_transformer"),
    ("Attention機構を積み重ねるニューラルネット構造は何ですか。", "tech_transformer"),
    ("NVIDIA GPUを汎用計算に使う技術は何ですか。", "tech_cuda"),
    ("NVIDIA製GPU向けの並列計算プラットフォームは何ですか。", "tech_cuda"),
    ("読みやすい文法で知られる汎用プログラミング言語は何ですか。", "tech_python"),
    ("機械学習やデータ分析でもよく使う高水準言語は何ですか。", "tech_python"),
]

NEGATIVE_ROWS = [
    "二つの実験結果を公平に比較する方法を教えてください。",
    "モデルAとモデルBを同じ条件で比べたいです。",
    "今の説明を別の言い方でお願いします。",
    "この話題はここで終わりにします。",
    "別のテーマへ移りましょう。",
    "コード実行時のエラー原因を調べたいです。",
    "エラーメッセージを見て原因を切り分けたいです。",
    "こんにちは。今日は調子がいいです。",
    "助かりました。ありがとうございます。",
    "少し疲れたので休憩したいです。",
    "日本の首都を教えてください。",
    "回答を短くしてください。",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--intent-head", default=DEFAULT_INTENT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=260)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--scope-weight", type=float, default=0.75)
    p.add_argument("--repair-weight", type=float, default=1.0)
    p.add_argument("--residual-weight", type=float, default=0.02)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)

    for filename in (args.tokenizer, args.model, args.intent_head):
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    benchmark = {str(case["prompt"]) for case in CASES}
    overlap = [p for p, _ in POSITIVE_ROWS if p in benchmark]
    overlap += [p for p in NEGATIVE_ROWS if p in benchmark]
    if overlap:
        raise RuntimeError("Exact DEV overlap: " + repr(overlap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    intent_head, intent_ck, intent_labels = load_intent_head(args.intent_head, model, device)

    for module in (model, intent_head):
        for p in module.parameters():
            p.requires_grad_(False)

    xs = []
    class_targets = []
    scope_targets = []
    row_types = []

    with torch.no_grad():
        for prompt, label in POSITIVE_ROWS:
            ids = tokenizer.encode(f"人: {prompt}\nAI: ", add_bos=True)[-model.context_length:]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            h = model.forward_hidden(x)[:, -1, :]
            tech = extract_technical_logits(intent_head(h), intent_labels)[0]
            xs.append(tech)
            class_targets.append(TECH_LABELS.index(label))
            scope_targets.append(1.0)
            row_types.append("pos")

        for prompt in NEGATIVE_ROWS:
            ids = tokenizer.encode(f"人: {prompt}\nAI: ", add_bos=True)[-model.context_length:]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            h = model.forward_hidden(x)[:, -1, :]
            tech = extract_technical_logits(intent_head(h), intent_labels)[0]
            xs.append(tech)
            class_targets.append(-1)
            scope_targets.append(0.0)
            row_types.append("neg")

    x_all = torch.stack(xs)
    scope_all = torch.tensor(scope_targets, dtype=torch.float32, device=device)
    class_all = torch.tensor(class_targets, dtype=torch.long, device=device)

    # Hold out one positive per class and three negatives.
    val_list = []
    for label in TECH_LABELS:
        idxs = [i for i, (p, l) in enumerate(POSITIVE_ROWS) if l == label]
        val_list.append(idxs[-1])
    neg_offset = len(POSITIVE_ROWS)
    val_list.extend([neg_offset + 1, neg_offset + 5, neg_offset + 9])

    val_set = set(val_list)
    train_list = [i for i in range(len(row_types)) if i not in val_set]
    train_idx = torch.tensor(train_list, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_list, dtype=torch.long, device=device)

    repair = SelectiveIntentRepair(len(TECH_LABELS), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(repair.parameters(), lr=args.lr, weight_decay=0.01)

    def compute(indices):
        inp = x_all[indices]
        repaired, scope_logit = repair(inp)
        scope_target = scope_all[indices]
        scope_loss = F.binary_cross_entropy_with_logits(scope_logit, scope_target)

        pos_mask = class_all[indices] >= 0
        if bool(pos_mask.any()):
            ce = F.cross_entropy(repaired[pos_mask], class_all[indices][pos_mask])
            raw_top = inp[pos_mask].argmax(dim=-1)
            repaired_top = repaired[pos_mask].argmax(dim=-1)
            cls_acc = (repaired_top == class_all[indices][pos_mask]).float().mean()
            preserve = (raw_top == class_all[indices][pos_mask]).float()
            # Penalize large corrections on already-correct raw classifications.
            residual = repair.repair(inp[pos_mask]).pow(2).mean()
        else:
            ce = torch.zeros((), device=device)
            cls_acc = torch.zeros((), device=device)
            preserve = torch.zeros((), device=device)
            residual = repair.repair(inp).pow(2).mean()

        total = (
            args.repair_weight * ce
            + args.scope_weight * scope_loss
            + args.residual_weight * residual
        )
        scope_prob = torch.sigmoid(scope_logit)
        scope_acc = ((scope_prob >= 0.5).float() == scope_target).float().mean()
        return total, ce, scope_loss, residual, cls_acc, scope_acc, scope_prob

    print("====================================================")
    print(" Selective Intent Repair v0.11.10")
    print("====================================================")
    print("Device:", device)
    print("Positive rows:", len(POSITIVE_ROWS), "Negative rows:", len(NEGATIVE_ROWS))
    print("Train:", len(train_idx), "Val:", len(val_idx))
    print("Exact DEV overlap:", len(overlap))

    best = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        repair.train()
        optimizer.zero_grad(set_to_none=True)
        loss, ce, scope_loss, residual, cls_acc, scope_acc, scope_prob = compute(train_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(repair.parameters(), 1.0)
        optimizer.step()

        repair.eval()
        with torch.no_grad():
            val, vce, vscope, vresidual, vcls, vscope_acc, vprob = compute(val_idx)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"| train={loss.item():.4f} cls={cls_acc.item():.3f} scope={scope_acc.item():.3f} "
                f"| val={val.item():.4f} cls={vcls.item():.3f} scope={vscope_acc.item():.3f} "
                f"| val_scope={vprob.mean().item():.3f}"
            )

        value = float(val.item())
        if value < best - 1e-6:
            best = value
            best_epoch = epoch
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in repair.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    repair.load_state_dict(best_state)

    save_checkpoint(
        args.output,
        repair,
        intent_labels,
        epoch=best_epoch,
        loss=best,
        base_model=args.model,
        intent_head=args.intent_head,
        scope_threshold=0.70,
        repair_margin_threshold=0.20,
        repair_confidence_threshold=0.55,
        binding_confidence_threshold=0.70,
    )

    print("Best epoch:", best_epoch)
    print("Best val loss:", f"{best:.6f}")
    print("Checkpoint:", args.output)


if __name__ == "__main__":
    main()
