# late_block_adaptation_diagnostic_v124.py
from __future__ import annotations

import argparse
import copy
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_router_dataset_v121 import DATASET, CLASSES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}


class LateSemanticAdapter(nn.Module):
    def __init__(self, base: LanguageModel, projection_dim: int = 64):
        super().__init__()
        if len(base.blocks) < 6:
            raise ValueError("v1.2.4 requires a 6-block base model.")
        self.block5 = copy.deepcopy(base.blocks[4])
        self.block6 = copy.deepcopy(base.blocks[5])
        self.final_norm = copy.deepcopy(base.final_norm)
        self.projection = nn.Sequential(
            nn.Linear(base.d_model, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
        )
        self.router = nn.Linear(projection_dim, len(CLASSES))

    def semantic_hidden(self, prefix_states):
        x = self.block5(prefix_states)
        x = self.block6(x)
        return self.final_norm(x)[:, -1, :]

    def forward(self, prefix_states):
        h = self.semantic_hidden(prefix_states)
        z = self.projection(h)
        logits = self.router(z)
        return h, z, logits


def encode_ids(tok, model, prompt):
    ids = tok.encode(f"人: {prompt}\nAI: ", add_bos=True)
    return ids[-model.context_length:]


@torch.no_grad()
def prefix_after_block4(base, ids):
    device = next(base.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    h = base.embedding(x)
    if base.position_embedding is not None:
        pos = torch.arange(x.shape[1], device=device)
        h = h + base.position_embedding(pos).unsqueeze(0)
    for block in base.blocks[:4]:
        h = block(h)
    return h[0].detach().cpu()


@torch.no_grad()
def original_final_hidden(base, ids):
    device = next(base.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return base.forward_hidden(x)[0, -1, :].detach().cpu()


def cache_dataset(base, tok, prompts):
    prefixes, refs, lengths = [], [], []
    for p in prompts:
        ids = encode_ids(tok, base, p)
        prefixes.append(prefix_after_block4(base, ids))
        refs.append(original_final_hidden(base, ids))
        lengths.append(len(ids))
    return prefixes, torch.stack(refs), lengths


def family_folds():
    fams = defaultdict(list)
    for label, family, _ in DATASET:
        fams[label].append(family)
    order = {c:list(dict.fromkeys(fams[c])) for c in CLASSES}
    folds = []
    for k in range(5):
        test = [i for i,(label,family,_) in enumerate(DATASET) if family == order[label][k]]
        folds.append(test)
    return folds


def supcon_loss(z, y, temp=0.1):
    z = F.normalize(z, dim=-1)
    sim = (z @ z.T) / temp
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    same = (y[:,None] == y[None,:]) & (~eye)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    cnt = same.sum(1).clamp_min(1)
    valid = same.sum(1) > 0
    loss = -(logp * same).sum(1) / cnt
    return loss[valid].mean()


def batch_groups(indices, lengths, batch_size, seed):
    by_len = defaultdict(list)
    for i in indices:
        by_len[lengths[i]].append(i)
    rng = random.Random(seed)
    batches = []
    for idxs in by_len.values():
        idxs = list(idxs)
        rng.shuffle(idxs)
        for s in range(0, len(idxs), batch_size):
            batches.append(idxs[s:s+batch_size])
    rng.shuffle(batches)
    return batches


def make_batch(prefixes, idxs, device):
    return torch.stack([prefixes[i] for i in idxs]).to(device)


def snapshot_named(module):
    return {k:v.detach().cpu().clone() for k,v in module.named_parameters()}


def relative_delta(before, module):
    num = 0.0
    den = 0.0
    max_abs = 0.0
    for k,v in module.named_parameters():
        a = before[k]
        b = v.detach().cpu()
        d = b - a
        num += float((d*d).sum().item())
        den += float((a*a).sum().item())
        max_abs = max(max_abs, float(d.abs().max().item()))
    rel = math.sqrt(num) / max(math.sqrt(den), 1e-12)
    return rel, max_abs


def module_grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float((p.grad.detach()**2).sum().item())
    return math.sqrt(total)


def train_fold(base, prefixes, refs, lengths, y, train_idx, *,
               projection_dim, epochs, batch_size, router_lr, base_lr,
               weight_decay, contrastive_weight, preservation_weight,
               temperature, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = next(base.parameters()).device
    model = LateSemanticAdapter(base, projection_dim).to(device)

    before_b5 = snapshot_named(model.block5)
    before_b6 = snapshot_named(model.block6)
    before_fn = snapshot_named(model.final_norm)

    late_params = (
        list(model.block5.parameters()) +
        list(model.block6.parameters()) +
        list(model.final_norm.parameters())
    )
    semantic_params = list(model.projection.parameters()) + list(model.router.parameters())

    opt = torch.optim.AdamW([
        {"params": late_params, "lr": base_lr},
        {"params": semantic_params, "lr": router_lr},
    ], weight_decay=weight_decay)

    ce_fn = nn.CrossEntropyLoss()

    grad_acc = {"block5":0.0,"block6":0.0,"final_norm":0.0}
    grad_steps = 0

    for epoch in range(epochs):
        model.train()
        for idxs in batch_groups(train_idx, lengths, batch_size, seed + epoch):
            xb = make_batch(prefixes, idxs, device)
            yb = y[idxs]
            ref = refs[idxs].to(device)

            opt.zero_grad()
            h,z,logits = model(xb)
            ce = ce_fn(logits, yb)
            sup = supcon_loss(z, yb, temperature) if len(idxs) > 1 else ce.new_zeros(())
            preserve = F.mse_loss(h, ref)
            loss = ce + contrastive_weight * sup + preservation_weight * preserve
            loss.backward()

            grad_acc["block5"] += module_grad_norm(model.block5)
            grad_acc["block6"] += module_grad_norm(model.block6)
            grad_acc["final_norm"] += module_grad_norm(model.final_norm)
            grad_steps += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    b5_rel,b5_max = relative_delta(before_b5, model.block5)
    b6_rel,b6_max = relative_delta(before_b6, model.block6)
    fn_rel,fn_max = relative_delta(before_fn, model.final_norm)

    for k in grad_acc:
        grad_acc[k] /= max(grad_steps,1)

    model.eval()
    return model, {
        "block5_rel_delta": b5_rel,
        "block6_rel_delta": b6_rel,
        "final_norm_rel_delta": fn_rel,
        "block5_max_abs_delta": b5_max,
        "block6_max_abs_delta": b6_max,
        "final_norm_max_abs_delta": fn_max,
        "block5_grad_norm": grad_acc["block5"],
        "block6_grad_norm": grad_acc["block6"],
        "final_norm_grad_norm": grad_acc["final_norm"],
    }


@torch.no_grad()
def predict(model, prefixes, lengths, indices, device):
    probs = torch.zeros((len(indices),len(CLASSES)),dtype=torch.float32)
    pos = {idx:j for j,idx in enumerate(indices)}
    by_len = defaultdict(list)
    for i in indices:
        by_len[lengths[i]].append(i)
    for idxs in by_len.values():
        xb = make_batch(prefixes, idxs, device)
        _,_,logits = model(xb)
        p = torch.softmax(logits,dim=-1).cpu()
        for row,idx in enumerate(idxs):
            probs[pos[idx]] = p[row]
    return probs


@torch.no_grad()
def hidden_drift(model, prefixes, refs, lengths, indices, device):
    vals = []
    by_len = defaultdict(list)
    for i in indices:
        by_len[lengths[i]].append(i)
    for idxs in by_len.values():
        xb = make_batch(prefixes, idxs, device)
        h = model.semantic_hidden(xb)
        ref = refs[idxs].to(device)
        vals.extend((1.0 - F.cosine_similarity(h,ref,dim=-1)).cpu().tolist())
    return sum(vals)/len(vals)


def metrics(y_true, probs):
    pred = probs.argmax(-1)
    cm = [[0]*len(CLASSES) for _ in CLASSES]
    for g,p in zip(y_true.tolist(),pred.tolist()):
        cm[g][p] += 1
    correct = sum(cm[i][i] for i in range(len(CLASSES)))
    per={}
    for i,c in enumerate(CLASSES):
        total=sum(cm[i]); hit=cm[i][i]
        per[c]=(hit,total,hit/total if total else 0.0)
    macro=sum(v[2] for v in per.values())/len(per)
    return correct/len(y_true),correct,macro,cm,per


def write_confusion(path,cm):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["gold"]+CLASSES)
        for i,c in enumerate(CLASSES):
            w.writerow([c]+cm[i])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--projection",type=int,default=64)
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--router-lr",type=float,default=8e-4)
    p.add_argument("--base-lrs",default="1e-5,3e-5,1e-4,3e-4")
    p.add_argument("--weight-decay",type=float,default=1e-3)
    p.add_argument("--contrastive-weight",type=float,default=0.10)
    p.add_argument("--preservation-weight",type=float,default=0.10)
    p.add_argument("--temperature",type=float,default=0.10)
    p.add_argument("--seeds",default="41,42")
    p.add_argument("--output-dir",default="results/late_block_adaptation_diagnostic_v124")
    args=p.parse_args()

    base_lrs=[float(x) for x in args.base_lrs.split(",")]
    seeds=[int(x) for x in args.seeds.split(",")]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    base,_=LanguageModel.load_checkpoint(args.model,device=device)
    base.eval()
    for q in base.parameters():
        q.requires_grad_(False)

    prompts=[x[2] for x in DATASET]
    y=torch.tensor([CLASS_TO_ID[x[0]] for x in DATASET],dtype=torch.long,device=device)

    print("="*100)
    print(" LLM_GPU v1.2.4 Late-Block Adaptation Diagnostic + LR Sweep")
    print("="*100)
    print("Device              :",device)
    print("Base                :",args.model)
    print("Trainable base      : Blocks 5-6 + FinalNorm")
    print("Frozen prefix       : Embedding + Blocks 1-4")
    print("Base LR sweep       :",base_lrs)
    print("Router LR           :",args.router_lr)
    print("Loss                : CE + 0.10*SupCon + 0.10*preservation")
    print("CV                  : 5-fold family-held-out")
    print("Seeds/fold          :",seeds)
    print()

    print("Caching frozen prefix states...")
    prefixes,refs,lengths=cache_dataset(base,tok,prompts)
    folds=family_folds()
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)

    summary=[]
    best=None

    for base_lr in base_lrs:
        oof=torch.zeros((len(DATASET),len(CLASSES)),dtype=torch.float32)
        diag_rows=[]
        drift_vals=[]

        for fold_idx,test_idx in enumerate(folds):
            test_set=set(test_idx)
            train_idx=[i for i in range(len(DATASET)) if i not in test_set]
            seed_probs=[]

            for seed in seeds:
                model,diag=train_fold(
                    base,prefixes,refs,lengths,y,train_idx,
                    projection_dim=args.projection,epochs=args.epochs,batch_size=args.batch_size,
                    router_lr=args.router_lr,base_lr=base_lr,weight_decay=args.weight_decay,
                    contrastive_weight=args.contrastive_weight,
                    preservation_weight=args.preservation_weight,temperature=args.temperature,
                    seed=seed+1000*fold_idx,
                )
                seed_probs.append(predict(model,prefixes,lengths,test_idx,device))
                drift=hidden_drift(model,prefixes,refs,lengths,test_idx,device)
                drift_vals.append(drift)
                diag_rows.append({"fold":fold_idx+1,"seed":seed,"base_lr":base_lr,"hidden_drift":drift,**diag})

            oof[test_idx]=torch.stack(seed_probs).mean(0)

        acc,correct,macro,cm,per=metrics(y.cpu(),oof)
        mean_diag={k:sum(float(r[k]) for r in diag_rows)/len(diag_rows) for k in diag_rows[0] if k not in ("fold","seed","base_lr")}
        mean_drift=sum(drift_vals)/len(drift_vals)

        print(f"base_lr={base_lr:.1e}")
        print("-"*82)
        print(f"Family-held-out OOF : {correct}/300 ({acc:.1%}) macro={macro:.1%}")
        print(f"Hidden drift        : {mean_drift:.12f}")
        print(f"Block5 rel delta    : {mean_diag['block5_rel_delta']:.12e}")
        print(f"Block6 rel delta    : {mean_diag['block6_rel_delta']:.12e}")
        print(f"FinalNorm rel delta : {mean_diag['final_norm_rel_delta']:.12e}")
        print(f"Block5 grad norm    : {mean_diag['block5_grad_norm']:.12e}")
        print(f"Block6 grad norm    : {mean_diag['block6_grad_norm']:.12e}")
        print(f"FinalNorm grad norm : {mean_diag['final_norm_grad_norm']:.12e}")
        for c in CLASSES:
            h,n,r=per[c]
            print(f"  {c:11s} {h:2d}/{n:2d} ({r:.1%})")
        print()

        tag=f"{base_lr:.0e}".replace("-","m").replace("+","p")
        write_confusion(outdir/f"lr_{tag}_cv_confusion.csv",cm)
        with (outdir/f"lr_{tag}_diagnostics.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(diag_rows[0].keys())); w.writeheader(); w.writerows(diag_rows)

        summary.append([
            base_lr,acc,macro,mean_drift,
            mean_diag["block5_rel_delta"],mean_diag["block6_rel_delta"],mean_diag["final_norm_rel_delta"],
            mean_diag["block5_grad_norm"],mean_diag["block6_grad_norm"],mean_diag["final_norm_grad_norm"],
        ])
        key=(acc,macro,-mean_drift)
        if best is None or key>best[0]:
            best=(key,base_lr)

    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow([
            "base_lr","family_cv_accuracy","macro_recall","mean_hidden_drift",
            "block5_rel_delta","block6_rel_delta","final_norm_rel_delta",
            "block5_grad_norm","block6_grad_norm","final_norm_grad_norm",
        ])
        w.writerows(summary)

    print("Comparison")
    print("----------")
    for row in summary:
        print(
            f"lr={row[0]:.1e} acc={row[1]:.1%} macro={row[2]:.1%} "
            f"drift={row[3]:.12f} b5d={row[4]:.3e} b6d={row[5]:.3e} fnd={row[6]:.3e}"
        )
    print()
    print(f"Best base LR by family-CV: {best[1]:.1e}")
    print("Reference v1.2.2 best family-CV: 55.0%")
    print("Reference v1.2.3 base LR 1e-5: 53.0%")
    print()
    print("Interpretation:")
    print("1) Non-zero gradients + non-zero parameter deltas confirm adaptation is active.")
    print("2) If delta grows with LR but accuracy stays flat, late blocks are not enough.")
    print("3) If accuracy rises above 55%, continue with late-block adaptation.")
    print("Output dir:",outdir)


if __name__=="__main__":
    main()
