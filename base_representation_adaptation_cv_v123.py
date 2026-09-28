# base_representation_adaptation_cv_v123.py
from __future__ import annotations

import argparse
import copy
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_router_dataset_v121 import DATASET, CLASSES
from fresh_generalization_cases_v114 import CASES as FRESH_V2
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}


class LateSemanticAdapter(nn.Module):
    """
    Trainable semantic adaptation path:
      cached output after block 4
        -> original block 5
        -> original block 6
        -> original FinalNorm
        -> projection 256->64
        -> 6-class router

    Blocks are copied from the clean base model for each fold/seed.
    """
    def __init__(self, base: LanguageModel, projection_dim: int = 64):
        super().__init__()
        if len(base.blocks) < 6:
            raise ValueError("v1.2.3 requires a 6-block base model.")
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
        h = self.final_norm(x)[:, -1, :]
        return h

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
        positions = torch.arange(x.shape[1], device=device)
        h = h + base.position_embedding(positions).unsqueeze(0)
    for block in base.blocks[:4]:
        h = block(h)
    return h[0].detach().cpu()


@torch.no_grad()
def original_final_hidden(base, ids):
    device = next(base.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return base.forward_hidden(x)[0, -1, :].detach().cpu()


def cache_dataset(base, tok, prompts):
    prefixes = []
    references = []
    lengths = []
    for p in prompts:
        ids = encode_ids(tok, base, p)
        prefixes.append(prefix_after_block4(base, ids))
        references.append(original_final_hidden(base, ids))
        lengths.append(len(ids))
    return prefixes, torch.stack(references), lengths


def family_folds():
    fams = defaultdict(list)
    for label,family,_ in DATASET:
        fams[label].append(family)
    order = {c:list(dict.fromkeys(fams[c])) for c in CLASSES}
    for c in CLASSES:
        assert len(order[c]) == 5
    folds = []
    for k in range(5):
        test = [i for i,(label,family,_) in enumerate(DATASET) if family == order[label][k]]
        assert len(test) == 60
        folds.append(test)
    return folds


def supervised_contrastive_loss(z, y, temp=0.1):
    z = F.normalize(z, dim=-1)
    sim = (z @ z.T) / temp
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    same = (y[:,None] == y[None,:]) & (~eye)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    count = same.sum(1).clamp_min(1)
    loss = -(log_prob * same).sum(1) / count
    valid = same.sum(1) > 0
    return loss[valid].mean()


def batch_groups(indices, lengths, batch_size, seed):
    # Exact-length grouping avoids attention over padding tokens.
    by_len = defaultdict(list)
    for i in indices:
        by_len[lengths[i]].append(i)
    rng = random.Random(seed)
    batches = []
    for _, idxs in by_len.items():
        rng.shuffle(idxs)
        for s in range(0, len(idxs), batch_size):
            batches.append(idxs[s:s+batch_size])
    rng.shuffle(batches)
    return batches


def make_prefix_batch(prefixes, idxs, device):
    return torch.stack([prefixes[i] for i in idxs]).to(device)


def train_fold(base, prefixes, references, lengths, y, train_idx, *,
               projection_dim, epochs, batch_size, router_lr, base_lr,
               weight_decay, contrastive_weight, preservation_weight,
               temperature, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = next(base.parameters()).device
    model = LateSemanticAdapter(base, projection_dim).to(device)

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

    for epoch in range(epochs):
        model.train()
        batches = batch_groups(train_idx, lengths, batch_size, seed + epoch)
        for idxs in batches:
            xb = make_prefix_batch(prefixes, idxs, device)
            yb = y[idxs]
            ref = references[idxs].to(device)

            opt.zero_grad()
            h,z,logits = model(xb)
            ce = ce_fn(logits, yb)
            sup = supervised_contrastive_loss(z, yb, temperature) if len(idxs) > 1 else ce.new_zeros(())
            preserve = F.mse_loss(h, ref)
            loss = ce + contrastive_weight * sup + preservation_weight * preserve
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    return model


@torch.no_grad()
def predict_indices(model, prefixes, lengths, indices, device, batch_size=32):
    probs = torch.zeros((len(indices), len(CLASSES)), dtype=torch.float32)
    pos = {idx:j for j,idx in enumerate(indices)}
    by_len = defaultdict(list)
    for i in indices:
        by_len[lengths[i]].append(i)
    for idxs in by_len.values():
        for s in range(0,len(idxs),batch_size):
            chunk = idxs[s:s+batch_size]
            xb = make_prefix_batch(prefixes, chunk, device)
            _,_,logits = model(xb)
            p = torch.softmax(logits, dim=-1).cpu()
            for row,idx in enumerate(chunk):
                probs[pos[idx]] = p[row]
    return probs


@torch.no_grad()
def hidden_drift(model, prefixes, references, lengths, indices, device):
    vals = []
    by_len = defaultdict(list)
    for i in indices:
        by_len[lengths[i]].append(i)
    for idxs in by_len.values():
        xb = make_prefix_batch(prefixes, idxs, device)
        h = model.semantic_hidden(xb)
        ref = references[idxs].to(device)
        cos = F.cosine_similarity(h, ref, dim=-1)
        vals.extend((1.0 - cos).detach().cpu().tolist())
    return sum(vals)/len(vals) if vals else 0.0


def metrics(y_true, probs):
    pred = probs.argmax(-1)
    cm = [[0]*len(CLASSES) for _ in CLASSES]
    for g,p in zip(y_true.tolist(),pred.tolist()):
        cm[g][p] += 1
    correct = sum(cm[i][i] for i in range(len(CLASSES)))
    per = {}
    for i,c in enumerate(CLASSES):
        total = sum(cm[i]); hit = cm[i][i]
        per[c] = (hit,total,hit/total if total else 0.0)
    macro = sum(v[2] for v in per.values()) / len(per)
    return correct/len(y_true), correct, macro, cm, per


def write_confusion(path, cm):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["gold"]+CLASSES)
        for i,c in enumerate(CLASSES):
            w.writerow([c]+cm[i])


def run_setting(base, prefixes, references, lengths, y, folds, *,
                projection_dim, epochs, batch_size, router_lr, base_lr,
                weight_decay, contrastive_weight, preservation_weight,
                temperature, seeds):
    oof = torch.zeros((len(DATASET),len(CLASSES)),dtype=torch.float32)
    drift_values = []
    device = next(base.parameters()).device

    for fold_idx,test_idx in enumerate(folds):
        test_set = set(test_idx)
        train_idx = [i for i in range(len(DATASET)) if i not in test_set]
        seed_probs = []
        seed_drifts = []
        for seed in seeds:
            model = train_fold(
                base,prefixes,references,lengths,y,train_idx,
                projection_dim=projection_dim,epochs=epochs,batch_size=batch_size,
                router_lr=router_lr,base_lr=base_lr,weight_decay=weight_decay,
                contrastive_weight=contrastive_weight,
                preservation_weight=preservation_weight,temperature=temperature,
                seed=seed+1000*fold_idx,
            )
            seed_probs.append(predict_indices(model,prefixes,lengths,test_idx,device))
            seed_drifts.append(hidden_drift(model,prefixes,references,lengths,test_idx,device))
        oof[test_idx] = torch.stack(seed_probs).mean(0)
        drift_values.append(sum(seed_drifts)/len(seed_drifts))

    acc,correct,macro,cm,per = metrics(y.cpu(),oof)
    return {
        "probs":oof,"accuracy":acc,"correct":correct,"macro":macro,
        "confusion":cm,"per":per,
        "drift":sum(drift_values)/len(drift_values),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    p.add_argument("--model",default=DEFAULT_MODEL)
    p.add_argument("--projection",type=int,default=64)
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--router-lr",type=float,default=8e-4)
    p.add_argument("--base-lr",type=float,default=1e-5)
    p.add_argument("--weight-decay",type=float,default=1e-3)
    p.add_argument("--contrastive-weight",type=float,default=0.10)
    p.add_argument("--preservation-weights",default="0.0,0.1,0.5,1.0")
    p.add_argument("--temperature",type=float,default=0.10)
    p.add_argument("--seeds",default="41,42")
    p.add_argument("--output-dir",default="results/base_representation_adaptation_v123")
    args=p.parse_args()

    preservation_weights=[float(x) for x in args.preservation_weights.split(",")]
    seeds=[int(x) for x in args.seeds.split(",")]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    base,_=LanguageModel.load_checkpoint(args.model,device=device)
    base.eval()
    for q in base.parameters():
        q.requires_grad_(False)

    prompts=[x[2] for x in DATASET]
    y=torch.tensor([CLASS_TO_ID[x[0]] for x in DATASET],dtype=torch.long,device=device)

    fresh=[(label,prompt) for label,prompt,_,_ in FRESH_V2 if label in CLASS_TO_ID]
    fresh_prompts=[p for _,p in fresh]
    fresh_y=torch.tensor([CLASS_TO_ID[l] for l,_ in fresh],dtype=torch.long)

    print("="*98)
    print(" LLM_GPU v1.2.3 Base Representation Adaptation")
    print("="*98)
    print("Device              :",device)
    print("Base                :",args.model)
    print("Trainable base      : Blocks 5-6 + FinalNorm")
    print("Frozen prefix       : Embedding + Blocks 1-4")
    print("Semantic projection :",f"{base.d_model} -> {args.projection}")
    print("Loss                : CE + 0.10*SupCon + alpha*representation-preservation")
    print("Base LR             :",args.base_lr)
    print("Router LR           :",args.router_lr)
    print("Preservation alpha  :",preservation_weights)
    print("CV                  : 5-fold family-held-out")
    print("Seeds/fold          :",seeds)
    print("Prefix cache        : enabled")
    print()

    print("Caching frozen prefix states...")
    prefixes,references,lengths=cache_dataset(base,tok,prompts)
    fresh_prefix,fresh_ref,fresh_lengths=cache_dataset(base,tok,fresh_prompts)
    folds=family_folds()

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    summary=[]
    best=None

    for alpha in preservation_weights:
        result=run_setting(
            base,prefixes,references,lengths,y,folds,
            projection_dim=args.projection,epochs=args.epochs,batch_size=args.batch_size,
            router_lr=args.router_lr,base_lr=args.base_lr,weight_decay=args.weight_decay,
            contrastive_weight=args.contrastive_weight,preservation_weight=alpha,
            temperature=args.temperature,seeds=seeds,
        )

        print(f"preservation={alpha:.2f}")
        print("-"*80)
        print(f"Family-held-out OOF : {result['correct']}/300 ({result['accuracy']:.1%}) macro={result['macro']:.1%}")
        print(f"Mean hidden drift    : {result['drift']:.6f}  (1 - cosine)")
        for c in CLASSES:
            h,n,r=result["per"][c]
            print(f"  {c:11s} {h:2d}/{n:2d} ({r:.1%})")
        print()

        tag=str(alpha).replace(".","p")
        write_confusion(outdir/f"preservation_{tag}_cv_confusion.csv",result["confusion"])
        summary.append([alpha,result["accuracy"],result["macro"],result["drift"]])
        key=(result["accuracy"],result["macro"],-result["drift"])
        if best is None or key>best[0]:
            best=(key,alpha)

    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["preservation_weight","family_cv_accuracy","macro_recall","mean_hidden_drift"])
        w.writerows(summary)

    print("Comparison")
    print("----------")
    for alpha,acc,macro,drift in summary:
        print(f"alpha={alpha:4.2f} family-CV={acc:.1%} macro={macro:.1%} drift={drift:.6f}")
    print()
    print(f"Best preservation weight: {best[1]:.2f}")
    print("Reference v1.2.2 best family-CV: 55.0% (frozen base, lambda=0.10)")
    print()
    print("Decision guide")
    print("--------------")
    print("If late-block adaptation materially exceeds 55% while drift remains small,")
    print("continue with adapted semantic representation. If accuracy does not improve,")
    print("the bottleneck likely lies earlier than Blocks 5-6 or in the base LM training.")
    print("Output dir:",outdir)


if __name__=="__main__":
    main()
