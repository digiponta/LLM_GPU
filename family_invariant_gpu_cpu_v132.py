# family_invariant_gpu_cpu_v132.py
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_router_dataset_v121 import DATASET, CLASSES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer

GPU, CPU = "gpu", "cpu"
CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}


class ResidualSemanticAdapter(nn.Module):
    def __init__(self, d_model=256, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # Start exactly at the frozen Block1 representation.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.mlp(x))


class BinaryHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        return self.fc(x)


class LinearProbe(nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.fc = nn.Linear(d_in, n_out)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def block1_vector(model, tok, prompt):
    device = next(model.parameters()).device
    ids = tok.encode(f"人: {prompt}\nAI: ", add_bos=True)
    ids = ids[-model.context_length:]
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)

    x = model.embedding(token_ids)
    if model.position_embedding is not None:
        pos = torch.arange(token_ids.shape[1], device=device)
        x = x + model.position_embedding(pos).unsqueeze(0)

    x = model.blocks[0](x)
    return x[0, -1, :].detach()


def family_order():
    fams = defaultdict(list)
    for label, fam, _ in DATASET:
        fams[label].append(fam)
    order = {c:list(dict.fromkeys(fams[c])) for c in CLASSES}
    return order


def idx_for(label=None, family=None):
    out = []
    for i,(l,f,_) in enumerate(DATASET):
        if label is not None and l != label:
            continue
        if family is not None and f != family:
            continue
        out.append(i)
    return out


def standardize(Xtr, Xte):
    mean = Xtr.mean(0, keepdim=True)
    std = Xtr.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (Xtr-mean)/std, (Xte-mean)/std


def family_invariance_loss(z, labels, families):
    losses = []
    for cls in (0,1):
        mask = labels == cls
        class_centroid = z[mask].mean(0)
        fam_names = sorted(set(families[i] for i in range(len(families)) if bool(mask[i])))
        for fam in fam_names:
            fmask = torch.tensor(
                [bool(mask[i]) and families[i] == fam for i in range(len(families))],
                dtype=torch.bool, device=z.device,
            )
            fam_centroid = z[fmask].mean(0)
            losses.append(F.mse_loss(fam_centroid, class_centroid))
    return torch.stack(losses).mean() if losses else z.new_zeros(())


def compactness_loss(z, labels):
    losses=[]
    for cls in (0,1):
        mask=labels==cls
        centroid=z[mask].mean(0,keepdim=True)
        losses.append(((z[mask]-centroid)**2).mean())
    return torch.stack(losses).mean()


def separation_loss(z, labels, margin):
    c0=F.normalize(z[labels==0].mean(0),dim=0)
    c1=F.normalize(z[labels==1].mean(0),dim=0)
    dist=torch.norm(c0-c1,p=2)
    return F.relu(torch.tensor(margin,device=z.device)-dist).pow(2)


def train_family_invariant(
    X, train_pair_idx, preserve_idx, families, *,
    hidden, epochs, lr, wd, w_compact, w_separation,
    w_family, w_preserve, margin, seed,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device=X.device
    adapter=ResidualSemanticAdapter(X.shape[1],hidden).to(device)
    head=BinaryHead(X.shape[1]).to(device)
    opt=torch.optim.AdamW(
        list(adapter.parameters())+list(head.parameters()),
        lr=lr, weight_decay=wd,
    )
    ce_fn=nn.CrossEntropyLoss()

    pair_x=X[train_pair_idx]
    labels=torch.tensor(
        [0 if DATASET[i][0]==GPU else 1 for i in train_pair_idx],
        dtype=torch.long, device=device,
    )
    pair_families=[DATASET[i][1] for i in train_pair_idx]
    preserve_x=X[preserve_idx]

    for _ in range(epochs):
        adapter.train(); head.train(); opt.zero_grad()

        z=adapter(pair_x)
        logits=head(z)
        ce=ce_fn(logits,labels)
        compact=compactness_loss(z,labels)
        sep=separation_loss(z,labels,margin)
        fam=family_invariance_loss(z,labels,pair_families)

        zp=adapter(preserve_x)
        preserve=F.mse_loss(zp,preserve_x)

        loss=(
            ce
            + w_compact*compact
            + w_separation*sep
            + w_family*fam
            + w_preserve*preserve
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters())+list(head.parameters()),1.0)
        opt.step()

    adapter.eval(); head.eval()
    return adapter,head


def train_probe(X,y,epochs,lr,wd,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    p=LinearProbe(X.shape[1],len(CLASSES)).to(X.device)
    opt=torch.optim.AdamW(p.parameters(),lr=lr,weight_decay=wd)
    loss_fn=nn.CrossEntropyLoss()
    for _ in range(epochs):
        p.train(); opt.zero_grad()
        loss=loss_fn(p(X),y); loss.backward(); opt.step()
    p.eval(); return p


def metrics(y,pred):
    cm=[[0]*len(CLASSES) for _ in CLASSES]
    for g,p in zip(y.tolist(),pred.tolist()):
        cm[g][p]+=1
    correct=sum(cm[i][i] for i in range(len(CLASSES)))
    per={}
    for i,c in enumerate(CLASSES):
        n=sum(cm[i]); h=cm[i][i]
        per[c]=(h,n,h/n if n else 0.0)
    return correct/len(y),correct,cm,per


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=500)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--hidden",type=int,default=128)
    ap.add_argument("--compact-weight",type=float,default=0.10)
    ap.add_argument("--separation-weight",type=float,default=0.50)
    ap.add_argument("--family-weight",type=float,default=1.00)
    ap.add_argument("--preservation-weight",type=float,default=0.25)
    ap.add_argument("--margin",type=float,default=1.0)
    ap.add_argument("--probe-epochs",type=int,default=250)
    ap.add_argument("--probe-lr",type=float,default=1e-3)
    ap.add_argument("--seeds",default="41,42")
    ap.add_argument("--output-dir",default="results/family_invariant_gpu_cpu_v132")
    args=ap.parse_args()

    seeds=[int(s) for s in args.seeds.split(",") if s.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok=Tokenizer.load(args.tokenizer)
    base,_=LanguageModel.load_checkpoint(args.model,device=device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    print("="*106)
    print(" LLM_GPU v1.3.2 Family-Invariant GPU-CPU Representation Learning")
    print("="*106)
    print("Device              :",device)
    print("Base                :",args.model)
    print("Source representation: frozen Block1")
    print("Adapter             : residual 256 -> 128 -> 256")
    print("Loss                : CE + compactness + centroid separation + family invariance + preservation")
    print("Weights             :",f"compact={args.compact_weight} separation={args.separation_weight} family={args.family_weight} preservation={args.preservation_weight}")
    print("Primary metric      : 25-pair leave-target-pair-out GPU/CPU generalization")
    print("Secondary metric    : 5-fold 6-class family-held-out routing")
    print("Reference           : v1.3.1 leave-pair-out mean=39.0%, v1.2.7 Block1 6-class=60.7%")
    print()

    print("Extracting frozen Block1 vectors...")
    X=torch.stack([block1_vector(base,tok,prompt) for _,_,prompt in DATASET]).to(device)
    order=family_order()
    gpu_fams=order[GPU]; cpu_fams=order[CPU]
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)

    # A. 25 target family-pair leave-out matrix.
    matrix={}
    details=[]
    for gfam in gpu_fams:
        line=[]
        for cfam in cpu_fams:
            test=idx_for(GPU,gfam)+idx_for(CPU,cfam)
            train_pair=[
                i for i,(l,f,_) in enumerate(DATASET)
                if l in (GPU,CPU)
                and not (l==GPU and f==gfam)
                and not (l==CPU and f==cfam)
            ]
            preserve=[
                i for i,(l,f,_) in enumerate(DATASET)
                if not ((l==GPU and f==gfam) or (l==CPU and f==cfam))
            ]

            probs=[]
            for seed in seeds:
                adapter,head=train_family_invariant(
                    X,train_pair,preserve,[x[1] for x in DATASET],
                    hidden=args.hidden,epochs=args.epochs,lr=args.lr,wd=args.weight_decay,
                    w_compact=args.compact_weight,w_separation=args.separation_weight,
                    w_family=args.family_weight,w_preserve=args.preservation_weight,
                    margin=args.margin,seed=seed,
                )
                with torch.no_grad():
                    probs.append(torch.softmax(head(adapter(X[test])),dim=-1).cpu())

            pred=torch.stack(probs).mean(0).argmax(-1)
            gold=torch.tensor([0 if DATASET[i][0]==GPU else 1 for i in test])
            acc=float((pred==gold).float().mean().item())
            gpu_rec=float((pred[:10]==gold[:10]).float().mean().item())
            cpu_rec=float((pred[10:]==gold[10:]).float().mean().item())
            matrix[(gfam,cfam)]=acc
            details.append([gfam,cfam,acc,gpu_rec,cpu_rec])
            line.append(f"{acc:.0%}")
        print(f"{gfam:14s}: "+" ".join(f"{v:>5s}" for v in line))

    vals=list(matrix.values())
    print()
    print(f"Leave-pair-out mean={sum(vals)/len(vals):.1%} min={min(vals):.1%} max={max(vals):.1%}")

    with (outdir/"leave_pair_out_accuracy.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["gpu_family\\cpu_family"]+cpu_fams)
        for g in gpu_fams:
            w.writerow([g]+[matrix[(g,c)] for c in cpu_fams])

    with (outdir/"leave_pair_out_details.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["gpu_family","cpu_family","accuracy","gpu_recall","cpu_recall"]); w.writerows(details)

    # B. Standard 5-fold family-held-out; adapt using seen GPU/CPU families,
    # then train a new 6-class linear probe on adapted representations.
    six_oof=torch.zeros((len(DATASET),len(CLASSES)),dtype=torch.float32)
    y_all=torch.tensor([CLASS_TO_ID[l] for l,_,_ in DATASET],dtype=torch.long)

    for k in range(5):
        test=[i for i,(l,f,_) in enumerate(DATASET) if f==order[l][k]]
        testset=set(test)
        train=[i for i in range(len(DATASET)) if i not in testset]
        train_pair=[i for i in train if DATASET[i][0] in (GPU,CPU)]

        seed_probs=[]
        for seed in seeds:
            adapter,_=train_family_invariant(
                X,train_pair,train,[x[1] for x in DATASET],
                hidden=args.hidden,epochs=args.epochs,lr=args.lr,wd=args.weight_decay,
                w_compact=args.compact_weight,w_separation=args.separation_weight,
                w_family=args.family_weight,w_preserve=args.preservation_weight,
                margin=args.margin,seed=seed+1000*k,
            )
            with torch.no_grad():
                Xtr=adapter(X[train]); Xte=adapter(X[test])
            Xtr,Xte=standardize(Xtr,Xte)
            probe=train_probe(
                Xtr,y_all[train].to(device),args.probe_epochs,args.probe_lr,args.weight_decay,
                seed+5000*k,
            )
            with torch.no_grad():
                seed_probs.append(torch.softmax(probe(Xte),dim=-1).cpu())

        six_oof[test]=torch.stack(seed_probs).mean(0)

    pred=six_oof.argmax(-1)
    acc,correct,cm,per=metrics(y_all,pred)
    print()
    print("6-class family-held-out after family-invariant adaptation")
    print("---------------------------------------------------------")
    print(f"{correct}/300 ({acc:.1%})")
    for c in CLASSES:
        h,n,r=per[c]
        print(f"  {c:11s} {h:2d}/{n:2d} ({r:.1%})")

    with (outdir/"six_class_confusion.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["gold"]+CLASSES)
        for i,c in enumerate(CLASSES):
            w.writerow([c]+cm[i])

    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["leave_pair_out_mean",sum(vals)/len(vals)])
        w.writerow(["leave_pair_out_min",min(vals)])
        w.writerow(["leave_pair_out_max",max(vals)])
        w.writerow(["six_class_accuracy",acc])
        for c in CLASSES:
            w.writerow([f"six_class_recall_{c}",per[c][2]])

    print()
    print("Decision")
    print("--------")
    print("Primary target: leave-pair-out 39.0% -> >=60%.")
    print("Secondary guardrail: keep 6-class accuracy near the 60.7% Block1 reference.")
    print("If leave-pair-out rises substantially, the family-invariant objective is working.")
    print("If it remains low, Block1 does not contain enough transferable GPU/CPU structure")
    print("and semantic supervision must reach the base LM pretraining stage.")
    print("Output dir:",outdir)


if __name__=="__main__":
    main()
