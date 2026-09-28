# relation_multiseed_stability_v148.py
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch

from semantic_explicit_role_span_v143 import DATASET as CONCEPT_DATASET, CLASSES
from semantic_relation_paraphrase_v145 import RELATION_SAMPLES, FAMILIES
from evaluate_transformer_continuation_category_repair_v01117 import DEFAULT_TOKENIZER, DEFAULT_MODEL
from model import LanguageModel
from tokenizer_bpe import Tokenizer
from relation_paraphrase_generalization_v145 import encode_text_stages, concept_cv_features
from relation_projection_regularization_sweep_v147 import family_cv, reconstruct_target

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
STAGE = "block5"
ARCH = "mlp16_16"
HIDDEN = 16
PROJ_DIM = 16
SUPCON_WEIGHT = 0.25
CONCEPT_SEEDS = [41,42,43]


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def oracle_target_accuracy(p1,p2):
    hits=[]
    for i,s in enumerate(CONCEPT_DATASET):
        pos=0 if s["target"]==s["first_class"] else 1
        pred=int(p1[i]) if pos==0 else int(p2[i])
        hits.append(pred==CLASS_TO_ID[s["target"]])
    return sum(hits)/len(hits)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",default=DEFAULT_TOKENIZER)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--epochs",type=int,default=400)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--weight-decay",type=float,default=1e-3)
    ap.add_argument("--temperature",type=float,default=0.1)
    ap.add_argument("--seeds",default="1-20",
                    help="Relation training seeds, e.g. 1-20 or 1,2,3")
    ap.add_argument("--output-dir",default="results/relation_multiseed_stability_v148")
    args=ap.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        a,b=args.seeds.split("-",1)
        relation_seeds=list(range(int(a),int(b)+1))
    else:
        relation_seeds=[int(x) for x in args.seeds.split(",") if x.strip()]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=Tokenizer.load(args.tokenizer)
    model,_=LanguageModel.load_checkpoint(args.model,device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("="*118)
    print(" LLM_GPU v1.4.8 Multi-seed Stability Evaluation")
    print("="*118)
    print("Device           :",device)
    print("Base             :",args.model)
    print("Stage            :",STAGE)
    print("Fixed setting    : mlp16_16 + SupCon 0.25")
    print("Projection       : 256 -> 16 -> 16")
    print("Relation seeds   :",f"{relation_seeds[0]}..{relation_seeds[-1]}" if len(relation_seeds)>1 else relation_seeds[0])
    print("Seed count       :",len(relation_seeds))
    print("Concept seeds    :",",".join(map(str,CONCEPT_SEEDS)),"(fixed ensemble)")
    print("Evaluation       : leave-one-relation-family-out")
    print("Base model       : completely frozen")
    print()

    print("Encoding relation phrases...")
    cache={s["text"]:encode_text_stages(model,tok,s["text"]) for s in RELATION_SAMPLES}
    X=torch.stack([cache[s["text"]][STAGE] for s in RELATION_SAMPLES]).to(device)
    y=torch.tensor([s["target_position"] for s in RELATION_SAMPLES],
                   dtype=torch.long,device=device)

    print("Preparing fixed concept decoder reference...")
    p1,p2=concept_cv_features(model,tok,STAGE,args,CONCEPT_SEEDS)
    y1=torch.tensor([CLASS_TO_ID[s["first_class"]] for s in CONCEPT_DATASET],dtype=torch.long)
    y2=torch.tensor([CLASS_TO_ID[s["second_class"]] for s in CONCEPT_DATASET],dtype=torch.long)
    concept_acc=(
        float((p1==y1).float().mean())+
        float((p2==y2).float().mean())
    )/2.0
    oracle=oracle_target_accuracy(p1,p2)

    rows=[]
    rel_scores=[]
    target_scores=[]
    centroid_scores=[]
    within_scores=[]
    per_family={f:[] for f in FAMILIES}

    print()
    print("Per-seed results")
    print("-"*118)
    for seed in relation_seeds:
        rel,pred,fams,centroid,within=family_cv(
            X,y,ARCH,HIDDEN,PROJ_DIM,SUPCON_WEIGHT,args,[seed]
        )
        target=reconstruct_target(pred,p1,p2)

        rel_scores.append(rel)
        target_scores.append(target)
        centroid_scores.append(centroid)
        within_scores.append(within)
        for fam in FAMILIES:
            per_family[fam].append(fams[fam])

        rows.append({
            "seed":seed,
            "relation_accuracy":rel,
            "target_accuracy":target,
            "centroid_cosine":centroid,
            "within":within,
            **{f"family_{fam}":fams[fam] for fam in FAMILIES},
        })

        famtxt=" ".join(f"{fam[:3]}={fams[fam]:.0%}" for fam in FAMILIES)
        print(
            f"seed={seed:>3d}  relation={rel:>6.1%}  target={target:>6.1%}  "
            f"centroid={centroid:+.3f}  within={within:.3f}  {famtxt}"
        )

    rel_mean,rel_std=mean_std(rel_scores)
    tgt_mean,tgt_std=mean_std(target_scores)
    cent_mean,cent_std=mean_std(centroid_scores)
    within_mean,within_std=mean_std(within_scores)

    outdir=Path(args.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)

    fields=[
        "seed","relation_accuracy","target_accuracy","centroid_cosine","within",
        *[f"family_{fam}" for fam in FAMILIES]
    ]
    with (outdir/"per_seed.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    family_summary=[]
    for fam in FAMILIES:
        m,s=mean_std(per_family[fam])
        family_summary.append((fam,m,s,min(per_family[fam]),max(per_family[fam])))

    with (outdir/"family_stability.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["family","mean_accuracy","std_accuracy","min_accuracy","max_accuracy"])
        for row in family_summary:
            w.writerow(row)

    with (outdir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow([
            "architecture","supcon_weight","seed_count",
            "relation_mean","relation_std","relation_min","relation_max",
            "target_mean","target_std","target_min","target_max",
            "concept_accuracy","oracle_target_accuracy",
            "centroid_mean","centroid_std","within_mean","within_std",
        ])
        w.writerow([
            ARCH,SUPCON_WEIGHT,len(relation_seeds),
            rel_mean,rel_std,min(rel_scores),max(rel_scores),
            tgt_mean,tgt_std,min(target_scores),max(target_scores),
            concept_acc,oracle,
            cent_mean,cent_std,within_mean,within_std,
        ])

    print()
    print("Multi-seed summary")
    print("-"*88)
    print(f"unseen relation mean ± std           : {rel_mean:.1%} ± {rel_std:.1%}")
    print(f"unseen relation min / max            : {min(rel_scores):.1%} / {max(rel_scores):.1%}")
    print(f"reconstructed Target mean ± std      : {tgt_mean:.1%} ± {tgt_std:.1%}")
    print(f"reconstructed Target min / max       : {min(target_scores):.1%} / {max(target_scores):.1%}")
    print(f"fixed concept accuracy               : {concept_acc:.1%}")
    print(f"oracle relation target upper bound   : {oracle:.1%}")
    print(f"centroid cosine mean ± std           : {cent_mean:+.3f} ± {cent_std:.3f}")
    print(f"within-class cosine mean ± std       : {within_mean:.3f} ± {within_std:.3f}")

    print()
    print("Per-family stability")
    print("-"*88)
    for fam,m,s,mn,mx in family_summary:
        print(f"{fam:10s} mean={m:.1%} std={s:.1%} min={mn:.1%} max={mx:.1%}")

    print()
    print("Reference")
    print("---------")
    print("v1.4.5 best unseen relation accuracy : 60.0%")
    print("v1.4.6 unseen relation accuracy      : 55.0%")
    print("v1.4.7 best single setting           : 70.0%")
    print("Binary chance                        : 50.0%")

    print()
    print("Interpretation")
    print("--------------")
    print("This experiment measures whether the v1.4.7 best configuration is stable")
    print("across relation-head initialization seeds while the frozen LLM, dataset,")
    print("concept decoder, architecture, and SupCon strength remain unchanged.")
    print("A mean clearly above the 60% v1.4.5 baseline with modest variance would")
    print("support proceeding to relation-dataset expansion rather than more tuning.")
    print("Output dir:",outdir)


if __name__=="__main__":
    main()
