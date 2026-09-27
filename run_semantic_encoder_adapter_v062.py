# run_semantic_encoder_adapter_v062.py

from __future__ import annotations
import os,subprocess,sys
from pathlib import Path

CHECKPOINT="model/model-gpu-v0.9.1-semantic-adapter-v062.pt"
RESULTS_DIR=Path("results/semantic_encoder_adapter_v062")
TRAIN_LOG=RESULTS_DIR/"train.log"
EVAL_LOG=RESULTS_DIR/"eval.log"

def env_utf8():
    e=os.environ.copy(); e["PYTHONIOENCODING"]="utf-8"; e["PYTHONUTF8"]="1"; return e

def run(cmd,log):
    c=subprocess.run(cmd,text=True,encoding="utf-8",errors="strict",
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False,env=env_utf8())
    out=c.stdout or ""; log.write_text(out,encoding="utf-8")
    print(out,end="" if out.endswith("\n") else "\n")
    if c.returncode!=0: raise SystemExit(c.returncode)

def main():
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    py=sys.executable
    train=[
        py,"train_semantic_encoder_adapter_v062.py",
        "--init-adapter","model/model-gpu-v0.9.1-semantic-adapter-v061.pt",
        "--output",CHECKPOINT,
        "--lr","1e-4",
        "--neighborhood-weight","2.0",
        "--neighborhood-margin","0.20",
        "--preservation-weight","1.0",
    ]
    evaluate=[py,"evaluate_semantic_encoder_adapter_v062.py","--adapter",CHECKPOINT]

    print()
    print("====================================================")
    print(" Semantic Encoder Adapter v0.6.2 Experiment")
    print("====================================================")
    print("Branch               : v0.9.1")
    print("Initialization       : semantic adapter v0.6.1")
    print("Focus                : G05-neighborhood binding")
    print("Exact G05 training   : no")
    print("Base encoder         : frozen")
    print("Adapter              : frozen")
    print("Concept/attr heads   : frozen")
    print("Hierarchy head       : trainable")
    print("Generation training  : none")
    print("Checkpoint           :",CHECKPOINT)
    print("Results dir          :",RESULTS_DIR)
    print()
    print("[1/2] Training G05-neighborhood binding")
    run(train,TRAIN_LOG)
    print()
    print("[2/2] Evaluating neighborhood binding")
    run(evaluate,EVAL_LOG)
    print()
    print("Experiment complete.")
    print("Primary gate:")
    print("  G05 centroid -> CPU")
    print("  G05 heterogeneous_instruction > homogeneous_computation")
    print("  G08 Transformer retained")
    print("  technical held-out accuracy does not regress")
    print("  G05-neighborhood paraphrases -> CPU-like")
    print("  homogeneous computation probes -> GPU-like")
    print()
    print("Logs:")
    print(" ",TRAIN_LOG)
    print(" ",EVAL_LOG)

if __name__=="__main__":
    main()
