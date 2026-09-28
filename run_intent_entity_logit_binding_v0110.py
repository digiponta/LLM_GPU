# run_intent_entity_logit_binding_v0110.py
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

RESULTS=Path("results/intent_entity_logit_binding_v0110")
TRAIN_LOG=RESULTS/"train.log"; EVAL_LOG=RESULTS/"evaluation.log"
CHECKPOINT="model/model-gpu-v0.11.0-intent-entity-binding.pt"

def env():
    e=os.environ.copy(); e["PYTHONIOENCODING"]="utf-8"; e["PYTHONUTF8"]="1"; return e

def run(cmd,log):
    p=subprocess.run(cmd,text=True,encoding="utf-8",errors="strict",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env())
    out=p.stdout or ""; log.write_text(out,encoding="utf-8"); print(out,end="" if out.endswith("\n") else "\n")
    if p.returncode!=0: raise SystemExit(p.returncode)

def main():
    RESULTS.mkdir(parents=True,exist_ok=True); py=sys.executable
    print("===================================================="); print(" Clean Semantic-to-Generation Binding v0.11.0"); print("====================================================")
    print("Base model  : model/model-gpu-v0.8-chat-clean.pt"); print("Intent head : model/model-gpu-v0.8-intent-head-clean.pt")
    print("Backbone/head frozen: yes"); print("Binding application: first generated token only"); print("Technical gate: 0.50"); print("Checkpoint:",CHECKPOINT)
    print("\n[1/2] Training binding adapter")
    run([py,"train_intent_entity_logit_binding_v0110.py","--output",CHECKPOINT,"--gate-threshold","0.50"],TRAIN_LOG)
    print("\n[2/2] Evaluating fixed 30-case benchmark")
    run([py,"evaluate_intent_entity_logit_binding_v0110.py","--binding",CHECKPOINT],EVAL_LOG)
    print("\n===================================================="); print(" v0.11.0 completed"); print("====================================================")
    print("Train log:",TRAIN_LOG); print("Eval log :",EVAL_LOG); print("Checkpoint:",CHECKPOINT)
if __name__=="__main__": main()
