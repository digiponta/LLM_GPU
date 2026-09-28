# cpu_gpu_relational_binding_v0111.py
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn

RELATION_LABELS = (
    "cpu_controls_gpu",
    "gpu_controlled_by_cpu",
    "cpu_assigns_work_gpu",
    "gpu_executes_for_cpu",
)

class RelationHead(nn.Module):
    def __init__(self, d_model:int, num_relations:int=len(RELATION_LABELS)):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.GELU(),
            nn.Linear(d_model,num_relations),
        )
    def forward(self,x): return self.net(x)

class RelationalEntityBinding(nn.Module):
    def __init__(self, intent_dim:int, relation_dim:int, vocab_size:int, rank:int=32, beta:float=1.0):
        super().__init__()
        self.intent_dim=int(intent_dim); self.relation_dim=int(relation_dim)
        self.vocab_size=int(vocab_size); self.rank=int(rank); self.beta=float(beta)
        self.down=nn.Linear(self.intent_dim+self.relation_dim,self.rank,bias=False)
        self.up=nn.Linear(self.rank,self.vocab_size,bias=False)
        nn.init.normal_(self.down.weight,mean=0.0,std=0.02)
        nn.init.zeros_(self.up.weight)
    def forward(self,intent_prob,relation_prob):
        x=torch.cat([intent_prob,relation_prob],dim=-1)
        return self.beta*self.up(torch.tanh(self.down(x)))

def save_checkpoint(filename, relation_head, binding, labels:Sequence[str], target_token_ids:dict, **meta):
    torch.save({
        "relation_labels":list(RELATION_LABELS),
        "relation_head_state":relation_head.state_dict(),
        "binding_state":binding.state_dict(),
        "intent_labels":list(labels),
        "target_token_ids":dict(target_token_ids),
        "d_model":relation_head.net[0].in_features,
        "rank":binding.rank,
        "beta":binding.beta,
        "vocab_size":binding.vocab_size,
        **meta,
    },filename)

def load_checkpoint(filename, labels:Sequence[str], device):
    ck=torch.load(filename,map_location=device)
    if list(ck["intent_labels"])!=list(labels):
        raise ValueError("Intent label order mismatch.")
    rh=RelationHead(int(ck["d_model"])).to(device)
    rh.load_state_dict(ck["relation_head_state"]); rh.eval()
    up_w=ck["binding_state"]["up.weight"]
    down_w=ck["binding_state"]["down.weight"]
    b=RelationalEntityBinding(
        len(labels),
        len(RELATION_LABELS),
        int(ck.get("vocab_size", up_w.shape[0])),
        int(ck.get("rank", down_w.shape[0])),
        float(ck.get("beta",1.0)),
    ).to(device)
    b.load_state_dict(ck["binding_state"]); b.eval()
    return rh,b,ck
