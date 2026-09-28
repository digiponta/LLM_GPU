# semantic_role_position_v142.py
# v1.4.2 Role-Position Disentanglement Dataset
#
# For each of the 30 semantic comparison topics from v1.3.7:
#   target=A, contrast=B, target-first
#   target=A, contrast=B, contrast-first
#   target=B, contrast=A, target-first
#   target=B, contrast=A, contrast-first
#
# Meaning/role and surface order are therefore independently varied.
# 30 topics x 4 conditions = 120 prompts.

from semantic_symmetric_comparison_v137 import CLASSES, PAIR_TOPICS

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}

PAIR_KEYS = []
PAIR_TO_ID = {}
DATASET = []

def canonical_pair(a, b):
    return tuple(sorted((a,b), key=lambda x: CLASS_TO_ID[x]))

def make_target_first(target_desc, contrast_desc):
    return f"「{target_desc}」を指し、「{contrast_desc}」ではないものは何ですか。"

def make_contrast_first(target_desc, contrast_desc):
    return f"「{contrast_desc}」ではなく、「{target_desc}」を指すものは何ですか。"

topic_id = 0
for (a,b), topics in PAIR_TOPICS.items():
    key = canonical_pair(a,b)
    if key not in PAIR_TO_ID:
        PAIR_TO_ID[key] = len(PAIR_KEYS)
        PAIR_KEYS.append(key)

    for local_topic_id,(a_desc,b_desc) in enumerate(topics, start=1):
        topic_id += 1
        pair_id = PAIR_TO_ID[key]

        # A is target.
        role_a = 0 if a == key[0] else 1
        DATASET.append({
            "topic_id": topic_id,
            "pair_id": pair_id,
            "pair": key,
            "target": a,
            "contrast": b,
            "role_direction": role_a,
            "surface_order": 0,  # target-first
            "prompt": make_target_first(a_desc,b_desc),
        })
        DATASET.append({
            "topic_id": topic_id,
            "pair_id": pair_id,
            "pair": key,
            "target": a,
            "contrast": b,
            "role_direction": role_a,
            "surface_order": 1,  # contrast-first
            "prompt": make_contrast_first(a_desc,b_desc),
        })

        # B is target.
        role_b = 0 if b == key[0] else 1
        DATASET.append({
            "topic_id": topic_id,
            "pair_id": pair_id,
            "pair": key,
            "target": b,
            "contrast": a,
            "role_direction": role_b,
            "surface_order": 0,
            "prompt": make_target_first(b_desc,a_desc),
        })
        DATASET.append({
            "topic_id": topic_id,
            "pair_id": pair_id,
            "pair": key,
            "target": b,
            "contrast": a,
            "role_direction": role_b,
            "surface_order": 1,
            "prompt": make_contrast_first(b_desc,a_desc),
        })

assert len(PAIR_KEYS) == 15
assert topic_id == 30
assert len(DATASET) == 120

for c in CLASSES:
    assert sum(1 for s in DATASET if s["target"] == c) == 20

assert sum(1 for s in DATASET if s["surface_order"] == 0) == 60
assert sum(1 for s in DATASET if s["surface_order"] == 1) == 60
assert sum(1 for s in DATASET if s["role_direction"] == 0) == 60
assert sum(1 for s in DATASET if s["role_direction"] == 1) == 60
