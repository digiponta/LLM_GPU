# semantic_explicit_role_span_v143.py
# v1.4.3 Explicit Role Marker / Span Dataset
#
# Structured sample:
#   concept_1 description
#   concept_2 description
#   relation phrase
#
# No TARGET/CONTRAST label token is inserted into the model input.
# Role must be inferred from the natural-language relation phrase.
#
# 30 topics x 2 semantic target roles x 2 surface orders = 120 samples.

from semantic_symmetric_comparison_v137 import CLASSES, PAIR_TOPICS

CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}
PAIR_KEYS = []
PAIR_TO_ID = {}
DATASET = []

def canonical_pair(a,b):
    return tuple(sorted((a,b), key=lambda x: CLASS_TO_ID[x]))

def add_sample(topic_id, pair_id, key, first_class, first_desc, second_class, second_desc,
               target_class, surface_order):
    role_direction = 0 if target_class == key[0] else 1

    # Neutral relation wording: refers only to "first concept" and "second concept".
    # It does not name GPU/CPU/etc. and does not contain target/contrast labels.
    if target_class == first_class:
        relation = "前の概念を指し、後ろの概念ではない"
    else:
        relation = "前の概念ではなく、後ろの概念を指す"

    DATASET.append({
        "topic_id": topic_id,
        "pair_id": pair_id,
        "pair": key,
        "first_class": first_class,
        "second_class": second_class,
        "first_desc": first_desc,
        "second_desc": second_desc,
        "target": target_class,
        "role_direction": role_direction,
        "surface_order": surface_order,  # 0 target-first, 1 contrast-first
        "relation": relation,
    })

topic_id = 0
for (a,b),topics in PAIR_TOPICS.items():
    key = canonical_pair(a,b)
    if key not in PAIR_TO_ID:
        PAIR_TO_ID[key] = len(PAIR_KEYS)
        PAIR_KEYS.append(key)
    pair_id = PAIR_TO_ID[key]

    for a_desc,b_desc in topics:
        topic_id += 1

        # Target=A, target-first.
        add_sample(topic_id,pair_id,key,a,a_desc,b,b_desc,a,0)
        # Target=A, contrast-first.
        add_sample(topic_id,pair_id,key,b,b_desc,a,a_desc,a,1)
        # Target=B, target-first.
        add_sample(topic_id,pair_id,key,b,b_desc,a,a_desc,b,0)
        # Target=B, contrast-first.
        add_sample(topic_id,pair_id,key,a,a_desc,b,b_desc,b,1)

assert len(DATASET) == 120
assert len(PAIR_KEYS) == 15
assert topic_id == 30
for c in CLASSES:
    assert sum(1 for s in DATASET if s["target"] == c) == 20
assert sum(1 for s in DATASET if s["surface_order"] == 0) == 60
assert sum(1 for s in DATASET if s["surface_order"] == 1) == 60
