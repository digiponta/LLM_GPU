# semantic_relation_paraphrase_v145.py
# v1.4.5 Relation Phrase Paraphrase Generalization
#
# Both FIRST and SECOND target-position labels appear in train and test.
# Generalization is measured by holding out relation wording families, not labels.

FIRST_RELATIONS = {
    "direct": [
        "前の概念を指し、後ろの概念ではない",
        "前者が対象で、後者ではない",
    ],
    "select": [
        "二つのうち前の概念を選ぶ",
        "後ろではなく前の概念を選択する",
    ],
    "ordinal": [
        "第一の概念が該当する",
        "二番目ではなく一番目の概念が答えである",
    ],
    "contrast": [
        "後者を除外し、前者を採用する",
        "後ろの概念ではなく前の概念が正しい",
    ],
    "referent": [
        "参照すべき対象は前の概念である",
        "該当するのは前者であり後者ではない",
    ],
}

SECOND_RELATIONS = {
    "direct": [
        "前の概念ではなく、後ろの概念を指す",
        "後者が対象で、前者ではない",
    ],
    "select": [
        "二つのうち後ろの概念を選ぶ",
        "前ではなく後ろの概念を選択する",
    ],
    "ordinal": [
        "第二の概念が該当する",
        "一番目ではなく二番目の概念が答えである",
    ],
    "contrast": [
        "前者を除外し、後者を採用する",
        "前の概念ではなく後ろの概念が正しい",
    ],
    "referent": [
        "参照すべき対象は後ろの概念である",
        "該当するのは後者であり前者ではない",
    ],
}

FAMILIES = list(FIRST_RELATIONS.keys())
assert FAMILIES == list(SECOND_RELATIONS.keys())
for fam in FAMILIES:
    assert len(FIRST_RELATIONS[fam]) == 2
    assert len(SECOND_RELATIONS[fam]) == 2

RELATION_SAMPLES = []
for fam in FAMILIES:
    for text in FIRST_RELATIONS[fam]:
        RELATION_SAMPLES.append({"family":fam,"target_position":0,"text":text})
    for text in SECOND_RELATIONS[fam]:
        RELATION_SAMPLES.append({"family":fam,"target_position":1,"text":text})

assert len(RELATION_SAMPLES) == 20
