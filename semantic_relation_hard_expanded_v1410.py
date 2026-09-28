# semantic_relation_hard_expanded_v1410.py
# v1.4.10 Hard-Family Expansion
# Base: v1.4.9 100 phrases
# Expand ordinal and contrast from 10 -> 20 paraphrases per label.
# Total: 140 phrases.

from copy import deepcopy
from semantic_relation_expanded_v149 import FIRST_RELATIONS as _FIRST_V149
from semantic_relation_expanded_v149 import SECOND_RELATIONS as _SECOND_V149

FIRST_RELATIONS = deepcopy(_FIRST_V149)
SECOND_RELATIONS = deepcopy(_SECOND_V149)

FIRST_RELATIONS["ordinal"] += [
    "順番では先にある概念が対象となる",
    "位置番号が1の概念を選ぶ",
    "先頭から数えて一つ目を答えとする",
    "第一位置にある概念が該当する",
    "順序を基準にすると前の方が正解である",
    "並びの最初にある概念を採用する",
    "一番手の概念を対象とみなす",
    "先頭側の候補が答えになる",
    "序列上で第一の概念を指す",
    "前から一番目にある概念が該当する",
]

SECOND_RELATIONS["ordinal"] += [
    "順番では後にある概念が対象となる",
    "位置番号が2の概念を選ぶ",
    "先頭から数えて二つ目を答えとする",
    "第二位置にある概念が該当する",
    "順序を基準にすると後ろの方が正解である",
    "並びの二番目にある概念を採用する",
    "二番手の概念を対象とみなす",
    "後方側の候補が答えになる",
    "序列上で第二の概念を指す",
    "前から二番目にある概念が該当する",
]

FIRST_RELATIONS["contrast"] += [
    "二者を比べると採用されるのは前者である",
    "後者を不採用とし、前者を残す",
    "対比した場合、正しい側は前である",
    "後ろ側は条件に合わず、前側が条件を満たす",
    "後者を外した結果、前者が対象になる",
    "二つを対照すると前者だけが該当する",
    "後の候補を退けて、先の候補を採る",
    "前者を肯定し、後者を否定する",
    "選別すると残るのは前側の概念である",
    "対立する二候補のうち前者を正解とする",
]

SECOND_RELATIONS["contrast"] += [
    "二者を比べると採用されるのは後者である",
    "前者を不採用とし、後者を残す",
    "対比した場合、正しい側は後ろである",
    "前側は条件に合わず、後ろ側が条件を満たす",
    "前者を外した結果、後者が対象になる",
    "二つを対照すると後者だけが該当する",
    "先の候補を退けて、後の候補を採る",
    "後者を肯定し、前者を否定する",
    "選別すると残るのは後ろ側の概念である",
    "対立する二候補のうち後者を正解とする",
]

FAMILIES = list(FIRST_RELATIONS.keys())
assert FAMILIES == list(SECOND_RELATIONS.keys())

EXPECTED = {
    "direct": 10,
    "select": 10,
    "ordinal": 20,
    "contrast": 20,
    "referent": 10,
}

for fam in FAMILIES:
    assert len(FIRST_RELATIONS[fam]) == EXPECTED[fam]
    assert len(SECOND_RELATIONS[fam]) == EXPECTED[fam]

RELATION_SAMPLES = []
for fam in FAMILIES:
    for text in FIRST_RELATIONS[fam]:
        RELATION_SAMPLES.append({"family": fam, "target_position": 0, "text": text})
    for text in SECOND_RELATIONS[fam]:
        RELATION_SAMPLES.append({"family": fam, "target_position": 1, "text": text})

assert len(RELATION_SAMPLES) == 140
