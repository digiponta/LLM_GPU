# semantic_contrast_decomposition_v1412.py
# v1.4.12 Contrast Relation Decomposition
# Explicitly split each contrast phrase into a negation span and a selection span.
# selected_position: 0=FIRST, 1=SECOND
# negated_position is always the opposite side.

CONTRAST_SAMPLES = [
    # FIRST selected / SECOND negated
    {"text":"後者を除外し、前者を採用する","negated_text":"後者を除外する","selected_text":"前者を採用する","selected_position":0},
    {"text":"後ろの概念ではなく前の概念が正しい","negated_text":"後ろの概念ではない","selected_text":"前の概念が正しい","selected_position":0},
    {"text":"後者ではない方、つまり前者が対象である","negated_text":"後者ではない","selected_text":"前者が対象である","selected_position":0},
    {"text":"後ろ側を退け、前側を残す","negated_text":"後ろ側を退ける","selected_text":"前側を残す","selected_position":0},
    {"text":"前者を採用し、後者は採用しない","negated_text":"後者は採用しない","selected_text":"前者を採用する","selected_position":0},
    {"text":"対象は後者ではなく前者である","negated_text":"後者ではない","selected_text":"前者が対象である","selected_position":0},
    {"text":"後ろの候補を除くと前の候補が残る","negated_text":"後ろの候補を除く","selected_text":"前の候補が残る","selected_position":0},
    {"text":"後者を否定して前者を選ぶ","negated_text":"後者を否定する","selected_text":"前者を選ぶ","selected_position":0},
    {"text":"前の概念を正とし、後ろの概念を除外する","negated_text":"後ろの概念を除外する","selected_text":"前の概念を正とする","selected_position":0},
    {"text":"比較すると前者の方が該当する","negated_text":"後者は該当しない","selected_text":"前者の方が該当する","selected_position":0},
    {"text":"二者を比べると採用されるのは前者である","negated_text":"後者は採用されない","selected_text":"前者が採用される","selected_position":0},
    {"text":"後者を不採用とし、前者を残す","negated_text":"後者を不採用とする","selected_text":"前者を残す","selected_position":0},
    {"text":"対比した場合、正しい側は前である","negated_text":"後ろ側は正しくない","selected_text":"前側が正しい","selected_position":0},
    {"text":"後ろ側は条件に合わず、前側が条件を満たす","negated_text":"後ろ側は条件に合わない","selected_text":"前側が条件を満たす","selected_position":0},
    {"text":"後者を外した結果、前者が対象になる","negated_text":"後者を外す","selected_text":"前者が対象になる","selected_position":0},
    {"text":"二つを対照すると前者だけが該当する","negated_text":"後者は該当しない","selected_text":"前者だけが該当する","selected_position":0},
    {"text":"後の候補を退けて、先の候補を採る","negated_text":"後の候補を退ける","selected_text":"先の候補を採る","selected_position":0},
    {"text":"前者を肯定し、後者を否定する","negated_text":"後者を否定する","selected_text":"前者を肯定する","selected_position":0},
    {"text":"選別すると残るのは前側の概念である","negated_text":"後ろ側の概念は残らない","selected_text":"前側の概念が残る","selected_position":0},
    {"text":"対立する二候補のうち前者を正解とする","negated_text":"後者は正解ではない","selected_text":"前者を正解とする","selected_position":0},

    # SECOND selected / FIRST negated
    {"text":"前者を除外し、後者を採用する","negated_text":"前者を除外する","selected_text":"後者を採用する","selected_position":1},
    {"text":"前の概念ではなく後ろの概念が正しい","negated_text":"前の概念ではない","selected_text":"後ろの概念が正しい","selected_position":1},
    {"text":"前者ではない方、つまり後者が対象である","negated_text":"前者ではない","selected_text":"後者が対象である","selected_position":1},
    {"text":"前側を退け、後ろ側を残す","negated_text":"前側を退ける","selected_text":"後ろ側を残す","selected_position":1},
    {"text":"後者を採用し、前者は採用しない","negated_text":"前者は採用しない","selected_text":"後者を採用する","selected_position":1},
    {"text":"対象は前者ではなく後者である","negated_text":"前者ではない","selected_text":"後者が対象である","selected_position":1},
    {"text":"前の候補を除くと後ろの候補が残る","negated_text":"前の候補を除く","selected_text":"後ろの候補が残る","selected_position":1},
    {"text":"前者を否定して後者を選ぶ","negated_text":"前者を否定する","selected_text":"後者を選ぶ","selected_position":1},
    {"text":"後ろの概念を正とし、前の概念を除外する","negated_text":"前の概念を除外する","selected_text":"後ろの概念を正とする","selected_position":1},
    {"text":"比較すると後者の方が該当する","negated_text":"前者は該当しない","selected_text":"後者の方が該当する","selected_position":1},
    {"text":"二者を比べると採用されるのは後者である","negated_text":"前者は採用されない","selected_text":"後者が採用される","selected_position":1},
    {"text":"前者を不採用とし、後者を残す","negated_text":"前者を不採用とする","selected_text":"後者を残す","selected_position":1},
    {"text":"対比した場合、正しい側は後ろである","negated_text":"前側は正しくない","selected_text":"後ろ側が正しい","selected_position":1},
    {"text":"前側は条件に合わず、後ろ側が条件を満たす","negated_text":"前側は条件に合わない","selected_text":"後ろ側が条件を満たす","selected_position":1},
    {"text":"前者を外した結果、後者が対象になる","negated_text":"前者を外す","selected_text":"後者が対象になる","selected_position":1},
    {"text":"二つを対照すると後者だけが該当する","negated_text":"前者は該当しない","selected_text":"後者だけが該当する","selected_position":1},
    {"text":"先の候補を退けて、後の候補を採る","negated_text":"先の候補を退ける","selected_text":"後の候補を採る","selected_position":1},
    {"text":"後者を肯定し、前者を否定する","negated_text":"前者を否定する","selected_text":"後者を肯定する","selected_position":1},
    {"text":"選別すると残るのは後ろ側の概念である","negated_text":"前側の概念は残らない","selected_text":"後ろ側の概念が残る","selected_position":1},
    {"text":"対立する二候補のうち後者を正解とする","negated_text":"前者は正解ではない","selected_text":"後者を正解とする","selected_position":1},
]

assert len(CONTRAST_SAMPLES) == 40
assert sum(s["selected_position"]==0 for s in CONTRAST_SAMPLES) == 20
assert sum(s["selected_position"]==1 for s in CONTRAST_SAMPLES) == 20
for s in CONTRAST_SAMPLES:
    s["negated_position"] = 1 - s["selected_position"]
