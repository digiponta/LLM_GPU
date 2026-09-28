# semantic_aligned_v3_v137.py
# v1.3.7 aligned six-class dataset.
# Reuses v1.3.6 definition/architecture/function/application axes.
# Replaces comparison with fully symmetric reciprocal comparison prompts.

from semantic_aligned_v2_v136 import DATASET as V136_DATASET, CLASSES, AXES
from semantic_symmetric_comparison_v137 import DATASET as SYMMETRIC_COMPARISON

DATASET = [x for x in V136_DATASET if x[1] != "comparison"]
DATASET += list(SYMMETRIC_COMPARISON)

assert len(DATASET) == 300
for c in CLASSES:
    assert sum(1 for x in DATASET if x[0] == c) == 50
    for a in AXES:
        assert sum(1 for x in DATASET if x[0] == c and x[1] == a) == 10
