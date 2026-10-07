# LLM_GPU v1.7.1 — NDC Semantic Classification Phase 1/2

This branch adds a hierarchical Japanese Decimal Classification (NDC) router to LLM_GPU.

## Scope

- **Phase 1:** 10 main classes, codes `0`–`9`
- **Phase 2:** 100 divisions, codes `00`–`99`
- Existing model weights are **not retrained**
- Existing `semantic_vector()` is reused as the representation
- Routing is hierarchical: **10 main classes -> one set of 10 child divisions**

## Architecture

```text
input
  |
  v
LLM_GPU semantic_vector()
  |
  v
Phase 1 centroid router
  |-- 0 General works
  |-- 1 Philosophy
  |-- 2 History
  |-- 3 Social sciences
  |-- 4 Natural sciences
  |-- 5 Technology / engineering
  |-- 6 Industry
  |-- 7 Arts
  |-- 8 Language
  `-- 9 Literature
  |
  v
Phase 2 child router
  |
  +-- e.g. 4 -> 40..49
  |             41 Mathematics
  |             42 Physics
  |             43 Chemistry
  |             ...
  |
  v
NDC Phase-2 result
```

## Examples

Expected semantic targets:

| Query | Phase 1 | Phase 2 |
|---|---|---|
| AIとは | 0 総記 | 00 総記 (007 information science is below this division) |
| 量子力学とは | 4 自然科学 | 42 物理学 |
| 数学とは | 4 自然科学 | 41 数学 |
| 日本史とは | 2 歴史 | 21 日本史 |
| 経済学とは | 3 社会科学 | 33 経済 |
| 電気回路とは | 5 技術・工学 | 54 電気工学 |
| 日本文学とは | 9 文学 | 91 日本文学 |

Phase 3 can later refine a Phase-2 result to 3-digit NDC classes, for example:
`00 -> 007 -> 007.13` for AI and `42 -> 421 -> 421.3` for quantum mechanics.

## Commands

Structural regression (does not require a model checkpoint):

```powershell
python verify_ndc_phase12.py
```

Display the Phase 1/2 taxonomy:

```powershell
python run_ndc_phase12.py --show-taxonomy
```

Classify a query:

```powershell
python run_ndc_phase12.py "量子力学とは"
python run_ndc_phase12.py "AIとは"
```

## Design note

This is a zero-shot semantic baseline. The branch intentionally does not train new NDC weights yet.
The next evaluation should measure Phase-1 accuracy, Phase-2 accuracy, hierarchy-consistent accuracy,
and semantic/tree distance on held-out paraphrases before adding learned projection or fine-tuning.
