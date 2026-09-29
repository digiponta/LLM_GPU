# LLM_GPU

A homemade Japanese Transformer language-model project implemented in Python /
PyTorch and accelerated with CUDA.

The project began as a minimal educational LLM and has evolved through tokenizer,
model-scale, conversational-learning, semantic-gating, and online-learning
experiments. The current stable experimental baseline is **v1.6.26**.

## Current Stable Baseline: v1.6.26

### Current model

```text
Tokenizer       : byte-level BPE
Vocabulary size : 8,000
Parameters      : 8,960,000
Context length  : 512 tokens
d_model         : 256
Transformer     : 6 layers
Attention heads : 8
GPU             : tested on NVIDIA GeForce RTX 3070 Ti
Online model    : model/model-gpu-v1.6.2-online.pt
Tokenizer file  : model/tokenizer-v0.7-bpe.json
Concept calib   : model/concept-calibration-v1512.pt
```

The current v1.6.x work is no longer only a text-generation experiment. It
also investigates whether a small self-made LLM can support controlled
conversation learning while rejecting low-quality, unknown, or semantically
inconsistent answers.

### Current chat pipeline

```text
User input
   |
   v
Input Quality Check
   |
   +-- malformed / subjectless ----------------> INPUT_REJECT
   |
   v
LLM generation
   |
   +-- multiple generation probes
   +-- token confidence
   +-- top-2 margin
   +-- lexical agreement
   +-- semantic probe agreement
   +-- intent / slot analysis
   +-- question-answer semantic consistency
   +-- concept calibration
   |
   v
Semantic Gate
   |
   +-- ACCEPT
   +-- LEARNING_GAP
   +-- UNKNOWN_KNOWLEDGE
   +-- GATE_REVIEW
   +-- INPUT_REJECT
```

Only accepted turns can enter normal conversational history. Rejected responses
are not silently treated as valid knowledge.

## v1.6 Learning and Recovery

The conversational learning loop uses explicit trusted examples rather than
blind self-training.

```text
ACCEPT
  -> normal response

LEARNING_GAP
  |
  +-- trusted teacher exists
  |      -> /maintain
  |      -> reactivate trusted pair
  |      -> /train
  |
  +-- no trusted teacher
         -> NEEDS_TEACHING
         -> /teach or /teachq
         -> /train

UNKNOWN_KNOWLEDGE
  -> knowledge queue

GATE_REVIEW
  -> gate-review queue

INPUT_REJECT
  -> ask/rephrase
```

### Learning data

Trusted conversational examples are stored in:

```text
data/chat_history.jsonl
```

Persistent deduplication state is stored in:

```text
data/chat_learning_state.json
```

Routing queues:

```text
data/teaching_queue.jsonl
data/knowledge_queue.jsonl
data/gate_review_queue.jsonl
```

Legacy automatic chat captures are ignored by the trainer. Trusted sources are
manual teaching, approved answers, and validated recovery examples.

### Incremental trainer

`online_train.py` performs assistant-answer-only incremental SFT.

Current behavior includes:

- SHA-256 pair fingerprints for deduplication;
- only new or explicitly reactivated trusted pairs are trained;
- weak replay of prior trusted examples;
- replay from the conversational corpus;
- stronger weighting for recovery examples;
- recovery-only stabilization at a lower learning rate;
- persistent tracking of consumed trusted examples.

Typical training weights:

```text
manual teaching     : x4
approved answer     : x4
recovery example    : x8
prior trusted replay: x1
corpus replay       : x1
```

For tiny updates, the trainer uses the training loss to select the best epoch
instead of creating an unstable validation split from only a few examples.

## Chat Commands

Start the current model with:

```powershell
python chat.py
```

Commands:

```text
/reset
    clear conversation history

/info
    show model/checkpoint information

/learn on
/learn off
/learn status
    control accepted-turn capture

/teach TEXT
    teach a corrected answer for the immediately preceding user question

/teachq QUESTION => ANSWER
    teach an explicit question/answer pair; does not depend on conversation state

/good
    approve the previous accepted AI answer as trusted learning data

/maintain
    recover the immediately preceding question from trusted teaching data

/maintain all
    inspect/recover all pending teaching candidates

/train
    run incremental training and reload the resulting checkpoint

/exit
    quit
```

Example explicit teaching:

```text
/teachq LLMとは => LLMは大量のテキストから学習し、言語を扱う大規模言語モデルです。
/train
LLMとは
```

A successful result is expected to reach:

```text
gate=KNOWN
resolution=ACCEPT
```

## Forgetting-Aware Recovery

v1.6.15-v1.6.25 introduced a recovery mechanism for knowledge that had been
trained previously but was no longer produced reliably after later incremental
updates.

Important safeguards include:

1. A previously trained pair can be explicitly reactivated.
2. Recovery candidates must pass the current teaching validator.
3. Definition recovery preserves the exact target concept.
   `AIとは` cannot repair `LLMとは` merely because both strings end in
   `とは`.
4. If no valid trusted teacher exists, recovery returns
   `NEEDS_TEACHING` instead of inventing a teacher.
5. `/maintain` is targeted to the previous question; `/maintain all` is
   explicit.
6. Rejected candidates can be displayed diagnostically so generation failures
   and gate failures can be distinguished.

## v1.6.26 Regression Baseline

The current integration test is:

```powershell
python run_chat_learning_regression_v1626.py
```

Latest verified result:

```text
Passed : 24/24
Failed : 0/24
Result : PASS
```

The 24 tests cover:

- subjectless and malformed input rejection;
- known-definition input acceptance;
- intent and semantic-slot extraction;
- informative versus echo-only teaching;
- wrong-concept teaching rejection;
- five-way resolution routing;
- exact-concept recovery;
- prevention of AI-to-LLM cross-concept recovery;
- safe behavior when no trusted teacher exists.

The regression suite uses temporary files for recovery tests and does not
modify the real learning log, learning state, or routing queues.

## Current Key Files

```text
model.py
    Transformer language model

tokenizer_bpe.py
    byte-level BPE tokenizer

chat.py
    v1.6.26 interactive chat, semantic gate, teaching and recovery routing

online_train.py
    forgetting-aware incremental conversational trainer

run_chat_learning_regression_v1626.py
    v1.6.26 integration/regression suite

evaluate_chat.py
    conversational evaluation

infer.py
    raw interactive inference

check_gpu.py
    CUDA/PyTorch diagnostic
```

## Installation

Install a CUDA-enabled PyTorch build appropriate for the local NVIDIA driver,
then install the project requirements.

```powershell
python -m pip install -r requirements.txt
python check_gpu.py
```

A successful CUDA check should report:

```text
CUDA available : True
Device         : cuda
GPU            : NVIDIA ...
```

## Recommended v1.6.26 Workflow

```powershell
python run_chat_learning_regression_v1626.py
python chat.py
```

When a known concept is forgotten:

```text
ask the question
/maintain
/train
ask the question again
```

When no trusted teacher exists:

```text
/teachq QUESTION => CORRECT_ANSWER
/train
ask the question again
```

## Experimental Status

v1.6.26 is an **experimental stable baseline**, not a claim of production-grade
general-purpose intelligence.

The main result of the v1.6 series is that this small homemade model now has a
controlled loop for:

```text
generation
-> semantic validation
-> rejection/routing
-> explicit teaching
-> incremental training
-> forgetting recovery
-> regression verification
```

This baseline is intended to make subsequent experiments reproducible without
losing the behavior established during the v1.6 development cycle.

---

# Historical Experiments

The sections below preserve the earlier LLM_GPU development history. Some
architecture sizes, checkpoint names, and commands are historical and should
not be interpreted as the current v1.6.26 defaults.



## Note: Practical Training Data Scale for LLM_GPU

The following figure summarizes how much training data is needed to make the current **LLM_GPU** configuration practical.

![How Much Training Data Is Needed to Make LLM_GPU Practical?](docs/llm_gpu_training_data_scale.svg)

### Current model

The current LLM_GPU configuration is approximately:

- Vocabulary: **~5,000**
- `d_model`: **64**
- Transformer layers: **2**
- Parameters: **~0.75M**

### Training scale guideline

| Stage | Training tokens (approx.) | Expected status |
|---|---:|---|
| Initial test | 100K–500K | Starts to generate sentence-like text |
| Small-scale experiment | 1M–3M | Learning behavior and loss trends become visible |
| Prototype for a specific domain | 3M–10M | May become useful for limited-domain tasks |
| Practical limit evaluation of the current model | 10M–30M | The capacity limit of the current architecture becomes visible |
| General-purpose conversational LLM | 100M–several billion | Requires a substantially larger model |

A useful rule of thumb is to train on roughly **10–30 times the number of model parameters**. For the current model:

```text
0.75M parameters × 20 ≈ 15M tokens
```

Therefore, **10M–20M tokens** is a good experimental target for the current LLM_GPU implementation.

If validation loss, perplexity, and generated-text quality stop improving significantly in the **10M–30M token** range, the bottleneck is likely to be **model capacity rather than data volume**.

### Suggested scale-up

A reasonable next model for comparison is:

- Vocabulary: **8,000–16,000**
- `d_model`: **128**
- Layers: **4–6**
- Heads: **4–8**
- Parameters: **a few million to ~10M**
- Training data: **30M–200M tokens**

This comparison is useful for determining whether the next limitation comes from **insufficient training data** or **insufficient model capacity**.

### Relevance to LLM_SEM and QHA

LLM_GPU does not necessarily need to become a large general-purpose conversational model. In the broader architecture, a more useful role is:

```text
Input Text
   ↓
LLM_GPU
   ↓
Semantic Vector
   ↓
LLM_SEM
   ↓
VM Routing (QHA)
```

For this use case, the main evaluation targets are:

- semantic representation quality,
- embedding stability,
- task-classification accuracy,
- routing quality for VM selection.

For QHA-oriented experiments, **10M–20M tokens of pretraining plus task-specific semantic data** can therefore be a meaningful practical target.

### Recommended next experiment

1. Train the current LLM_GPU model to about **10M tokens**.
2. Evaluate training loss, validation loss, perplexity, generated-text quality, and semantic representation quality.
3. Compare it with a larger model such as **`d_model=128` and 4 layers**.
4. Use the comparison to separate **data-scale limits** from **model-scale limits**.


---

## v0.5 Conversational Learning Experiment

v0.5 adds a second training stage for testing whether the existing sub-million-
parameter model can acquire basic short Japanese conversational behavior.

The design deliberately keeps the v0.4 architecture and tokenizer unchanged:

```text
general-ja.txt + data-nagato.txt
        |
        v
train_corpus.py
        |
        v
model-gpu-v0.4.pt
        |
        +---- data/conversation-ja.txt
        |
        v
train_conversation.py
        |
        v
model-gpu-v0.5-chat.pt
        |
        +---- chat.py
        |
        +---- evaluate_chat.py
```

### Conversation training data

The included compact corpus uses short role markers to save context:

```text
人: こんにちは。
AI: こんにちは。今日は何について話しましょうか。

人: GPUとは何ですか。
AI: 多数の計算を並列に処理するのが得意な演算装置です。
```

The current model has a context length of only 64 characters, so the examples
and expected replies are intentionally short.

### Step 1: prepare the v0.4 base model

If `model/model-gpu-v0.4.pt` and `model/tokenizer.json` already exist,
reuse them. Otherwise run:

```powershell
python train_corpus.py
```

### Step 2: conversational fine-tuning

```powershell
python train_conversation.py
```

Default fine-tuning settings (revised SFT):

```text
base checkpoint : model/model-gpu-v0.4.pt
conversation data: data/conversation-ja.txt
output checkpoint: model/model-gpu-v0.5-chat.pt
epochs (maximum): 40
learning rate   : 5e-5
batch size      : 16
validation ratio: 0.15
early-stop patience: 6
```

The revised trainer treats each `人:` / `AI:` pair as one supervised
training example. Cross-entropy loss is calculated only on the AI answer;
the user prompt and padding are masked out. This avoids the first v0.5
implementation's excessive repetition of a tiny continuous corpus, which
could produce a very low loss while still giving poor conversational replies.

The script keeps the existing tokenizer. It reports the percentage of
`<UNK>` tokens before training and warns when the conversation corpus contains
too many characters not represented by the v0.4 vocabulary.

Parameters can be changed from the command line, for example:

```powershell
python train_conversation.py --epochs 60 --learning-rate 5e-5 --patience 8
```

### Step 3: chat

```powershell
python chat.py
```

Commands:

```text
/reset   clear short conversation history
/exit    quit
```

The chat interface uses the same `人:` / `AI:` format as the fine-tuning
corpus and retains a small amount of dialogue history. The model still has a
64-character context limit, so long multi-turn conversation is not expected.

### Step 4: evaluate

```powershell
python evaluate_chat.py
```

The evaluation uses greedy decoding for reproducibility and reports:

- keyword hit rate on held-out prompts,
- percentage of non-empty replies,
- a simple repetition sanity check,
- mean reply length.

These are regression metrics for this experiment, not a claim of general
conversational intelligence. Generated replies should also be inspected
manually.

### Experiment objective

The v0.5 experiment asks:

> Can a sub-million-parameter Transformer acquire basic Japanese
> conversational behavior through dialogue-oriented fine-tuning?

A useful comparison is:

```text
v0.4 pretrained model
        vs.
v0.5 conversation-fine-tuned model
```

If v0.5 learns speaker turn-taking and short responses but factual coverage,
coherence, or multi-turn memory remain weak, the next bottleneck is likely the
small model capacity and 64-character context rather than the chat interface
itself.


### v0.5 SFT correction

The initial conversational experiment used a continuous next-character window
dataset with 50,000 repeated samples. A checkpoint loss near zero could
therefore indicate memorization rather than useful question-to-answer
behavior.

The revised implementation uses:

```text
one dialogue pair
      |
      v
人: <question>
AI: <answer>
      |
      +-- prompt tokens: loss masked
      |
      +-- AI answer tokens: loss enabled
      v
validation split + early stopping
```

`chat.py` was also changed to stop on the first generated newline or EOS,
use a lower default temperature, and apply repetition penalty only to tokens
already generated in the answer. This is especially important for questions
such as "GPUとは何ですか", because prompt words are no longer penalized when
the answer needs to reuse them.


---

## v0.6: Larger Conversational Model

v0.6 is the next experiment after the v0.5 conversational evaluation showed
that the 0.75M-parameter / context-64 model did not generalize reliably.

The v0.6 pipeline is:

```text
             general-ja + data-nagato
                       70%
                         \
conversation-ja 20% ---> Mixed Pretraining ---> v0.6 pretrained model
                         /
instruction-ja 10% -----+
                                |
                                v
                    Assistant-only SFT
                                |
                                v
                    model-gpu-v0.6-chat.pt
                         /              \
                        v                v
                    chat.py      evaluate_chat.py
```

### v0.6 architecture

```text
Vocabulary           : built from all mixed-training sources
d_model              : 128
Transformer layers   : 4
Attention heads      : 4
FFN dimension        : 512
Context length       : 256
Positional encoding  : learned positional embedding
Parameter scale      : approximately 2M+ (depends on vocabulary size)
```

The v0.6 model adds true multi-head causal attention and learned positional
embeddings. Older v0.4/v0.5 checkpoints remain readable because missing
`num_heads` and positional-embedding settings default to the legacy behavior.

### New files

```text
train_mixed.py          v0.6 70/20/10 mixed pretraining
train_sft_v06.py        v0.6 assistant-answer-only SFT
data/instruction-ja.txt compact Japanese instruction / QA corpus
```

The existing `chat.py` and `evaluate_chat.py` use the v0.6 tokenizer and
chat checkpoint by default on this branch.

### Stage 1: mixed pretraining

Required local corpora:

```text
data/general-ja.txt      or ../LLM/data/general-ja.txt
data/data-nagato.txt     or ../LLM/data/data-nagato.txt
data/conversation-ja.txt
data/instruction-ja.txt
```

Run:

```powershell
python train_mixed.py
```

Default configuration:

```text
mixture          : 70% general / 20% conversation / 10% instruction
samples          : 500,000
context          : 256
epochs           : 1
batch size       : 32
learning rate    : 3e-4
token exposures  : about 128M
```

Outputs:

```text
model/tokenizer-v0.6.json
model/model-gpu-v0.6-pretrain.pt
```

For a shorter first smoke test:

```powershell
python train_mixed.py --samples 20000
```

For a larger run:

```powershell
python train_mixed.py --samples 1000000 --epochs 1
```

### Stage 2: conversational SFT

After mixed pretraining:

```powershell
python train_sft_v06.py
```

Default SFT configuration:

```text
base checkpoint  : model/model-gpu-v0.6-pretrain.pt
tokenizer        : model/tokenizer-v0.6.json
context          : inherited from model (256)
epochs maximum   : 30
batch size       : 16
learning rate    : 2e-5
validation split : 15%
early stopping   : patience 5
```

The SFT loss is calculated only for assistant answer tokens:

```text
人: <user prompt>     -> masked from loss
AI: <assistant reply> -> optimized by cross entropy
```

Output:

```text
model/model-gpu-v0.6-chat.pt
```

### Stage 3: evaluation and chat

Evaluate first:

```powershell
python evaluate_chat.py
```

Then interact:

```powershell
python chat.py
```

The v0.6 chat defaults are less restrictive than v0.5 because the model has a
larger context and greater capacity:

```text
max new tokens     : 96
temperature        : 0.45
top-k              : 20
history turns      : 3
repetition penalty : 1.05
```

### Recommended experimental sequence

Do not start with the full mixed-pretraining run. First verify the complete
pipeline:

```powershell
python train_mixed.py --samples 20000
python train_sft_v06.py
python evaluate_chat.py
python chat.py
```

If the pipeline works correctly, delete or overwrite the smoke-test v0.6
checkpoints by running the normal mixed-pretraining command:

```powershell
python train_mixed.py
python train_sft_v06.py
python evaluate_chat.py
```

This provides a direct experimental comparison:

```text
v0.5
~0.75M params / context 64 / 1 head
                  versus
v0.6
~2M+ params / context 256 / 4 heads / positional embedding
/ mixed pretraining / assistant-only SFT
```


### v0.6 conversational-quality correction

After the first v0.6 smoke evaluation improved the keyword hit rate but still
showed phrase collapse such as malformed Japanese and repeated high-frequency
phrases, the training pipeline was revised.

The mixed-pretraining schedule is now curriculum-based:

```text
first 80% of samples:
  90% general / 7% conversation / 3% instruction

last 20% of samples:
  70% general / 20% conversation / 10% instruction
```

This prevents the very small dialogue and instruction corpora from dominating
before the base Japanese language distribution is learned.

The v0.6 SFT stage now combines:

```text
conversation-ja.txt
        +
instruction-ja.txt converted to user/assistant pairs
        |
        v
deduplication
        |
assistant-only loss
        |
label smoothing = 0.05
        |
early stopping
```

The default SFT learning rate was reduced from `2e-5` to `1e-5` to reduce
damage to the mixed-pretrained language model.

Because the tokenizer vocabulary and the pretraining distribution changed,
old v0.6 checkpoints should not be reused after this correction. Retrain from
the beginning:

```powershell
Remove-Item model\model-gpu-v0.6-pretrain.pt -ErrorAction SilentlyContinue
Remove-Item model\model-gpu-v0.6-chat.pt -ErrorAction SilentlyContinue
Remove-Item model\tokenizer-v0.6.json -ErrorAction SilentlyContinue

python train_mixed.py --samples 20000
python train_sft_v06.py
python evaluate_chat.py
```

If the corrected smoke test is sound, proceed to the full run with
`python train_mixed.py`.


---

## v0.7: Byte-level BPE Tokenizer Experiment

v0.7 keeps the successful v0.6 Transformer architecture and changes the main
experimental variable from character-level tokenization to byte-level BPE.

The motivation is the remaining v0.6 failure mode where semantically related
Japanese strings can still be confused at the character level. BPE can learn
multi-character units such as frequently occurring Japanese expressions,
technical terms, and response fragments while byte-level fallback keeps the
tokenizer robust for previously unseen Unicode text.

### v0.7 architecture

```text
Tokenizer             : byte-level BPE
Target vocabulary     : 8,000
d_model               : 128
Transformer layers    : 4
Attention heads       : 4
FFN dimension         : 512
Context length        : 256 subword tokens
Positional embedding  : learned
Mixed pretraining     : curriculum
SFT                   : assistant-only + label smoothing
```

The Transformer dimensions deliberately remain the same as v0.6 so the
tokenizer effect can be compared more directly.

### New files

```text
tokenizer_bpe.py       byte-level BPE wrapper
train_mixed_v07.py     v0.7 BPE curriculum pretraining
train_sft_v07.py       v0.7 BPE conversational/instruction SFT
```

The original `tokenizer.py`, `train_mixed.py`, and
`train_sft_v06.py` remain in the repository so the v0.6 experiment is
reproducible.

### Dependency

v0.7 adds Hugging Face `tokenizers`:

```powershell
python -m pip install -r requirements.txt
```

### Important: v0.6 checkpoints cannot be reused

A tokenizer change changes the vocabulary IDs and embedding/output dimensions.
Therefore v0.7 must be trained from scratch.

v0.7 uses separate files:

```text
model/tokenizer-v0.7-bpe.json
model/model-gpu-v0.7-pretrain.pt
model/model-gpu-v0.7-chat.pt
```

### Smoke test

```powershell
git checkout v0.7
git pull
python -m pip install -r requirements.txt

python train_mixed_v07.py --samples 20000
python train_sft_v07.py
python evaluate_chat.py
python chat.py
```

During pretraining the script reports `Chars/token`. With character-level
tokenization this value is effectively near 1 character per token; a value
above 1 for v0.7 indicates that BPE has learned multi-character units.

### Full experiment

After the complete smoke-test pipeline works:

```powershell
python train_mixed_v07.py
python train_sft_v07.py
python evaluate_chat.py
python chat.py
```

Default pretraining remains 500,000 samples, context 256, batch size 32, and
one epoch, preserving the v0.6 curriculum:

```text
Phase A (first 80%):
  90% general / 7% conversation / 3% instruction

Phase B (last 20%):
  70% general / 20% conversation / 10% instruction
```

### BPE-aware generation

With BPE, a newline or role marker may span multiple token IDs. The v0.7
`chat.py` therefore no longer assumes that newline is one token. It decodes
the generated subword sequence incrementally and stops at textual newline /
role boundaries.

### Comparison target

```text
v0.6:
  character tokenizer
  500k curriculum pretraining
  evaluation: 7/10 keyword hits

v0.7:
  byte-level BPE tokenizer
  same Transformer dimensions
  same curriculum/SFT concept
  evaluation: to be measured
```

The key v0.7 questions are:

1. Does BPE improve Japanese sentence stability?
2. Does it reduce intent confusion for paraphrased prompts?
3. Does the same context length carry more semantic content because common
   multi-character sequences become single tokens?
4. Does evaluation improve beyond the v0.6 70% result without increasing the
   Transformer depth or width?


### v0.7.1 targeted refinement

After v0.7 reached 9/10 on the original conversational regression set, the
remaining clear semantic error was confusion among closely related technical
terms (for example GPU versus LLM). The v0.7 branch therefore adds a targeted
refinement without changing model width, depth, context length, or tokenizer.

Changes:

- added contrastive paraphrases for GPU / CPU / LLM / Transformer / CUDA /
  Python,
- added explicit "X is not Y" distinction examples,
- changed SFT validation from a global random split to an intent-stratified
  split,
- reduced default SFT label smoothing from 0.05 to 0.02,
- kept the original 10-case regression benchmark,
- added a separate six-case Technical contrast diagnostic.

Because the BPE tokenizer was trained before these new examples were added,
two experiment modes are possible.

For the fastest targeted refinement, reuse the existing v0.7 pretrained BPE
checkpoint and rerun only SFT:

```powershell
git checkout v0.7
git pull
python train_sft_v07.py
python evaluate_chat.py
```

For a fully controlled v0.7.1-style experiment in which the BPE vocabulary and
mixed pretraining also see the new technical examples, retrain from scratch:

```powershell
Remove-Item model\tokenizer-v0.7-bpe.json -ErrorAction SilentlyContinue
Remove-Item model\model-gpu-v0.7-pretrain.pt -ErrorAction SilentlyContinue
Remove-Item model\model-gpu-v0.7-chat.pt -ErrorAction SilentlyContinue

python train_mixed_v07.py
python train_sft_v07.py
python evaluate_chat.py
```

The second procedure is the clean comparison against the previous v0.7 result.

### Semantic-aware evaluation correction

The original v0.7 regression evaluator counted a case as PASS when any one
keyword appeared. This could produce a false positive, for example:

```text
Prompt: GPUは何をするものですか。
Reply : GPUはコンピュータ全体の汎用的な処理を担当する演算装置です。
```

The reply contains the string `GPU`, but semantically describes a CPU.

The evaluator now uses two rules:

1. every required semantic group must match at least one synonym;
2. no forbidden/conflicting phrase may appear.

For example, the GPU case now requires both:

```text
GPU
AND
one of: 並列 / 多数の計算 / 大量の計算
```

and rejects CPU-like descriptions such as:

```text
汎用的な処理を担当
命令実行
```

The summary metric is therefore now:

```text
Semantic pass rate
```

and the technical diagnostic reports:

```text
Technical semantic rate
```

This is still a lightweight rule-based regression test rather than a general
semantic benchmark, but it avoids the main false-PASS failure found in the
previous keyword-only evaluation.



### Final v0.7 generalization hardening

The final v0.7 refinement addresses two separate goals:

1. reduce the local GPU/CPU confusion with symmetric training paraphrases;
2. measure true paraphrase generalization on prompts not present in SFT data.

Training data now contains balanced GPU/CPU examples covering role, strengths,
parallel versus sequential processing, and explicit contrast. The wording is
kept different from the generalization benchmark.

A new independent evaluator is available:

```powershell
python evaluate_generalization_v07.py
```

It contains 30 held-out prompts spanning:

- GPU / CPU and technical contrasts,
- LLM / Transformer / CUDA / Python,
- short-answer / topic-change / repeat / conversation-end controls,
- debugging and research comparison,
- simple facts and conversational intents.

The script reports:

```text
Generalization semantic rate: ?/30
Per-intent:
...
```

For this final refinement, the existing BPE tokenizer and mixed-pretrained
checkpoint may be reused because the vocabulary and architecture are unchanged.
Only SFT needs to be rerun:

```powershell
git pull
python train_sft_v07.py
python evaluate_chat.py
python evaluate_generalization_v07.py
```

For publication-quality comparison, keep both scores: the original regression
set measures regression stability, while the held-out set measures paraphrase
generalization.

### v0.7 augmented SFT + intent multi-task learning

The final v0.7 SFT now addresses the weak 7/30 held-out paraphrase result with
two complementary mechanisms.

#### 1. Deterministic SFT data augmentation

`augment_sft_v07.py` expands the original conversation/instruction pairs with
intent-specific Japanese paraphrase templates. The default setting generates up
to 24 variants per supported intent family.

Covered intents include:

```text
GPU, CPU, LLM, Transformer, CUDA, Python
short answer, topic change, repeat explanation, conversation end
debug/error, research comparison
greeting, fatigue, thanks, Japan capital
```

The augmentation is deterministic and requires no external API or LLM.
The 30 held-out prompts in `evaluate_generalization_v07.py` were checked
against the augmentation templates; there are no exact prompt overlaps.

#### 2. Intent multi-task learning

SFT now optimizes two objectives simultaneously:

```text
total_loss
  = assistant_language_model_loss
  + 0.25 * intent_classification_loss
```

The intent classifier reads the final hidden representation at the end of the
user/assistant prompt prefix. Its gradients also update the base Transformer,
encouraging semantically similar paraphrases to occupy intent-consistent
representations.

The classifier head is used only during training. Normal chat inference still
uses the same v0.7 language-model architecture and checkpoint format.

A separate diagnostic head is saved as:

```text
model/model-gpu-v0.7-intent-head.pt
```

The chat model remains:

```text
model/model-gpu-v0.7-chat.pt
```

#### Training

The existing v0.7 BPE tokenizer and mixed-pretrained checkpoint can be reused:

```powershell
git checkout v0.7
git pull

python train_sft_v07.py
python evaluate_chat.py
python evaluate_generalization_v07.py
```

Default multi-task SFT settings:

```text
variants per intent : 24
intent loss weight  : 0.25
label smoothing     : 0.02
learning rate       : 1e-5
max epochs          : 30
early stopping      : patience 5
```

Training now reports both language-model and intent metrics:

```text
train=... lm=... intent=... intent_acc=...
val=...   lm=... intent=... intent_acc=...
```

Note that the saved checkpoint loss is now the combined validation objective,
so it should not be compared directly with older answer-only SFT loss values.

The key v0.7 success criteria are now:

```text
Regression semantic rate       : preserve the previous high score
Technical semantic rate        : preserve 6/6 if possible
Generalization semantic rate   : improve substantially beyond 7/30
```

### v0.7 multi-label intent learning

The intent auxiliary task has been upgraded from one-label classification to
multi-label semantic tagging.

A prompt may now carry several tags simultaneously. Example:

```text
CPUとGPUのどちらが並列計算向きですか。

tags:
  tech_cpu
  tech_gpu
  relation_compare
  relation_distinction
  property_parallel
```

Another example:

```text
LLMとTransformerは同じ意味ですか。

tags:
  tech_llm
  tech_transformer
  relation_compare
  relation_distinction
```

The auxiliary classifier therefore uses a multi-hot target and
`BCEWithLogitsLoss` instead of single-class cross entropy.

Because most tags are absent from any one prompt, positive-class weights are
computed from the training split and capped at 10.0 to reduce all-zero bias.
Training reports multi-label micro-F1 rather than ordinary class accuracy.

```text
total_loss
  = assistant LM loss
  + 0.25 * weighted multi-label BCE loss
```

Additional relation-oriented augmentation covers CPU/GPU, LLM/Transformer,
CUDA/GPU, and Python/CUDA comparisons. These relation prompts were checked
against the 30 held-out generalization prompts; there are no exact overlaps.

Run:

```powershell
git checkout v0.7
git pull

python train_sft_v07.py
python evaluate_chat.py
python evaluate_generalization_v07.py
```

Expected training diagnostics now include:

```text
Intent tags        : ...
Positive weights   : min=... max=...
train=... lm=... intent=... tag_f1=...
val=...   lm=... intent=... tag_f1=...
```

The intent head remains auxiliary. Normal `chat.py` inference continues to
use only the language model checkpoint.

## v0.8: Capacity Scaling Experiment

v0.8 tests whether the remaining v0.7 generalization ceiling is primarily a
model-capacity limitation.

The tokenizer, SFT data, augmentation logic, multi-label intent objective, and
held-out 30-case generalization benchmark are retained. The main change is the
Transformer capacity.

### Architecture

```text
                     v0.7            v0.8
Tokenizer            BPE 8k          same v0.7 BPE
d_model              128             256
Layers               4               6
Attention heads      4               8
FFN dimension        512             1024
Context length       256             512
SFT objective        multi-label     multi-label
```

v0.8 intentionally reuses:

```text
model/tokenizer-v0.7-bpe.json
```

so tokenization does not become another experimental variable.

v0.8 writes separate checkpoints:

```text
model/model-gpu-v0.8-pretrain.pt
model/model-gpu-v0.8-chat.pt
model/model-gpu-v0.8-intent-head.pt
```

### RTX 3070 Ti defaults

Because attention memory grows strongly with context length, the default batch
sizes are reduced:

```text
pretraining batch size : 16
SFT batch size         : 8
```

If CUDA runs out of memory, reduce them further:

```powershell
python train_mixed_v08.py --batch-size 8
python train_sft_v08.py --batch-size 4
```

### Smoke test

Before the full run:

```powershell
git checkout v0.8
git pull

python train_mixed_v08.py --samples 20000
python train_sft_v08.py
python evaluate_chat.py
python evaluate_generalization_v07.py
```

### Full experiment

After the smoke test succeeds, rerun pretraining at the normal scale:

```powershell
python train_mixed_v08.py
python train_sft_v08.py
python evaluate_chat.py
python evaluate_generalization_v07.py
python chat.py
```

The 30 held-out prompts are unchanged from v0.7. The primary comparison is:

```text
v0.7 multi-label generalization : 16/30 = 53.3%
v0.8 larger model               : to be measured
```

The experiment asks whether increasing width, depth, attention heads, and
context can improve unseen paraphrase generalization while preserving the
existing regression and technical-semantic scores.

Note: v0.8 uses a 512-token context during pretraining, so token exposure per
sample is twice that of the 256-token v0.7 setup. Therefore the experiment is
best interpreted as a practical capacity-and-context scaling test rather than a
perfect single-variable parameter-count ablation.

### v0.8 technical concept binding refinement

After v0.8 reached 19/30 (63.3%) on the fixed held-out generalization set,
the architecture is kept unchanged and only technical concept binding is
refined.

The target concepts are:

```text
GPU
CPU
LLM
Transformer
CUDA
Python
```

The refinement uses matched minimal-pair prompt structures. Each concept is
trained with the same question forms, while the canonical answer changes with
the concept. Canonical answers explicitly repeat the concept name so the model
must bind the entity to its defining property rather than emit only a generic
property phrase.

Example pattern:

```text
GPUの中心的な役割を説明してください。
→ GPUの中心的な役割は大量の並列計算を効率よく処理することです。

CPUの中心的な役割を説明してください。
→ CPUの中心的な役割は多様な命令を実行し、汎用処理を制御することです。

LLMの中心的な役割を説明してください。
→ LLMの中心的な役割は言語を理解し、文章を生成することです。

Transformerの中心的な役割を説明してください。
→ Transformerの中心的な役割はAttentionで情報間の関係を処理することです。
```

Technical rows are also oversampled in the training split only:

```text
--technical-repeat 3
```

Validation rows are not duplicated. This keeps the validation distribution
unchanged while giving the six technical concepts stronger gradient exposure.

No architecture constants were changed:

```text
d_model       : 256
layers        : 6
heads         : 8
FFN           : 1024
context       : 512
tokenizer     : fixed v0.7 BPE
```

The new minimal-pair prompts were checked against both the fixed regression set
and the 30-case held-out generalization set; there are no exact prompt
overlaps.

Because only SFT data weighting and augmentation changed, v0.8 pretraining does
not need to be repeated. Run:

```powershell
git checkout v0.8
git pull

python train_sft_v08.py
python evaluate_chat.py
python evaluate_generalization_v07.py
```

The baseline to beat remains:

```text
v0.8 before binding refinement: 19/30 = 63.3%
```

For an ablation, the training-only technical emphasis can be changed without
modifying the data:

```powershell
python train_sft_v08.py --technical-repeat 1
python train_sft_v08.py --technical-repeat 2
python train_sft_v08.py --technical-repeat 3
```

`1` disables technical oversampling; `3` is the default.

### v0.8 multidimensional generalization evaluation

The fixed 30-case generalization benchmark now separates several failure modes
instead of collapsing every case into one PASS/MISS result.

For each reply, the evaluator reports:

```text
semantic-content
entity
fluency
strict
```

Definitions:

```text
semantic-content
  Checks the answer's required meaning while allowing a concept name to be
  omitted when that same concept is already explicit in the user prompt.

entity
  Checks whether prompt-mentioned technical entities are explicitly repeated
  in the answer. This distinguishes "the meaning is correct but the name was
  omitted" from a true semantic error.

fluency
  Flags obvious surface corruption such as replacement characters, long
  malformed ASCII fragments, underscore noise, and repeated fragments.
  It is intentionally conservative and is not a general grammar judge.

strict
  PASS only when semantic-content, entity explicitness, and fluency all pass.
```

The original rule-based score is retained as:

```text
Legacy rule rate
```

so older v0.7/v0.8 experiment results remain comparable.

Example:

```text
Prompt: GPU is suitable for many simultaneous calculations. What is it good at?
Reply : Large-scale parallel computation.

semantic-content : PASS
entity           : MISS
fluency          : PASS
strict           : MISS
legacy           : MISS
```

This avoids treating a concept-name omission as the same failure type as an
incorrect answer such as confusing GPU with CPU.

Run the evaluator as before:

```powershell
python evaluate_generalization_v07.py
```

The summary now reports:

```text
Semantic-content rate
Entity-explicit rate
Fluency rate
Strict composite rate
Legacy rule rate
```

The 30 prompts themselves are unchanged.

### v0.8 evaluator refinement: semantic slots and entity N/A

The multidimensional evaluator has been tightened without changing any of the
30 held-out prompts.

#### Tri-state entity scoring

Entity explicitness now uses three states:

```text
PASS  concept name was expected and explicitly present
MISS  concept name was expected but omitted
N/A   explicit entity repetition is not applicable
```

`N/A` is used when the prompt does not already name the entity and the task is
to identify it. It does not count as a failure in the strict composite score.

This prevents conversational, debugging, comparison, and name-identification
questions from artificially inflating the entity-explicit rate.

#### Stronger semantic slots

Selected cases now require multiple semantic slots instead of passing from one
broad keyword OR-group.

Examples:

```text
Transformer:
  Transformer
  AND Attention
  AND structure/model/neural-network category

CUDA:
  CUDA
  AND GPU
  AND technology/mechanism/general-computing category

Python:
  Python
  AND language category

Comparison:
  comparison/difference action
  AND common conditions/metrics
```

This specifically prevents malformed outputs that happen to contain a word such
as `条件` from being counted as semantically correct.

The summary remains backward compatible and reports:

```text
Semantic-content rate
Entity-explicit rate (applicable cases only, with N/A count)
Fluency rate
Strict composite rate
Legacy rule rate
```

Run:

```powershell
git checkout v0.8
git pull
python evaluate_generalization_v07.py
```

### v0.8 reverse-definition binding

The v0.8 architecture and the fixed 30-case evaluator remain unchanged.
This refinement targets semantic reverse lookup: infer the concept name from a
description that does not explicitly contain the name.

Fifteen training rows were added for the weak concepts:

```text
GPU
CPU
Transformer
CUDA
Python
```

Examples:

```text
Attentionを主要な仕組みとして文脈中の情報関係を扱うモデル構造は何ですか。
→ Transformerです。TransformerはAttentionを中心に文脈を処理するモデル構造です。

NVIDIAのGPUを一般的な計算処理に利用するための計算基盤は何ですか。
→ CUDAです。CUDAはNVIDIA GPUを汎用計算に利用するための技術です。

読みやすい文法と幅広い用途で知られる汎用プログラミング言語は何ですか。
→ Pythonです。Pythonは読みやすい文法を持つ汎用プログラミング言語です。
```

Two hard-negative CPU/GPU selection examples are also included so that the
same comparison structure leads to opposite answers depending on the semantic
property.

The reverse-definition prompts have no exact prompt overlap with the fixed
30-case generalization benchmark.

To isolate this refinement from the previous oversampling experiment,
`--technical-repeat` now defaults to:

```text
1
```

which disables broad technical oversampling. The architecture is unchanged:

```text
d_model       : 256
layers        : 6
heads         : 8
FFN           : 1024
context       : 512
tokenizer     : fixed v0.7 BPE
```

Pretraining does not need to be repeated. Run only SFT and evaluation:

```powershell
git checkout v0.8
git pull

python train_sft_v08.py
python evaluate_chat.py
python evaluate_generalization_v07.py
```

The refined evaluator should be used for comparison. The current pre-refinement
baseline is:

```text
Semantic-content : 18/30 = 60.0%
Strict composite : 16/30 = 53.3%
Fluency          : 29/30 = 96.7%
```

### v0.8 reverse-definition + replay balancing

The reverse-definition refinement improved the multidimensional benchmark to:

```text
Semantic-content : 20/30 = 66.7%
Strict composite : 20/30 = 66.7%
Fluency          : 30/30 = 100.0%
```

However, debug/error and some conversational-control intents regressed. To
reduce this interference without changing the architecture, v0.8 now adds
training-only replay balancing.

Protected replay tags:

```text
debug_error
control_repeat
control_topic
```

Default replay factor:

```text
--replay-repeat 2
```

This means one extra training copy is added for protected rows. Validation rows
are not replayed.

Broad technical oversampling remains disabled:

```text
--technical-repeat 1
```

so the experiment isolates:

```text
reverse-definition binding
+
small replay of regressed nontechnical intents
```

The repeat-intent tagger was also tightened. The generic word `説明` is no
longer sufficient to assign `control_repeat`, preventing ordinary technical
"explain X" prompts from being accidentally replayed as repeat-control data.

The v0.8 architecture remains unchanged:

```text
d_model       : 256
layers        : 6
heads         : 8
FFN           : 1024
context       : 512
tokenizer     : fixed v0.7 BPE
```

Pretraining does not need to be repeated:

```powershell
git checkout v0.8
git pull

python train_sft_v08.py
python evaluate_chat.py
python evaluate_generalization_v07.py
```

For ablation:

```powershell
python train_sft_v08.py --replay-repeat 1
python train_sft_v08.py --replay-repeat 2
python train_sft_v08.py --replay-repeat 3
```

`1` disables replay; `2` is the new default.

### v0.8 pairwise hard-negative binding

The v0.8 architecture, reverse-definition rows, replay balancing, and fixed
30-case generalization benchmark are retained. This refinement adds pairwise
hard-negative training for the remaining confused concepts.

Target pairs:

```text
GPU <-> CPU
Transformer <-> CUDA
CUDA <-> Python
```

Twelve new training rows use matched question structures with opposite semantic
properties. The model must choose the correct concept and explain the property
that distinguishes it from the competing concept.

Examples:

```text
CPUとGPUのうち、大量の同種計算を並列に処理する側は?
-> GPU. Parallel computation.

CPUとGPUのうち、多様な命令実行や汎用制御を主に担当する側は?
-> CPU. General-purpose processing/control.

TransformerとCUDAのうち、Attentionを中心に情報関係を処理する構造は?
-> Transformer.

TransformerとCUDAのうち、NVIDIA GPUを汎用計算に使う技術は?
-> CUDA.

CUDAとPythonのうち、GPU計算技術は?
-> CUDA.

CUDAとPythonのうち、汎用プログラミング言語は?
-> Python.
```

The prompts are paired deliberately so that the surface form stays similar
while the semantic property changes the correct answer. This is intended to
strengthen entity-property binding rather than simply increase exposure to one
technical term.

The CUDA/GPU relation is also now tagged as `relation_compare`, making the
multi-label auxiliary task more consistent with the other technical pairs.

No exact prompt overlap was found between the new pairwise rows and the fixed
30-case generalization benchmark.

Training defaults remain:

```text
--technical-repeat 1
--replay-repeat 2
```

and the architecture remains unchanged:

```text
d_model       : 256
layers        : 6
heads         : 8
FFN           : 1024
context       : 512
tokenizer     : fixed v0.7 BPE
```

Pretraining does not need to be repeated:

```powershell
git checkout v0.8
git pull

python train_sft_v08.py
python evaluate_chat.py
python evaluate_generalization_v07.py
```

The baseline immediately before this refinement is:

```text
Semantic-content : 21/30 = 70.0%
Strict composite : 21/30 = 70.0%
Fluency          : 30/30 = 100.0%

GPU             : 3/3
CPU             : 2/3
GPU/CPU         : 0/2
Transformer     : 0/1
CUDA            : 0/1
Python          : 0/1
```

## v0.9 intent-head diagnostic

Before changing the generation architecture, v0.9 adds:

```text
evaluate_intent_v09.py
```

The script runs the same fixed 30 generalization prompts and compares:

```text
expected intent tags
predicted intent tags
top intent probabilities
current generated answer
```

It loads:

```text
model/model-gpu-v0.8-chat.pt
model/model-gpu-v0.8-intent-head.pt
model/tokenizer-v0.7-bpe.json
```

and reports:

```text
Expected-tag case recall
Exact tag-set match
Micro precision
Micro recall
Micro F1
Per-intent expected-tag recall
```

The diagnostic purpose is to distinguish two cases:

```text
intent tags correct + answer wrong
    -> intent recognition exists, but generation does not directly use it

intent tags wrong + answer wrong
    -> improve intent representation/training before conditioning generation
```

Run:

```powershell
git checkout v0.9
git pull
python evaluate_intent_v09.py
```

The default intent threshold is read from the saved intent-head checkpoint.
It can be overridden for diagnosis:

```powershell
python evaluate_intent_v09.py --threshold 0.4
```

## v0.9 soft Intent-Conditioned Generation

v0.9 now implements soft intent conditioning without hard thresholding.

The conditioning path is:

```text
Prompt
  -> frozen v0.8 Transformer
  -> prompt hidden state
  -> frozen multi-label Intent Head
  -> sigmoid probabilities (24 dims)
  -> zero-initialized Linear(24 -> 256)
  -> alpha * intent bias
  -> add to LM hidden state
  -> frozen LM head
  -> answer
```

The first v0.9 experiment is intentionally projection-only. The v0.8
pairwise-best language model and its intent head are frozen, and only the small
intent projection is trained. This isolates the effect of the new
intent-to-generation path and reduces catastrophic regression risk.

Defaults:

```text
base model   : model/model-gpu-v0.8-chat-pairwise-best.pt
intent head  : model/model-gpu-v0.8-intent-head.pt
projection   : model/model-gpu-v0.9-soft-intent.pt
alpha        : 0.1
projection   : 24 -> 256
initialization: zero (exact no-op at step zero)
learning rate: 1e-3
epochs       : 20
```

No threshold is used for generation conditioning:

```python
intent_prob = sigmoid(intent_logits)
intent_bias = alpha * projection(intent_prob)
conditioned_hidden = hidden + intent_bias
```

This preserves confidence information such as a 0.46 Python score versus a
0.19 short-control score instead of converting both into hard ON/OFF tags.

Train only the v0.9 projection:

```powershell
git checkout v0.9
git pull

python train_sft_v09.py
```

Then run the unchanged 30-case benchmark through the new generation path:

```powershell
python evaluate_generalization_v09.py
```

Interactive chat:

```powershell
python chat_v09.py
```

The comparison baseline is the saved v0.8 pairwise-best checkpoint:

```text
Semantic-content : 22/30 = 73.3%
Strict composite : 22/30 = 73.3%
Fluency          : 30/30 = 100.0%
```

A useful success condition is improvement on cases where the intent head was
already correct but generation failed (for example GPU/CPU general-processing,
topic/repeat/end controls), without regressing the stable GPU, error, compare,
CUDA/GPU, and LLM/Transformer cases.

### v0.9 inference-time alpha sweep

The trained soft-intent projection can now be evaluated at multiple inference
strengths without retraining.

Run:

```powershell
git checkout v0.9
git pull

python evaluate_alpha_sweep_v09.py
```

Default sweep:

```text
alpha = 0.0, 0.1, 0.25, 0.5, 1.0
```

All values reuse the same:

```text
v0.8 pairwise-best base model
v0.8 intent head
v0.9 trained projection weights
fixed 30-case generalization benchmark
greedy generation settings
```

Only `projection.alpha` changes at inference time.

The script reports:

```text
overall semantic / strict / entity / fluency / legacy scores
strict delta versus alpha=0.0
which cases improved
which cases regressed
which replies changed
per-intent strict comparison
best observed alpha(s)
```

To print every generated reply for every alpha:

```powershell
python evaluate_alpha_sweep_v09.py --show-cases
```

Custom sweep values are also supported:

```powershell
python evaluate_alpha_sweep_v09.py --alphas 0 0.05 0.1 0.2 0.5 1.0
```

This is an inference-strength ablation only. The projection was trained at
alpha=0.1, so results at other alpha values measure post-training scaling, not
separately optimized projection checkpoints.

### v0.9 Mid-Layer Intent Conditioning

The inference-time alpha sweep showed no score change from alpha=0.0 through
1.0 for the final-layer soft-intent method. v0.9 therefore adds a second,
cleanly separated conditioning path that injects the soft intent signal inside
the Transformer instead of immediately before the LM head.

Default path:

```text
Prompt
  -> frozen v0.8 Transformer prompt representation
  -> frozen multi-label Intent Head
  -> sigmoid probabilities (24 dims)
  -> zero-initialized Linear(24 -> 256)
  -> alpha * intent embedding
  -> Transformer Block 1
  -> Transformer Block 2
  -> Transformer Block 3
  -> ADD intent embedding
  -> Transformer Block 4
  -> Transformer Block 5
  -> Transformer Block 6
  -> final norm
  -> LM head
```

The base language model and intent head remain frozen. Only the 24 -> 256
projection is trained. This preserves the v0.8 pairwise-best checkpoint while
allowing Blocks 4-6 to transform the injected semantic signal.

Defaults:

```text
base model    : model/model-gpu-v0.8-chat-pairwise-best.pt
intent head   : model/model-gpu-v0.8-intent-head.pt
projection    : model/model-gpu-v0.9-mid-intent.pt
inject-after  : 3
alpha         : 0.1
learning rate : 1e-3
epochs        : 20
```

The projection is zero-initialized, so the initial conditioned path is an exact
no-op before training.

Train:

```powershell
git checkout v0.9
git pull

python train_mid_intent_v09.py
```

Evaluate on the unchanged 30-case benchmark:

```powershell
python evaluate_mid_intent_v09.py
```

Interactive chat:

```powershell
python chat_mid_intent_v09.py
```

The main baseline remains:

```text
Semantic-content : 22/30 = 73.3%
Strict composite : 22/30 = 73.3%
```

The key cases are the prompts where intent recognition was already correct but
generation failed, especially G14, G15, G18, and G28. Improvement there would
support the hypothesis that the intent signal needs to enter the Transformer
before the final LM head.

### v0.9 Partial Fine-Tuning

The projection-only mid-layer experiment remained at the same 22/30 strict
score as the v0.8 baseline. v0.9 therefore adds partial fine-tuning so the
Transformer layers after the injection point can learn how to use the intent
signal.

Architecture:

```text
Frozen intent path
------------------
v0.8 pairwise-best model
  -> frozen intent head
  -> sigmoid probabilities (24 dims)

Generation path
---------------
Blocks 1-3      : frozen
Intent projection 24 -> 256 : trainable
Inject after Block 3
Blocks 4-6      : trainable
FinalNorm       : trainable
LM Head         : frozen
```

The intent path uses a separate frozen copy of the v0.8 model. This keeps the
intent-head input distribution fixed while the generation-side Blocks 4-6 are
adapted.

Default learning rates:

```text
Intent projection : 1e-3
Blocks 4-6        : 2e-6
FinalNorm         : 2e-6
LM Head           : frozen
```

Default checkpoint:

```text
model/model-gpu-v0.9-partial-intent.pt
```

Train:

```powershell
git checkout v0.9
git pull

python train_partial_intent_v09.py
```

Evaluate on the unchanged 30-case benchmark:

```powershell
python evaluate_partial_intent_v09.py
```

Interactive chat:

```powershell
python chat_partial_intent_v09.py
```

The comparison baseline remains:

```text
Semantic-content : 22/30 = 73.3%
Strict composite : 22/30 = 73.3%
```

The first success criterion is improvement on G14, G15, G18, and G28 without
regressing the already stable GPU, error, comparison, CUDA/GPU, and
LLM/Transformer cases.

### v0.9 Partial Fine-Tuning block-LR sweep

The partial fine-tuning experiment can now be swept automatically across:

```text
2e-6
5e-6
1e-5
2e-5
```

while keeping:

```text
projection LR : 1e-3
alpha         : 0.1
inject-after  : Block 3
LM Head       : frozen
benchmark     : same fixed 30 cases
```

Run the complete sweep:

```powershell
git checkout v0.9
git pull

python run_partial_lr_sweep_v09.py
```

For each block learning rate the script:

```text
1. trains an independent partial-intent checkpoint
2. evaluates it on the same 30-case benchmark
3. saves training and evaluation logs
4. records validation loss and evaluation metrics
5. prints an overall and per-intent comparison
```

Independent checkpoints are written as:

```text
model/model-gpu-v0.9-partial-intent-blocklr-2e-6.pt
model/model-gpu-v0.9-partial-intent-blocklr-5e-6.pt
model/model-gpu-v0.9-partial-intent-blocklr-1e-5.pt
model/model-gpu-v0.9-partial-intent-blocklr-2e-5.pt
```

Logs and CSV summary are written under:

```text
results/partial_lr_sweep_v09/
```

including:

```text
partial_lr_sweep_v09.csv
```

To print the full child-process output during the sweep:

```powershell
python run_partial_lr_sweep_v09.py --show-output
```

To re-evaluate existing sweep checkpoints without retraining:

```powershell
python run_partial_lr_sweep_v09.py --skip-training
```

The primary comparison remains the v0.8 pairwise-best baseline:

```text
Strict composite : 22/30 = 73.3%
Semantic-content : 22/30 = 73.3%
```

The experiment tests whether the unchanged 22/30 ceiling is due to insufficient
adaptation strength in Blocks 4-6, or whether the current intent representation
and training data are the more likely bottleneck.

### v0.9 local block-LR refinement

After the coarse Partial Fine-Tuning sweep, the first improvement beyond the
22/30 ceiling was observed at:

```text
block LR : 1e-5
strict   : 23/30 = 76.7%
```

The next experiment narrows the search around that point:

```text
7.5e-6
1.0e-5
1.25e-5
1.5e-5
```

Run:

```powershell
git checkout v0.9
git pull

python run_partial_lr_local_sweep_v09.py
```

The wrapper reuses the same training and fixed 30-case evaluation pipeline and
writes its results separately under:

```text
results/partial_lr_local_sweep_v09/
```

Checkpoint filenames now preserve fractional scientific-notation values without
collisions, for example:

```text
7.5e-6  -> model-gpu-v0.9-partial-intent-blocklr-7p5e-6.pt
1.25e-5 -> model-gpu-v0.9-partial-intent-blocklr-1p25e-5.pt
```

The goal is to find whether a point near 1e-5 can preserve the new `end`
improvement while avoiding the CPU regression seen at the stronger 2e-5
setting.

### v0.9 Targeted Boundary Training

After the local block-LR sweep stabilized at 23/30 across roughly
`7.5e-6` through `1.5e-5`, the next experiment targets the remaining
semantic boundaries directly while keeping the current best training setup
fixed.

The experiment adds 24 matched boundary rows covering:

```text
CPU <-> GPU
Transformer <-> CUDA <-> Python
short <-> repeat <-> topic <-> end
```

Examples are deliberately paired so that very similar surface forms require
different answers depending on the decisive semantic cue.

The boundary rows are optional and do not change the default behavior of
`augment_pairs()`. They are enabled only with:

```text
--targeted-boundary
```

This preserves reproducibility of all earlier v0.8/v0.9 runs.

Fixed settings for the first boundary experiment:

```text
block LR      : 1e-5
projection LR : 1e-3
alpha         : 0.1
inject-after  : Block 3
LM Head       : frozen
```

Run the complete experiment:

```powershell
git checkout v0.9
git pull

python run_targeted_boundary_v09.py
```

The runner trains:

```text
model/model-gpu-v0.9-partial-intent-boundary.pt
```

and then evaluates it on the unchanged fixed 30-case benchmark.

Logs are written to:

```text
results/targeted_boundary_v09/train.log
results/targeted_boundary_v09/eval.log
```

The 24 new boundary prompts were checked against the fixed 30 benchmark prompts
and have zero exact prompt overlap.

Reference before Targeted Boundary Training:

```text
Semantic-content : 23/30 = 76.7%
Strict composite : 23/30 = 76.7%
```

The main residual targets are:

```text
cpu
gpu_cpu
transformer
python
repeat
short
topic
```

The purpose of this experiment is to test whether the current plateau is caused
primarily by insufficient semantic-boundary coverage rather than by optimizer
strength or conditioning architecture.

### v0.9 Targeted Boundary Training v2

Boundary v1 improved the fixed 30-case development benchmark to:

```text
Semantic-content : 25/30 = 83.3%
Strict composite : 25/30 = 83.3%
Entity-explicit  : 9/9 = 100.0%
```

v2 keeps the v1 rows and adds 21 more matched boundary examples focused only
on the five remaining failures:

```text
G05 CPU
G08 Transformer
G12 short
G15 repeat
G18 end
```

The new rows target:

```text
CPU reverse identification
Transformer = Attention + model/structure category
short <-> repeat
end <-> topic
```

The v2 rows are optional and are enabled with:

```text
--targeted-boundary-v2
```

The dedicated v2 runner enables both v1 and v2 so that the previously gained
GPU/CPU, Python, and topic improvements are retained.

Fixed settings:

```text
block LR      : 1e-5
projection LR : 1e-3
alpha         : 0.1
inject-after  : Block 3
LM Head       : frozen
boundary v1   : enabled
boundary v2   : enabled
```

Run:

```powershell
git checkout v0.9
git pull

python run_targeted_boundary_v2_v09.py
```

Checkpoint:

```text
model/model-gpu-v0.9-partial-intent-boundary-v2.pt
```

Logs:

```text
results/targeted_boundary_v2_v09/train.log
results/targeted_boundary_v2_v09/eval.log
```

The 21 v2 prompts were checked against the unchanged fixed 30 development
prompts and have zero exact prompt overlap. Child-process output is forced to
UTF-8 on Windows.

### v0.9 Targeted Boundary Training v3 + Protected Replay

Boundary v2 remained at:

```text
Semantic-content : 25/30 = 83.3%
Strict composite : 25/30 = 83.3%
```

but improved `repeat` and the "continue later" form of `end` while
regressing Python and the immediate-stop form of `end`. This indicates
training interference rather than simple lack of boundary data.

v3 therefore keeps Boundary v1 + v2 and adds a small protected replay set for:

```text
Python
GPU/CPU relation
topic switching
end-now
continue-later
```

The protected set contains 13 prompts and is replayed in the training split
with:

```text
--protected-replay-repeat 3
```

It is deliberately separate from the normal replay tags, so this experiment
tests targeted anti-forgetting rather than broad oversampling.

Fixed settings:

```text
block LR        : 1e-5
projection LR   : 1e-3
alpha           : 0.1
inject-after    : Block 3
boundary v1     : enabled
boundary v2     : enabled
protected replay: enabled
protected repeat: 3
LM Head         : frozen
```

Run:

```powershell
git checkout v0.9
git pull

python run_targeted_boundary_v3_v09.py
```

Checkpoint:

```text
model/model-gpu-v0.9-partial-intent-boundary-v3.pt
```

Logs:

```text
results/targeted_boundary_v3_v09/train.log
results/targeted_boundary_v3_v09/eval.log
```

The 13 protected replay prompts have zero exact prompt overlap with the fixed
30 development prompts. Windows child-process output is forced to UTF-8.

The main goal is to retain the v2 gains on `repeat` and `end` while
restoring Python and preserving GPU/CPU and topic performance. Residual hard
cases G05 CPU, G08 Transformer, and G12 short should be monitored separately.

### v0.9 Boundary v4: Balanced Control Replay

Boundary v3 restored Python and both end forms, while preserving GPU/CPU and
topic, but repeat collapsed to 0/2. v4 therefore replaces the single protected
replay strength with class-specific replay strengths.

Balanced replay groups:

```text
stable protected group (repeat=2)
  Python
  GPU/CPU
  topic
  end

control boundary group (repeat=3)
  short
  repeat
```

The v3 protected replay path is intentionally disabled in the dedicated v4
runner so this experiment isolates class-specific replay balancing.

Fixed settings:

```text
block LR            : 1e-5
projection LR       : 1e-3
alpha               : 0.1
inject-after        : Block 3
boundary v1         : enabled
boundary v2         : enabled
v3 protected replay : disabled
balanced replay     : enabled
stable repeat       : 2
short/repeat repeat : 3
LM Head             : frozen
```

Run:

```powershell
git checkout v0.9
git pull

python run_targeted_boundary_v4_v09.py
```

Checkpoint:

```text
model/model-gpu-v0.9-partial-intent-boundary-v4.pt
```

Logs:

```text
results/targeted_boundary_v4_v09/train.log
results/targeted_boundary_v4_v09/eval.log
```

The balanced replay set contains 8 stable-protection prompts and 8
short/repeat-control prompts. All 16 have zero exact prompt overlap with the
fixed 30 development prompts.

The main success condition is to restore repeat while keeping Python,
GPU/CPU, topic, and both end forms correct. G05 CPU and G08 Transformer remain
hard residuals to monitor separately.

### v0.9 LM-Head Partial Unfreeze experiment

Boundary v4 regressed to 23/30, while Boundary v1 remained the cleanest
25/30 result without later replay interference. The next experiment therefore
returns to Boundary v1 data and changes only one architectural degree of
freedom: the LM head is partially unfrozen with a very small learning rate.

Trainable parameter groups:

```text
Intent projection : 1e-3
Blocks 4-6        : 1e-5
FinalNorm         : 1e-5
LM Head           : 1e-6
```

Frozen:

```text
Blocks 1-3
intent model
intent head
```

Boundary configuration:

```text
Boundary v1 : enabled
Boundary v2 : disabled
v3 replay   : disabled
v4 replay   : disabled
```

This isolates whether limited output-token adaptation can solve hard residuals
such as:

```text
G05 CPU reverse identification
G08 Transformer category completion
```

without reintroducing the data-interference effects seen in Boundary v2-v4.

Run:

```powershell
git checkout v0.9
git pull

python run_lm_head_unfreeze_v09.py
```

Checkpoint:

```text
model/model-gpu-v0.9-partial-intent-lmhead.pt
```

Logs:

```text
results/lm_head_unfreeze_v09/train.log
results/lm_head_unfreeze_v09/eval.log
```

Reference:

```text
Boundary v1 Semantic : 25/30 = 83.3%
Boundary v1 Strict   : 25/30 = 83.3%
```

Regression watch items include Python, GPU/CPU, topic, end, repeat, and compare.

### v0.9 Intent Representation v2: semantic hidden + intent probabilities

The Boundary v1 configuration remains the cleanest 25/30 development result.
Later replay experiments and LM-head unfreezing introduced interference or
output-distribution regressions. Intent Representation v2 therefore changes
only the conditioning representation.

Old representation:

```text
24-d intent probabilities
  -> Linear(24 -> 256)
  -> inject after Block 3
```

New representation:

```text
24-d intent probabilities
+
256-d frozen prompt semantic hidden
=
280-d fused representation
  -> zero-initialized Linear(280 -> 256)
  -> alpha scaling
  -> inject after Block 3
```

The semantic hidden state is read from the same frozen v0.8 intent model used
by the intent head. This keeps the semantic representation stationary while
Blocks 4-6 learn how to use the richer signal.

Fixed experiment settings:

```text
Boundary data        : v1 only
Projection LR        : 1e-3
Blocks 4-6 LR        : 1e-5
FinalNorm LR         : 1e-5
LM Head              : frozen
Alpha                : 0.1
Inject after         : Block 3
Intent model/head    : frozen
```

Run:

```powershell
git checkout v0.9
git pull

python run_intent_representation_v2_v09.py
```

Checkpoint:

```text
model/model-gpu-v0.9-intent-representation-v2.pt
```

Logs:

```text
results/intent_representation_v2_v09/train.log
results/intent_representation_v2_v09/eval.log
```

Reference:

```text
Boundary v1 Semantic : 25/30 = 83.3%
Boundary v1 Strict   : 25/30 = 83.3%
```

The main targets are G05 CPU reverse identification and G08 Transformer
category completion. The experiment also watches for regressions on Python,
GPU/CPU, topic, end, repeat, and compare.

### v0.9 Intent Representation v2.1: Semantic Bottleneck

Intent Representation v2 kept the overall score at 25/30 but did not improve
G05 CPU or G08 Transformer, and it introduced a G02 GPU regression. Its direct
280 -> 256 projection contains 71,680 trainable parameters and reached its best
validation loss at epoch 1, suggesting that the semantic path may be too large
for the available Boundary v1 training data.

v2.1 therefore compresses the frozen semantic hidden state before fusion:

```text
256-d frozen semantic hidden
  -> Linear(256 -> 32)
  -> tanh
  -> 32-d semantic feature

24-d intent probabilities
+
32-d semantic feature
=
56-d fused representation
  -> zero-initialized Linear(56 -> 256)
  -> alpha scaling
  -> inject after Block 3
```

Projection-side trainable parameters:

```text
semantic bottleneck : 256 x 32 = 8,192
fusion projection   : 56 x 256 = 14,336
total               : 22,528
```

This is substantially smaller than the v2 direct fusion path:

```text
v2   : 71,680 parameters
v2.1 : 22,528 parameters
```

Fixed experiment settings:

```text
Boundary data        : v1 only
Semantic bottleneck  : 32
Projection LR        : 1e-3
Blocks 4-6 LR        : 1e-5
FinalNorm LR         : 1e-5
LM Head              : frozen
Alpha                : 0.1
Inject after         : Block 3
Intent model/head    : frozen
```

Run:

```powershell
git checkout v0.9
git pull

python run_intent_representation_v21_v09.py
```

Checkpoint:

```text
model/model-gpu-v0.9-intent-representation-v21.pt
```

Logs:

```text
results/intent_representation_v21_v09/train.log
results/intent_representation_v21_v09/eval.log
```

Reference:

```text
Boundary v1 Semantic : 25/30 = 83.3%
Boundary v1 Strict   : 25/30 = 83.3%
Intent Rep v2        : 25/30 = 83.3%
```

Primary targets remain G05 CPU reverse identification and G08 Transformer
category completion. G02 GPU and the short/repeat/end control intents are
explicit regression-watch cases.

### v0.9 Semantic Probe Diagnostic

Before changing the conditioning or generation architecture again, v0.9 now
adds a no-training diagnostic for the frozen v0.8 pairwise-best encoder:

```text
semantic_probe_v09.py
```

The diagnostic builds technical concept centroids for:

```text
GPU
CPU
LLM
Transformer
CUDA
Python
```

using training-side reference prompts from the existing augmentation bank,
minimal-pair rows, reverse-definition rows, and Boundary v1 technical rows.

For the hard residuals G05 and G08 it reports:

```text
nearest concept centroid
runner-up centroid
top1-top2 cosine margin
expected-concept cosine
all six centroid similarities
frozen intent-head technical probabilities
```

It also defines two semantic attribute directions:

```text
CPU general/control  <-> GPU parallel
Transformer structure/model category <-> non-structure technical categories
```

and prints the target prompt projection on each direction together with anchor
scores. This helps distinguish:

```text
encoder-side confusion
vs.
semantic-to-generation mapping failure
```

Run the focused G05/G08 diagnostic:

```powershell
git checkout v0.9
git pull

python semantic_probe_v09.py
```

To include all held-out technical cases:

```powershell
python semantic_probe_v09.py --all-tech-cases
```

No model parameters are updated and no new checkpoint is produced.

### v0.9 Semantic Encoder Adapter v0.1: Concept Binding + Attribute Supervision

The Semantic Probe showed that the two hardest cases fail for different
reasons:

```text
G05 CPU
  nearest centroid : GPU
  CPU property axis: positive
  -> CPU properties exist, but concept binding is wrong

G08 Transformer
  nearest centroid : Transformer
  structure axis   : positive but weak
  -> concept identity exists, but structure/category signal is weak
```

v0.1 therefore adapts the frozen semantic encoder representation directly
without updating the base v0.8 model.

Architecture:

```text
Frozen v0.8 prompt hidden (256)
        |
        v
Residual Semantic Adapter
256 -> 64 -> 256
        |
        +---- concept head   : 256 -> 6
        |
        +---- attribute head : 256 -> 4
```

The adapter begins close to the identity mapping. The final up-projection is
zero-initialized and the residual contribution is scaled by 0.25.

Concept classes:

```text
GPU
CPU
LLM
Transformer
CUDA
Python
```

Attribute targets:

```text
parallel
general/control
language
attention/structure
```

Training objective:

```text
total loss
  = 1.00 * concept cross-entropy
  + 0.75 * attribute BCE
  + 0.25 * representation-preservation cosine loss
```

Training uses the existing technical reference prompts from the augmentation
bank, minimal pairs, reverse-definition rows, and Boundary v1 technical rows.
The fixed 30 development prompts are checked for exact overlap and training
stops if any overlap is found.

Run:

```powershell
git checkout v0.9
git pull

python run_semantic_encoder_adapter_v01.py
```

Checkpoint:

```text
model/model-gpu-v0.9-semantic-adapter-v01.pt
```

Logs:

```text
results/semantic_encoder_adapter_v01/train.log
results/semantic_encoder_adapter_v01/eval.log
```

This first experiment does not update generation parameters. It evaluates
semantic geometry first and reports, for G05 and G08:

```text
raw nearest centroid
adapted nearest centroid
raw/adapted margin
adapter representation drift
supervised concept probabilities
supervised attribute probabilities
```

Primary success conditions:

```text
G05 adapted centroid -> CPU
G08 adapted centroid -> Transformer
G08 attention/structure probability is strong
representation drift remains small
```

Only after these conditions are met should the adapter be integrated into the
generation-conditioning path.

### v0.9 Semantic Encoder Adapter v0.2: Margin-Aware Contrastive Binding

Semantic Encoder Adapter v0.1 successfully learned the supervision heads, but
G05 remained geometrically closer to the GPU centroid even though the concept
head classified it as CPU. v0.2 therefore continues from the v0.1 adapter and
adds an explicit centroid-margin objective.

New objective:

```text
sim(sample, positive centroid)
  >=
max sim(sample, negative centroid) + 0.05
```

Total loss:

```text
1.00 * Concept CE
0.75 * Attribute BCE
0.50 * Centroid Margin Loss
0.25 * Preservation Loss
```

The base v0.8 encoder remains frozen. The adapter architecture is unchanged:

```text
256 -> 64 -> 256 residual
residual scale = 0.25
```

v0.2 initializes from:

```text
model/model-gpu-v0.9-semantic-adapter-v01.pt
```

and writes:

```text
model/model-gpu-v0.9-semantic-adapter-v02.pt
```

Run:

```powershell
git checkout v0.9
git pull

python run_semantic_encoder_adapter_v02.py
```

Logs:

```text
results/semantic_encoder_adapter_v02/train.log
results/semantic_encoder_adapter_v02/eval.log
```

The evaluation now checks not only G05/G08 but also centroid accuracy across all
held-out technical prompts. The integration gate is:

```text
G05 adapted centroid -> CPU
G08 adapted centroid -> Transformer
held-out technical centroid accuracy does not regress
```

Only if that gate is met should the semantic adapter be connected to the
generation-conditioning path.

### v0.9 Semantic Encoder Adapter v0.3: Hard Negative Pair + Acronym/Full-name Binding

v0.2 preserved the technical held-out centroid accuracy at 8/10 and improved
G08, but G05 still remained closer to GPU than CPU. The concept head already
classified G05 as CPU with high confidence, so v0.3 targets the remaining
geometry mismatch directly.

v0.3 adds two mechanisms.

First, explicit hard-negative pairwise ranking:

```text
CPU          > GPU
GPU          > CPU
CUDA         > Python
Python       > CUDA
Transformer  > Python / CUDA
```

The pairwise objective is:

```text
sim(sample, positive concept)
  >=
sim(sample, specified hard negative) + 0.08
```

Second, CPU/GPU acronym and full-name binding:

```text
CPU
  <-> Central Processing Unit
  <-> central processing
  <-> general instruction execution / control

GPU
  <-> Graphics Processing Unit
  <-> graphics processing
  <-> high-throughput parallel computation
```

The GPU rows explicitly avoid teaching that GPU is graphics-only; the full name
is linked to both its historical naming and its broader modern parallel-compute
role.

Training continues from:

```text
model/model-gpu-v0.9-semantic-adapter-v02.pt
```

and writes:

```text
model/model-gpu-v0.9-semantic-adapter-v03.pt
```

Loss weights:

```text
Concept CE            : 1.00
Attribute BCE         : 0.75
Centroid Margin Loss  : 0.25
Pairwise Hard Negative: 0.75
Preservation Loss     : 0.25

centroid margin       : 0.05
pairwise margin       : 0.08
learning rate         : 2e-4
```

Run:

```powershell
git checkout v0.9
git pull

python run_semantic_encoder_adapter_v03.py
```

Logs:

```text
results/semantic_encoder_adapter_v03/train.log
results/semantic_encoder_adapter_v03/eval.log
```

The v0.3 evaluation includes unseen acronym/full-name probes rather than
replaying the binding training prompts verbatim.

Integration gate:

```text
G05 adapted centroid -> CPU
G08 adapted centroid -> Transformer
G09 CUDA improves or does not regress
technical held-out centroid accuracy does not regress
CPU/GPU unseen full-name probes pass
```

Generation remains untouched until this semantic-geometry gate is met.

### v0.9 Semantic Encoder Adapter v0.4: Acronym Contrastive Alignment

v0.3 showed that simply adding acronym/full-name rows was not sufficient:
the unseen CPU/GPU full-name probes still mapped to the wrong concept cluster.

v0.4 therefore adds an explicit acronym/alias contrastive alignment objective.
The base v0.8 encoder remains frozen and training continues from v0.3.

Alignment groups:

```text
CPU
  CPU
  Central Processing Unit
  CPU <-> Central Processing Unit statement
  CPU general/control definition

GPU
  GPU
  Graphics Processing Unit
  GPU <-> Graphics Processing Unit statement
  GPU parallel-compute definition

LLM
  LLM
  Large Language Model
  LLM <-> Large Language Model statement
  language-model definition

CUDA
  CUDA
  NVIDIA GPU computing platform
  NVIDIA GPU general-compute definition
```

For CUDA, v0.4 uses semantic aliases and definitions rather than depending on
a formal acronym expansion.

The alignment loss adapts all aliases and constructs a normalized center for
each group. Each alias must classify to its own group center under a cosine
similarity softmax:

```text
z(alias) -> adapter -> normalize
group centers = mean(normalized aliases)
logits = cosine(alias, centers) / temperature
loss = cross entropy(group)
```

Loss weights:

```text
Concept CE             : 1.00
Attribute BCE          : 0.75
Centroid Margin Loss   : 0.20
Pairwise Hard Negative : 0.50
Acronym Alignment      : 1.00
Preservation Loss      : 0.25

alignment temperature  : 0.10
learning rate          : 1e-4
```

Training continues from:

```text
model/model-gpu-v0.9-semantic-adapter-v03.pt
```

and writes:

```text
model/model-gpu-v0.9-semantic-adapter-v04.pt
```

Run:

```powershell
git checkout v0.9
git pull

python run_semantic_encoder_adapter_v04.py
```

Logs:

```text
results/semantic_encoder_adapter_v04/train.log
results/semantic_encoder_adapter_v04/eval.log
```

The integration gate remains conservative:

```text
G05 adapted centroid -> CPU
G08 adapted centroid -> Transformer
G09 CUDA improves or does not regress
technical held-out centroid accuracy does not regress
unseen CPU/GPU/LLM/CUDA alias probes pass
```

Generation remains unchanged until the semantic geometry passes this gate.

### v0.9 Semantic Encoder Adapter v0.5: Hierarchical Processor Semantics

v0.4 showed that acronym/full-name alignment alone was not enough to solve
the CPU/GPU boundary. v0.5 therefore changes the semantic supervision from a
flat CPU-vs-GPU distinction to a small hierarchy.

Hierarchy:

```text
processor
├─ CPU branch
│  ├─ general-purpose
│  └─ control-oriented
└─ GPU branch
   ├─ throughput-oriented
   └─ data-parallel
```

The hierarchy deliberately keeps CPU and GPU under the shared parent
`processor`. The distinction is therefore not "processor versus non-processor";
it is the design emphasis inside the processor family.

Hierarchy labels:

```text
processor
general_purpose
control_oriented
throughput_oriented
data_parallel
```

CPU supervision emphasizes:

```text
diverse instructions
branching and control
OS / sequential control
general-purpose workloads
low-latency mixed workloads
```

GPU supervision emphasizes:

```text
many similar operations
data parallelism
high throughput
matrix / numerical parallel workloads
```

A dedicated hierarchy head is trained together with the existing semantic
adapter:

```text
adapted hidden 256
  -> hierarchy head
  -> 5 sigmoid hierarchy probabilities
```

In addition to BCE, v0.5 adds a hierarchy contrast objective:

```text
CPU examples:
general_purpose + control_oriented
  >
throughput_oriented + data_parallel

GPU examples:
throughput_oriented + data_parallel
  >
general_purpose + control_oriented
```

Loss weights:

```text
Concept CE             : 1.00
Attribute BCE          : 0.75
Hierarchy BCE          : 1.00
Hierarchy Contrast     : 0.75
Centroid Margin Loss   : 0.20
Pairwise Hard Negative : 0.50
Preservation Loss      : 0.25

learning rate          : 1e-4
```

Training continues from:

```text
model/model-gpu-v0.9-semantic-adapter-v04.pt
```

and writes:

```text
model/model-gpu-v0.9-semantic-adapter-v05.pt
```

Run:

```powershell
git checkout v0.9
git pull

python run_semantic_encoder_adapter_v05.py
```

Logs:

```text
results/semantic_encoder_adapter_v05/train.log
results/semantic_encoder_adapter_v05/eval.log
```

The v0.5 evaluation adds a hierarchical processor probe. For G05 the key
success condition is no longer only the nearest centroid. It also requires:

```text
processor probability >= 0.5

CPU-style score
  = mean(general_purpose, control_oriented)

GPU-style score
  = mean(throughput_oriented, data_parallel)

CPU-style score > GPU-style score
```

Primary integration gate:

```text
G05 adapted centroid -> CPU
G05 hierarchy identifies processor
G05 CPU-style score > GPU-style score
G08 adapted centroid -> Transformer
technical held-out centroid accuracy does not regress
```

Generation remains unchanged until the semantic hierarchy passes this gate.

### v0.9 Semantic Adapter v0.5 -> Generation Conditioning Integration

Semantic Encoder Adapter v0.5 passed the semantic integration gate:

```text
G05 centroid        : GPU -> CPU
G05 processor       : 0.848
G05 CPU-style score : 0.963
G05 GPU-style score : 0.372
G08                 : Transformer retained
technical centroid  : 8/10 -> 9/10
```

The next experiment connects that semantic representation to generation.

Frozen semantic path:

```text
v0.8 prompt encoder
  -> Semantic Encoder Adapter v0.5
  -> adapted semantic hidden       256
  -> concept probabilities           6
  -> attribute probabilities         4
  -> hierarchy probabilities         5
                                    ---
                                    271 dims
```

Generation conditioning:

```text
271-d semantic feature
  -> Linear(271 -> 256)
  -> alpha = 0.1
  -> inject after Transformer Block 3
  -> Blocks 4-6
  -> FinalNorm
  -> LM Head
```

Training policy:

```text
semantic v0.8 encoder : frozen
semantic adapter v0.5 : frozen
semantic heads        : frozen
hierarchy head        : frozen

generation Blocks 1-3 : frozen
generation Blocks 4-6 : trainable
FinalNorm             : trainable
LM Head               : frozen

projection LR         : 1e-3
Block 4-6 LR          : 1e-5
Boundary data         : v1 only
```

The projection is zero-initialized, so conditioning begins as a no-op and the
late Transformer blocks learn how to use the semantic signal.

Run the complete experiment:

```powershell
git checkout v0.9
git pull

python run_semantic_generation_v09.py
```

Checkpoint:

```text
model/model-gpu-v0.9-semantic-generation-v05.pt
```

Logs:

```text
results/semantic_generation_v05/train.log
results/semantic_generation_v05/eval.log
```

Standalone commands:

```powershell
python train_semantic_generation_integration_v09.py
python evaluate_semantic_generation_v09.py
python chat_semantic_generation_v09.py
```

The evaluation uses the unchanged fixed 30-case development benchmark. The
reference remains the clean Boundary v1 result:

```text
Semantic-content : 25/30 = 83.3%
Strict composite : 25/30 = 83.3%
```

Primary success conditions:

```text
G05 generation becomes CPU-correct
G08 remains Transformer-correct
strict composite >= 25/30
no broad regression across other intents
```

CUDA G09 and acronym/full-name generalization remain separate residual semantic
issues and are not treated as blockers for this first generation-integration
experiment.

### v0.9.1 Semantic-to-Generation Interface v0.6

The v0.5 semantic adapter successfully corrected the semantic geometry for G05,
but the first generation integration still produced a GPU-like answer for G05.
This shows that the remaining bottleneck is the semantic-to-generation mapping,
not the semantic encoder.

v0.6 replaces the single fused 271 -> 256 projection with an explicit
semantic-gated interface.

Frozen semantic source:

```text
Semantic Adapter v0.5
  -> adapted hidden
  -> concept probabilities
  -> attribute probabilities
  -> hierarchy probabilities
```

Explicit gates:

```text
cpu_general
gpu_parallel
transformer_structure
cuda_platform
python_language
llm_language
```

The main CPU/GPU gates are defined from both concept identity and hierarchy:

```text
CPU gate
  = P(CPU)
    * mean(general_purpose, control_oriented)

GPU gate
  = P(GPU)
    * mean(
        throughput_oriented,
        data_parallel,
        property_parallel
      )
```

Transformer uses both identity and structural evidence:

```text
Transformer gate
  = P(Transformer)
    * property_attention_structure
```

Generation interface:

```text
adapted hidden (256)
  -> semantic projection (256 -> 256)
                             \
                              + -> alpha -> inject after Block 3
                             /
semantic gates (6)
  -> gate projection (6 -> 256)
```

Both projections are zero-initialized, so the interface starts as a no-op.

Training policy:

```text
semantic encoder       : frozen
semantic adapter v0.5  : frozen
semantic heads         : frozen
hierarchy head         : frozen

generation Blocks 1-3  : frozen
generation Blocks 4-6  : trainable
FinalNorm              : trainable
LM Head                : frozen

Boundary data          : v1 only
projection LR          : 1e-3
block LR               : 1e-5
alpha                  : 0.1
gate alpha             : 1.0
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_generation_gate_v091.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-gated-v06.pt
```

Logs:

```text
results/semantic_generation_gate_v091/train.log
results/semantic_generation_gate_v091/eval.log
```

The evaluator prints explicit gate values for G05, G08 and G09.

Reference before v0.6:

```text
v0.9 semantic integration:
Semantic-content : 26/30 = 86.7%
Strict composite : 25/30 = 83.3%

Residuals:
G05 CPU          : semantic signal correct, generated answer wrong
G08 Transformer  : semantic correct, fluency wrong
G12 short        : miss
G15 repeat       : miss
G18 end          : miss
```

Primary targets:

```text
G05 CPU gate > GPU gate and CPU answer becomes correct
G08 remains Transformer-correct and fluency improves
G09 CUDA remains correct
Semantic-content >= 26/30
Strict composite > 25/30 if possible
```

### v0.9.1 Semantic-to-Generation Interface v0.7: Semantic Token Conditioning

v0.6 showed that the explicit semantic gates were correct, but additive hidden
injection still did not force the language model to use those semantics during
generation. G05 had a strong CPU gate and almost no GPU gate, yet the generated
answer remained GPU-like.

v0.7 therefore changes the interface itself.

Instead of adding a semantic bias to the hidden state, v0.7 creates three
virtual semantic tokens that are visible to Self-Attention:

```text
[SEM_IDENTITY]
[SEM_CONCEPT]
[SEM_HIERARCHY]
[text hidden states...]
```

The tokens are inserted after Block 3. Blocks 4-6 therefore process the
sequence:

```text
Blocks 1-3(text)
        |
        +-- prepend semantic tokens
        |
        v
Blocks 4-6(Self-Attention over semantic tokens + text)
        |
     FinalNorm
        |
remove semantic-token output positions
        |
      LM Head
```

Semantic token definitions:

```text
SEM_IDENTITY
  adapted semantic hidden (256)
    -> Linear(256 -> 256)

SEM_CONCEPT
  concept probabilities (6)
  + attribute probabilities (4)
    -> Linear(10 -> 256)

SEM_HIERARCHY
  hierarchy probabilities (5)
    -> Linear(5 -> 256)
```

The token projectors start with a very small initialization rather than exact
zero, so an attention path exists from the beginning while the influence is
still initially small.

Training policy:

```text
semantic encoder       : frozen
semantic adapter v0.5  : frozen
semantic heads         : frozen
hierarchy head         : frozen

generation Blocks 1-3  : frozen
generation Blocks 4-6  : trainable
FinalNorm              : trainable
LM Head                : frozen

Boundary data          : v1 only
projector LR           : 1e-3
block LR               : 1e-5
semantic token count   : 3
token scale            : 1.0
insert after           : Block 3
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_token_conditioning_v091.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-token-v07.pt
```

Logs:

```text
results/semantic_token_conditioning_v091/train.log
results/semantic_token_conditioning_v091/eval.log
```

Reference:

```text
v0.9 semantic integration:
Semantic-content : 26/30 = 86.7%
Strict composite : 25/30 = 83.3%

v0.6 semantic gated:
Semantic-content : 25/30 = 83.3%
Strict composite : 24/30 = 80.0%
```

Primary targets:

```text
G05 CPU becomes generation-correct
G08 Transformer remains correct and fluency improves
G09 CUDA remains correct
Semantic-content >= 26/30
Strict composite > 25/30 if possible
```

### v0.9.1 Semantic-to-Generation Interface v0.8: Balanced Semantic Tokens

v0.7 made semantic information attention-visible, but its three semantic tokens
had extremely different magnitudes:

```text
G05
SEM_IDENTITY  = 42.784
SEM_CONCEPT   = 0.259
SEM_HIERARCHY = 0.437
```

The information needed to solve G05 lives primarily in concept/hierarchy, so
this imbalance can let the identity token dominate attention.

v0.8 balances token magnitude before Blocks 4-6.

For each semantic token:

```text
project semantic source
    ->
L2 normalize direction
    ->
multiply by sqrt(d_model)
    ->
multiply by positive learned token scale
```

With d_model=256:

```text
base token norm = sqrt(256) = 16
```

Initial learned relative scales:

```text
SEM_IDENTITY  = 1.0
SEM_CONCEPT   = 1.0
SEM_HIERARCHY = 1.0
```

The scales are parameterized as exponentials of trainable log-scales, so they
stay positive while the model can learn the relative semantic-token strength.

Architecture:

```text
SEM_IDENTITY:
  adapted hidden
    -> Linear(256 -> 256)
    -> L2 normalize
    -> norm 16 * learned identity scale

SEM_CONCEPT:
  concept + attribute
    -> Linear(10 -> 256)
    -> L2 normalize
    -> norm 16 * learned concept scale

SEM_HIERARCHY:
  hierarchy
    -> Linear(5 -> 256)
    -> L2 normalize
    -> norm 16 * learned hierarchy scale

[SEM_IDENTITY][SEM_CONCEPT][SEM_HIERARCHY][text hidden...]
    -> Blocks 4-6 Self-Attention
    -> FinalNorm
    -> remove semantic-token positions
    -> LM Head
```

Training policy remains comparable to v0.7:

```text
semantic encoder       : frozen
semantic adapter v0.5  : frozen
semantic heads         : frozen
hierarchy head         : frozen

generation Blocks 1-3  : frozen
generation Blocks 4-6  : trainable
FinalNorm              : trainable
LM Head                : frozen

Boundary data          : v1 only
projector LR           : 1e-3
block LR               : 1e-5
insert after           : Block 3
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_token_balanced_v091.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-token-balanced-v08.pt
```

Logs:

```text
results/semantic_token_balanced_v091/train.log
results/semantic_token_balanced_v091/eval.log
```

The evaluator reports:

```text
learned per-token scales
actual token norms for G05/G08/G09
30-case semantic/strict scores
```

Reference:

```text
v0.9 semantic integration : semantic 26/30, strict 25/30
v0.6 semantic gated       : semantic 25/30, strict 24/30
v0.7 semantic tokens      : semantic 25/30, strict 25/30
```

Primary targets:

```text
semantic token norms remain comparable
G05 CPU becomes generation-correct
G08 Transformer is correct and fluent
G09 CUDA remains correct
Semantic-content >= 26/30
Strict composite > 25/30 if possible
```

### v0.9.1 Semantic Attention Probe

Balanced Semantic Tokens v0.8 equalized the three semantic-token norms, but
G05 and G08 still did not improve. The next diagnostic therefore measures
whether Blocks 4-6 actually attend from text positions to the semantic tokens.

No training is performed.

The probe inspects:

```text
G05 CPU reverse identification
G08 Transformer structure
G09 CUDA reference case
```

For every Transformer block after semantic-token insertion and every attention
head, it reports:

```text
last prompt token -> SEM_IDENTITY
last prompt token -> SEM_CONCEPT
last prompt token -> SEM_HIERARCHY

total last-token semantic attention mass
mean text-to-semantic attention mass
ratio to uniform-attention baseline
```

The uniform baseline is:

```text
semantic token count / visible key count
```

Interpretation:

```text
uniform_ratio < 1.0
  semantic tokens are under-attended

uniform_ratio ~ 1.0
  semantic attention is approximately uniform

uniform_ratio > 1.0
  semantic tokens are preferentially attended
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_attention_probe_v091.py
```

Direct probe:

```powershell
python semantic_attention_probe_v091.py --cases 5,8,9
```

Log:

```text
results/semantic_attention_probe_v091/probe.log
```

Decision rule:

```text
If G05 semantic representation remains correct
but text-to-semantic attention is consistently weak,
the next interface candidate is explicit Semantic Cross-Attention:

Q = text hidden
K,V = semantic tokens
```

This probe intentionally does not change any model parameter or checkpoint.

### v0.9.1 Semantic-to-Generation Interface v0.9: Semantic Cross-Attention

The Semantic Attention Probe showed that semantic tokens are not completely
ignored. Some heads attend to them strongly, but the attention is highly uneven
across blocks and heads, and the correct semantic signal does not reliably
control generation.

v0.9 therefore introduces a dedicated semantic read path:

```text
Q   = text hidden after Block 3
K,V = balanced semantic tokens
```

Architecture:

```text
text tokens
  -> Blocks 1-3
  -> Semantic Cross-Attention
       Q = text hidden
       K = SEM_IDENTITY / SEM_CONCEPT / SEM_HIERARCHY
       V = SEM_IDENTITY / SEM_CONCEPT / SEM_HIERARCHY
  -> gated residual
  -> Blocks 4-6
  -> FinalNorm
  -> LM Head
```

The three semantic tokens come from the trained Balanced Semantic Tokens v0.8
projector, but that projector is frozen during this experiment.

Controlled comparison:

```text
semantic encoder v0.5      : frozen
balanced projector v0.8    : frozen

generation model start     : v0.8 pairwise-best
Blocks 1-3                 : frozen
Semantic Cross-Attention   : trainable
Blocks 4-6                 : trainable
FinalNorm                  : trainable
LM Head                    : frozen
```

Cross-Attention uses 8 heads. The semantic residual has a learned sigmoid scale
initialized to 0.1, so the dedicated path starts with a controlled influence.

Training:

```text
Boundary data : v1 only
Cross LR      : 1e-3
Block LR      : 1e-5
Inject after  : Block 3
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_cross_attention_v091.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-cross-attn-v09.pt
```

Logs:

```text
results/semantic_cross_attention_v091/train.log
results/semantic_cross_attention_v091/eval.log
```

The evaluator reports head-level Cross-Attention weights for G05, G08, and G09:

```text
text -> SEM_IDENTITY
text -> SEM_CONCEPT
text -> SEM_HIERARCHY
```

References before v0.9:

```text
v0.9 semantic integration : semantic 26/30, strict 25/30
v0.6 semantic gated       : semantic 25/30, strict 24/30
v0.7 semantic tokens      : semantic 25/30, strict 25/30
v0.8 balanced tokens      : semantic 25/30, strict 25/30
```

Primary targets:

```text
G05 CPU becomes generation-correct
G08 Transformer becomes correct and fluent
G09 CUDA remains correct
Semantic-content >= 26/30
Strict composite > 25/30 if possible
```

### v0.9.1 Semantic Encoder Adapter v0.6: Instruction Semantics

The G05 residual suggests that the model may still compress "instruction
execution" toward "computation". v0.6 therefore refines CPU/GPU semantics
before further generation-interface work.

The existing v0.5 processor hierarchy is extended with two explicit axes:

```text
processor
├─ CPU branch
│  ├─ general-purpose
│  ├─ control-oriented
│  └─ heterogeneous-instruction
│
└─ GPU branch
   ├─ throughput-oriented
   ├─ data-parallel
   └─ homogeneous-computation
```

CPU supervision now explicitly teaches that instructions include more than
arithmetic:

```text
arithmetic / logic
branch / jump
compare
load / store
memory access
call / return
interrupt
I/O control
OS / control flow
```

The intended distinction is:

```text
CPU
  = flexible execution of heterogeneous instruction types,
    including control and memory operations

GPU
  = high-throughput parallel execution of many similar computations
```

This experiment continues from semantic adapter v0.5. The base v0.8 encoder
remains frozen.

New hierarchy labels:

```text
heterogeneous_instruction
homogeneous_computation
```

The new instruction-contrast objective enforces:

```text
CPU examples:
heterogeneous_instruction + general_purpose + control_oriented
  >
homogeneous_computation + throughput_oriented + data_parallel

GPU examples:
homogeneous_computation + throughput_oriented + data_parallel
  >
heterogeneous_instruction + general_purpose + control_oriented
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_encoder_adapter_v06.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-adapter-v06.pt
```

Logs:

```text
results/semantic_encoder_adapter_v06/train.log
results/semantic_encoder_adapter_v06/eval.log
```

The evaluation includes direct probes for:

```text
G05
non-arithmetic instructions
heterogeneous CPU instructions
homogeneous GPU computation
GPU throughput
```

Primary gate:

```text
G05 centroid -> CPU
G05 heterogeneous_instruction > homogeneous_computation
non-arithmetic instruction probe -> CPU-like
heterogeneous instruction probe -> CPU-like
homogeneous computation probe -> GPU-like
G08 Transformer retained
technical held-out accuracy does not regress
```

Generation is intentionally not retrained in this step. The semantic definition
is validated first.

### v0.9.1 Semantic Encoder Adapter v0.6.1: Instruction Binding Refinement

v0.6 passed 7/8 semantic gates. The remaining failure was specific:

```text
G05:
heterogeneous_instruction
<
homogeneous_computation
```

even though the overall CPU-style score and centroid classification were
correct.

v0.6.1 therefore does not change the architecture. It performs a targeted
binding refinement from the v0.6 checkpoint.

Focus:

```text
heterogeneous_instruction
  >
homogeneous_computation
```

for CPU-like diverse-instruction prompts, while preserving the v0.6 semantic
geometry.

Important constraints:

```text
exact G05 prompt used for training : no
base encoder                       : frozen
concept / attribute heads          : frozen
adapter                            : trainable
7-axis hierarchy head              : trainable
generation                         : unchanged
```

New paraphrase supervision includes:

```text
CPU handles arithmetic and non-arithmetic instructions.
Diverse instructions are not equivalent to repeating one computation.
CPU instruction processing includes branch, compare, load, store and control.
Mixed instruction streams are CPU-like.
Repeated homogeneous computation is GPU-like.
```

The direct binding loss uses a pairwise margin:

```text
CPU-like:
P(heterogeneous_instruction)
  >=
P(homogeneous_computation) + 0.30

GPU-like:
P(homogeneous_computation)
  >=
P(heterogeneous_instruction) + 0.30
```

The experiment also preserves the existing v0.6 hierarchy predictions and
adapter representation on the previous semantic supervision bank.

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_encoder_adapter_v061.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-adapter-v061.pt
```

Logs:

```text
results/semantic_encoder_adapter_v061/train.log
results/semantic_encoder_adapter_v061/eval.log
```

Primary gate:

```text
G05 centroid -> CPU
G05 heterogeneous_instruction > homogeneous_computation
G08 Transformer retained
technical held-out accuracy does not regress
diverse / mixed instructions -> CPU-like
repeated homogeneous computation -> GPU-like
```

Only after this binding gate passes should the refined semantic adapter be
reconnected to a generation experiment.

### v0.9.1 Semantic Encoder Adapter v0.6.2: G05-Neighborhood Binding

v0.6.1 improved instruction semantics broadly and passed 10/11 gates, but the
exact G05 development prompt still had:

```text
heterogeneous_instruction
<
homogeneous_computation
```

even though G05 remained CPU-like overall.

v0.6.2 therefore performs a local lexical/semantic binding refinement around
G05 without using the exact G05 prompt.

Neighborhood expressions include:

```text
central/core device
diverse instructions
broad instruction set
varied instructions
different instruction types
general-purpose processor
control + branch + memory operations
```

Important constraints:

```text
exact G05 prompt used for training : no
base encoder                       : frozen
semantic adapter                   : frozen
concept / attribute heads          : frozen
7-axis hierarchy head              : trainable
generation                         : unchanged
```

Freezing the adapter is intentional. v0.6.1 already preserved good CPU
centroid geometry and G08 Transformer behavior; v0.6.2 changes only the mapping
from that semantic representation into the instruction hierarchy axes.

Direct neighborhood objective:

```text
CPU-neighborhood:
P(heterogeneous_instruction)
  >=
P(homogeneous_computation) + 0.20

GPU-neighborhood:
P(homogeneous_computation)
  >=
P(heterogeneous_instruction) + 0.20
```

The previous hierarchy behavior is preserved with an MSE replay loss over the
existing semantic supervision bank.

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_encoder_adapter_v062.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-adapter-v062.pt
```

Logs:

```text
results/semantic_encoder_adapter_v062/train.log
results/semantic_encoder_adapter_v062/eval.log
```

Primary gate:

```text
G05 centroid -> CPU
G05 heterogeneous_instruction > homogeneous_computation
G08 Transformer retained
technical held-out accuracy does not regress
central/diverse/broad/varied instruction paraphrases -> CPU-like
homogeneous computation probes -> GPU-like
```

Generation remains untouched until this local binding gate is evaluated.

### v0.9.1 Semantic Adapter v0.6.2 -> Original v0.9 Generation Integration

This experiment reconnects the completed Instruction Semantics adapter v0.6.2
to the original v0.9 additive generation integration.

The purpose is a controlled comparison: only the semantic representation is
updated, while the generation-side architecture and training policy remain the
same as the original v0.9 integration.

Semantic source:

```text
v0.8 base encoder
  -> Semantic Adapter v0.6.2
  -> adapted hidden        256
  -> concept probabilities   6
  -> attribute probabilities 4
  -> hierarchy probabilities 7
                            ---
                            273 dims
```

The hierarchy now contains:

```text
processor
general_purpose
control_oriented
throughput_oriented
data_parallel
heterogeneous_instruction
homogeneous_computation
```

Generation interface:

```text
273-d semantic feature
  -> zero-init Linear(273 -> 256)
  -> alpha = 0.1
  -> inject after Block 3
  -> Blocks 4-6
  -> FinalNorm
  -> LM Head
```

Controlled training policy:

```text
semantic encoder          : frozen
semantic adapter v0.6.2  : frozen
semantic heads            : frozen
hierarchy head            : frozen

generation Blocks 1-3     : frozen
generation Blocks 4-6     : trainable
FinalNorm                 : trainable
LM Head                   : frozen

Boundary data             : v1 only
projection LR             : 1e-3
block LR                  : 1e-5
alpha                     : 0.1
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_generation_v062.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-generation-v062.pt
```

Logs:

```text
results/semantic_generation_v062/train.log
results/semantic_generation_v062/eval.log
```

The evaluator uses the unchanged fixed 30-case benchmark and prints all seven
hierarchy signals for G05, G08, G09, and G28.

Primary questions:

```text
Does G05 generation finally become CPU-correct?
Does G05 keep heterogeneous_instruction > homogeneous_computation?
Does G08 remain Transformer-correct?
Does G09 CUDA remain correct?
Does the 30-case score meet or exceed the original v0.9 integration?
```

Reference:

```text
original v0.9 integration (semantic v0.5)
  semantic : 26/30
  strict   : 25/30

v0.8 balanced semantic tokens
  semantic : 25/30
  strict   : 25/30

v0.9 cross-attention
  semantic : 25/30
  strict   : 25/30
```

### v0.9.1 Semantic Encoder Adapter v0.7: Instruction-Computation Hierarchy

v0.6.x treated CPU-side heterogeneous instructions and GPU-side homogeneous
computation as opposing semantic axes. The next refinement makes the underlying
relationship explicit:

```text
computation is part of instruction execution
```

rather than treating `computation` and `instruction` as synonyms.

Hierarchy:

```text
processor
└─ instruction_execution
   ├─ computation
   │  ├─ arithmetic_logic
   │  └─ repeated_computation
   ├─ control_flow
   ├─ memory_operation
   ├─ data_movement
   └─ heterogeneous_instruction_stream

CPU-related characteristics:
  general_purpose
  control_oriented
  heterogeneous_instruction_stream

GPU-related characteristics:
  throughput_oriented
  data_parallel
  repeated_computation
```

The semantic intent is:

```text
CPU
  = executes mixed instruction streams containing
    computation + control + memory + data movement

GPU
  = also executes instructions, but is optimized for
    repeated / parallel computation over many data items
```

The experiment explicitly teaches examples such as:

```text
ADD is computation and an instruction.
Branch/jump is instruction execution but not arithmetic computation.
LOAD/STORE are instruction execution for memory operations.
Programs combine computation, control, memory and data-movement instructions.
GPU parallel computation is still instruction execution.
```

Relation constraints enforce child <= parent:

```text
computation <= instruction_execution
arithmetic_logic <= computation
repeated_computation <= computation
control_flow <= instruction_execution
memory_operation <= instruction_execution
data_movement <= instruction_execution
heterogeneous_instruction_stream <= instruction_execution
```

To preserve the semantic geometry already obtained in v0.6.2:

```text
base encoder        : frozen
semantic adapter    : frozen
concept/attr heads  : frozen
new hierarchy head  : trainable
generation          : unchanged
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_encoder_adapter_v07.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-adapter-v07.pt
```

Logs:

```text
results/semantic_encoder_adapter_v07/train.log
results/semantic_encoder_adapter_v07/eval.log
```

Primary gate:

```text
computation <= instruction_execution
arithmetic_logic <= computation
control/memory/data movement <= instruction_execution
G05 remains CPU and heterogeneous-instruction-stream
GPU repeated computation remains instruction execution
G08 Transformer remains correct
technical held-out centroid accuracy does not regress
```

Generation is not retrained in this step. The semantic ontology is validated
first.

### v0.9.1 Semantic Encoder Adapter v0.7.1: G05 Instruction-Computation Relation Refinement

v0.7 introduced the explicit ontology:

```text
computation ⊂ instruction_execution
```

and passed 11/12 gates. The only remaining failure was the exact G05 relation:

```text
G05:
instruction_execution < computation
```

even though G05 still mapped to CPU and retained a strong heterogeneous
instruction-stream signal.

v0.7.1 therefore performs a local relation refinement without changing the
ontology.

Important constraints:

```text
exact G05 prompt used for training : no
base encoder                       : frozen
semantic adapter                   : frozen
concept / attribute heads          : frozen
13-axis hierarchy head             : trainable
generation                         : unchanged
```

The new paraphrases focus on G05-neighbor expressions such as:

```text
central/core processor
diverse instructions
broad instruction set
varied instructions
mixed instruction stream
computation + control + memory + data movement
```

The direct relation objective is:

```text
P(instruction_execution)
  >=
P(computation) + 0.15
```

for these mixed-instruction neighborhood prompts.

The existing 13-axis hierarchy is preserved with replay over the previous
semantic supervision bank. The child-parent constraints are also retained.

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_encoder_adapter_v071.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-adapter-v071.pt
```

Logs:

```text
results/semantic_encoder_adapter_v071/train.log
results/semantic_encoder_adapter_v071/eval.log
```

Primary gate:

```text
G05 instruction_execution > computation
G05 centroid -> CPU
G05 heterogeneous_instruction_stream remains strong
G08 Transformer retained
GPU repeated-computation behavior retained
technical held-out centroid accuracy does not regress
```

Generation remains untouched until this final semantic relation gate is
evaluated.

### v0.9.1 Semantic Encoder Adapter v0.7.2: Program Composition Hierarchy

v0.7.1 fixed the G05 relation:

```text
instruction_execution > computation
```

but one remaining failure appeared at the program-composition level:

```text
program processing
  = computation + branch + load + store
```

was still represented too strongly as `computation`.

v0.7.2 therefore adds two explicit upper-level semantic axes:

```text
program_execution
instruction_sequence
```

and extends the hierarchy to:

```text
program_execution
└─ instruction_sequence
   └─ instruction_execution
      ├─ computation
      │  ├─ arithmetic_logic
      │  └─ repeated_computation
      ├─ control_flow
      ├─ memory_operation
      ├─ data_movement
      └─ heterogeneous_instruction_stream
```

The key semantic distinction is:

```text
program execution != computation only
```

A program is executed through an instruction sequence, and that sequence can
contain computation, control, memory operations, and data movement.

The previous semantic geometry is preserved:

```text
base encoder        : frozen
semantic adapter    : frozen
concept/attr heads  : frozen
15-axis hierarchy   : trainable
generation          : unchanged
```

Relation constraints include:

```text
instruction_sequence <= program_execution
instruction_execution <= instruction_sequence
computation <= instruction_execution
arithmetic_logic <= computation
repeated_computation <= computation
control_flow <= instruction_execution
memory_operation <= instruction_execution
data_movement <= instruction_execution
heterogeneous_instruction_stream <= instruction_sequence
```

Additional margin constraints keep both `program_execution` and
`instruction_sequence` broader than `computation`.

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_encoder_adapter_v072.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-adapter-v072.pt
```

Logs:

```text
results/semantic_encoder_adapter_v072/train.log
results/semantic_encoder_adapter_v072/eval.log
```

Primary gate:

```text
program_execution >= instruction_sequence
instruction_sequence >= instruction_execution
program_execution > computation
instruction_sequence > computation
computation/control/memory remain below instruction_execution
G05 remains CPU and instruction_execution > computation
G08 Transformer remains correct
technical held-out centroid accuracy does not regress
```

Generation remains untouched until this program-composition hierarchy is
validated.

### v0.9.1 Semantic Encoder Adapter v0.8: Hierarchy-Constrained Semantic Head

v0.7.2 showed that independent sigmoid outputs can learn the right concepts
while still violating the ontology:

```text
program_execution
>= instruction_sequence
>= instruction_execution
>= computation
>= repeated_computation
```

could be reversed on individual prompts.

v0.8 changes the head architecture instead of adding more paraphrases.

For every hierarchical child:

```text
P(child)
=
P(parent) * P(child | parent)
```

Therefore parent-child ordering is guaranteed by construction.

Examples:

```text
P(instruction_sequence)
  = P(program_execution)
    * P(instruction_sequence | program_execution)

P(instruction_execution)
  = P(instruction_sequence)
    * P(instruction_execution | instruction_sequence)

P(computation)
  = P(instruction_execution)
    * P(computation | instruction_execution)

P(repeated_computation)
  = P(computation)
    * P(repeated_computation | computation)
```

The constrained hierarchy is:

```text
processor
└─ program_execution
   └─ instruction_sequence
      ├─ instruction_execution
      │  ├─ computation
      │  │  ├─ arithmetic_logic
      │  │  └─ repeated_computation
      │  ├─ control_flow
      │  ├─ memory_operation
      │  └─ data_movement
      └─ heterogeneous_instruction_stream

processor also parents:
  general_purpose
  control_oriented
  throughput_oriented
  data_parallel
```

The semantic adapter and concept/attribute heads remain frozen:

```text
base encoder        : frozen
semantic adapter    : frozen
concept/attr heads  : frozen
constrained head    : trainable
generation          : unchanged
```

The head still returns logits, so existing code can continue to use:

```python
torch.sigmoid(hierarchy_head(hidden))
```

but the resulting marginals are already hierarchy constrained.

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_encoder_adapter_v08.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-adapter-v08.pt
```

Logs:

```text
results/semantic_encoder_adapter_v08/train.log
results/semantic_encoder_adapter_v08/eval.log
```

Primary gate:

```text
zero parent-child hierarchy violations
program >= sequence >= instruction >= computation
computation >= arithmetic_logic
computation >= repeated_computation
instruction >= control_flow / memory_operation / data_movement
G05 remains CPU
G08 remains Transformer
technical held-out centroid accuracy does not regress
```

Only semantic calibration can now fail; structural hierarchy ordering itself is
not learnable and therefore cannot be violated.

### v0.9.1 Semantic v0.8 -> Original v0.9 Generation Integration

This controlled experiment reconnects the completed hierarchy-constrained
semantic head to the original v0.9 additive generation interface.

Semantic representation:

```text
adapted semantic hidden  : 256
concept probabilities    :   6
attribute probabilities  :   4
constrained hierarchy    :  15
                         ----
total                    : 281
```

Generation interface:

```text
281
 -> Linear(281 -> 256), zero initialized
 -> alpha = 0.1
 -> inject after Block 3
 -> Blocks 4-6
 -> FinalNorm
 -> frozen LM Head
```

Controlled training policy:

```text
semantic encoder/head    : frozen
Blocks 1-3               : frozen
Blocks 4-6               : trainable
FinalNorm                 : trainable
LM Head                   : frozen
Boundary data             : v1 only
projection LR             : 1e-3
block LR                  : 1e-5
```

The purpose is to isolate the effect of the constrained semantic ontology.
The generation architecture is otherwise kept equivalent to the prior v0.9
additive integration experiment.

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_generation_v08.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-generation-v08.pt
```

Logs:

```text
results/semantic_generation_v08/train.log
results/semantic_generation_v08/eval.log
```

Primary targets:

```text
G05 generation -> CPU
G08 -> Transformer
G09 -> CUDA
G28 -> CPU
semantic >= 26/30
strict >= 26/30
```

The evaluator prints all 15 constrained hierarchy marginals for G05, G08,
G09 and G28.

### v0.9.1 Semantic Consistency Training v0.9

The v0.8 constrained semantic representation is correct, but additive
conditioning alone still allows generation hidden states to drift toward the
wrong semantic answer.

v0.9 adds an explicit consistency objective.

Teacher semantic target:

```text
concept probabilities     :  6
attribute probabilities   :  4
constrained hierarchy     : 15
                           ---
total                     : 25
```

The frozen v0.8 semantic path produces this 25-dimensional target.

Generation-side consistency head:

```text
generation prompt hidden 256
  -> LayerNorm
  -> Linear(256 -> 25)
  -> semantic prediction
```

Training objective:

```text
L_total
=
L_LM
+
0.35 * L_semantic_consistency
```

where `L_semantic_consistency` is BCE between the generation-side semantic
prediction and the frozen v0.8 semantic teacher target.

Architecture:

```text
v0.8 semantic teacher
  -> 281-d semantic condition
  -> additive projection 281 -> 256
  -> inject after Block 3
  -> Blocks 4-6
  -> FinalNorm
       |-> LM Head
       \-> Semantic Consistency Head 256 -> 25
```

Training policy:

```text
semantic teacher      : frozen
Blocks 1-3            : frozen
Blocks 4-6            : trainable
FinalNorm             : trainable
LM Head               : frozen
281 -> 256 projection : trainable
consistency head      : trainable
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_consistency_v09.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-consistency-v09.pt
```

Logs:

```text
results/semantic_consistency_v09/train.log
results/semantic_consistency_v09/eval.log
```

The evaluator reports both generation quality and semantic consistency for
G05, G08, G09 and G28, including teacher vs generation-side concept
probabilities.

Primary targets:

```text
G05 generation -> CPU
G05 generation hidden predicts CPU semantics
G08 -> Transformer
G09 -> CUDA
G28 -> CPU
semantic >= 26/30
strict >= 26/30
```

### v0.9.1 Semantic-Aware LM Head Adaptation v0.10

v0.9 Semantic Consistency showed that generation hidden states can recover the
correct semantic concept while the frozen LM head still emits a wrong token
sequence. v0.10 tests whether this is a hidden-to-vocabulary mismatch.

Controlled change from v0.9:

```text
LM Head:
  frozen
    ->
  trainable at 1e-6
```

All other main conditions remain the same:

```text
semantic teacher      : v0.8 constrained, frozen
Blocks 1-3            : frozen
Blocks 4-6            : trainable @ 1e-5
FinalNorm             : trainable @ 1e-5
281 -> 256 projection : trainable @ 1e-3
consistency head      : trainable @ 1e-3
LM Head               : trainable @ 1e-6
consistency weight    : 0.35
```

Loss:

```text
L_total
=
L_LM
+
0.35 * L_semantic_consistency
```

The model starts again from the same v0.8 base checkpoint rather than from the
already trained v0.9 consistency checkpoint, so the effect of LM-head unfreezing
can be compared cleanly.

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_aware_lm_head_v010.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-aware-lm-head-v010.pt
```

Logs:

```text
results/semantic_aware_lm_head_v010/train.log
results/semantic_aware_lm_head_v010/eval.log
```

Primary question:

```text
Does a very-low-LR LM-head adaptation convert
correct generation-hidden semantics into correct tokens?
```

Primary targets:

```text
G05 generated answer -> CPU
G05 hidden semantic -> CPU
G08 -> Transformer
G09 -> CUDA
G28 -> CPU
semantic > 26/30 if possible
strict > 26/30 if possible
```

### v0.9.1 v0.10.1: LM Head LR Sweep

v0.10 showed that LM-head learning rate `1e-6` produced almost the same
behavior as the frozen-head v0.9 reference. v0.10.1 therefore sweeps only the
LM-head learning rate while keeping all other conditions fixed.

Sweep:

```text
1e-7
3e-7
1e-6
3e-6
1e-5
```

Fixed conditions:

```text
semantic teacher      : v0.8 constrained, frozen
base model            : v0.8 pairwise-best
Blocks 1-3            : frozen
Blocks 4-6            : trainable @ 1e-5
FinalNorm             : trainable @ 1e-5
281 -> 256 projection : trainable @ 1e-3
consistency head      : trainable @ 1e-3
consistency weight    : 0.35
LM Head               : trainable @ sweep LR
```

Each LR starts from the same base checkpoint and writes an independent model:

```text
model/model-gpu-v0.9.1-semantic-aware-lm-head-v0101-1e-7.pt
model/model-gpu-v0.9.1-semantic-aware-lm-head-v0101-3e-7.pt
model/model-gpu-v0.9.1-semantic-aware-lm-head-v0101-1e-6.pt
model/model-gpu-v0.9.1-semantic-aware-lm-head-v0101-3e-6.pt
model/model-gpu-v0.9.1-semantic-aware-lm-head-v0101-1e-5.pt
```

The runner records, for every LR:

```text
best epoch
validation total loss
validation LM loss
validation semantic-consistency loss
30-case semantic score
30-case strict score
legacy score
G05 / G08 / G09 / G28 PASS/MISS
generated answers for those hard cases
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_aware_lm_head_sweep_v0101.py
```

Results:

```text
results/semantic_aware_lm_head_v0101_sweep/
  train-1e-7.log
  eval-1e-7.log
  train-3e-7.log
  eval-3e-7.log
  train-1e-6.log
  eval-1e-6.log
  train-3e-6.log
  eval-3e-6.log
  train-1e-5.log
  eval-1e-5.log
  summary.csv
```

The sweep intentionally does not choose a winner automatically. The decision
should consider validation loss, 30-case semantic/strict scores, and hard-case
behavior together.

### v0.9.1 v0.10.2: CPU Name-Meaning Binding

This experiment separates CPU lexical identity from functional semantics.

Functional semantics already learned:

```text
CPU
  -> general-purpose
  -> control-oriented
  -> heterogeneous instruction processing
```

v0.10.2 adds direct cross-lingual name binding:

```text
CPU
  <-> Central Processing Unit
  <-> 中央処理装置
  <-> 中央演算処理装置
```

and component-level correspondences:

```text
Central    <-> 中央 / 中心
Processing <-> 処理 / 演算処理
Unit       <-> 装置
```

The exact G05 development prompt is not used for training.

Architecture:

```text
v0.8 frozen semantic representation (256)
  -> LayerNorm
  -> Linear(256 -> 64)
  -> L2 normalization
  -> CPU name-binding space
```

Training pulls English/Japanese CPU names and paraphrases together while
separating them from GPU / Graphics Processing Unit distractors.

Run:

```powershell
git checkout v0.9.1
git pull

python run_cpu_name_binding_v0102.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-cpu-name-binding-v0102.pt
```

Logs:

```text
results/cpu_name_binding_v0102/train.log
results/cpu_name_binding_v0102/eval.log
```

Primary gate:

```text
Central Processing Unit ~= 中央処理装置
Central Processing Unit ~= 中央演算処理装置
Central / Processing / Unit Japanese mappings -> CPU
exact G05 holdout -> CPU name centroid
```

Generation is intentionally untouched in this experiment. If the name-binding
gate passes, the next controlled step is to inject this lexical CPU identity
signal into generation/logit alignment.

### v0.9.1 v0.11: Lexical Identity -> Generation Alignment

v0.10.2 showed that the CPU lexical identity is strongly represented:

```text
CPU
<-> Central Processing Unit
<-> 中央処理装置
<-> 中央演算処理装置
```

and the exact G05 holdout mapped toward the CPU name centroid.

v0.11 tests whether this lexical identity signal improves generation when
combined with the existing constrained semantic representation.

Condition vector:

```text
adapted semantic hidden   : 256
concept probabilities     :   6
attribute probabilities   :   4
constrained hierarchy     :  15
CPU lexical identity      :  64
                            ---
total                     : 345
```

Generation path:

```text
345
 -> Linear(345 -> 256), zero initialized
 -> alpha = 0.1
 -> inject after Block 3
 -> Blocks 4-6
 -> FinalNorm
 -> frozen LM Head
```

Controlled training policy:

```text
v0.8 semantic path      : frozen
v0.10.2 name binding    : frozen
Blocks 1-3              : frozen
Blocks 4-6              : trainable @ 1e-5
FinalNorm               : trainable @ 1e-5
LM Head                 : frozen
345 -> 256 projection   : trainable @ 1e-3
```

This intentionally removes the LM-head adaptation variable tested in v0.10
and v0.10.1. The only new information relative to the semantic-only additive
integration is the frozen 64-d lexical identity.

Run:

```powershell
git checkout v0.9.1
git pull

python run_lexical_generation_alignment_v011.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-lexical-generation-v011.pt
```

Logs:

```text
results/lexical_generation_v011/train.log
results/lexical_generation_v011/eval.log
```

The evaluator prints CPU/GPU lexical-identity similarities for:

```text
G04 CPU
G05 CPU hard case
G06 CPU
G08 Transformer
G09 CUDA
G27 GPU
G28 CPU
```

Primary question:

```text
Does adding the successful CPU name identity signal
change G05 generation from GPU-like wording to CPU?
```

Primary targets:

```text
G05 lexical identity -> CPU
G05 generated answer -> CPU
G04/G06 CPU retained
G08 Transformer retained
G09 CUDA retained
G27 GPU retained
G28 CPU retained
semantic >= 26/30 if possible
strict >= 26/30 if possible
```

### v0.9.1 v0.11.1: G05 Name-Request Evaluation

This is an evaluation-only experiment. No model parameters are changed.

It compares the original G05 wording:

```text
コンピュータの中心で多様な命令を処理する装置は何ですか。
```

with a minimally modified prompt that explicitly requests the entity name:

```text
コンピュータの中心で多様な命令を処理する装置の名前は何ですか。
```

Both prompts use exactly the same v0.11 checkpoint, semantic adapter,
CPU name-binding checkpoint, greedy decoding, and repetition penalty.

The evaluator reports:

```text
generated reply
CPU lexical similarity
GPU lexical similarity
CPU-GPU lexical margin
whether the answer explicitly contains:
  CPU
  Central Processing Unit
  中央処理装置
  中央演算処理装置
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_g05_name_request_v0111.py
```

Log:

```text
results/g05_name_request_v0111/eval.log
```

Interpretation:

```text
old G05 MISS + name-request G05 PASS
  -> answer-type / entity-request wording is a major factor

old G05 MISS + name-request G05 MISS
  -> lexical identity is present but generation still does not act on it
```

The fixed 30-case benchmark is intentionally left unchanged until this
controlled prompt comparison is measured.

### v0.9.1 v0.12: Semantic / Lexical -> Logit Alignment

v0.11.1 showed that changing G05 wording to explicitly ask for the device name
did not make generation emit CPU. The frozen semantic and lexical paths still
identify CPU, so v0.12 moves the intervention closer to the actual token
decision.

The existing v0.11 model is frozen. A new low-rank adapter maps the same
345-dimensional condition directly into vocabulary logits:

```text
v0.8 constrained semantic features  : 281
v0.10.2 lexical identity            :  64
                                      ---
condition                            : 345

345
 -> LayerNorm
 -> Linear(345 -> 64)
 -> GELU
 -> Linear(64 -> vocab)
 -> beta * direct logit bias
```

The final token decision is:

```text
frozen v0.11 LM logits
+
semantic / lexical direct logit bias
=
token logits
```

For this first controlled experiment the bias is applied only to the first
assistant token. This isolates entity/answer selection and avoids perturbing
the rest of the generated sentence.

Training policy:

```text
v0.11 generation model/projection : frozen
v0.8 semantic path                : frozen
v0.10.2 CPU name binding          : frozen
LM Head                           : frozen
direct logit adapter              : trainable
rank                              : 64
beta                              : 0.10
learning rate                     : 5e-4
application                       : first assistant token only
```

The adapter is trained against the actual frozen v0.11 first-token logits:

```text
combined_logits
=
frozen_v0.11_first_token_logits
+
direct_semantic_lexical_bias
```

so it learns a correction to the real model decision rather than an isolated
vocabulary classifier.

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_lexical_logit_alignment_v012.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-semantic-lexical-logit-v012.pt
```

Logs:

```text
results/semantic_lexical_logit_v012/train.log
results/semantic_lexical_logit_v012/eval.log
```

Evaluation includes the fixed 30-case benchmark plus both G05 wordings and
prints the top direct-bias vocabulary tokens.

Primary targets:

```text
Original G05 -> CPU name
Name-request G05 -> CPU name
G04/G06 CPU retained
G08 Transformer retained
G09 CUDA retained
G27 GPU retained
G28 CPU retained
semantic >= 26/30
strict >= 26/30
```
### v0.9.1 v0.12.1: Entity-Target Logit Alignment

v0.12 proved that a direct semantic/lexical logit path can change first-token
generation, but its supervision was still generic SFT first-token prediction.
v0.12.1 replaces that target with explicit technical entity names.

Target map:

```text
tech_cpu         -> CPU
tech_gpu         -> GPU
tech_llm         -> LLM
tech_transformer -> Transformer
tech_cuda        -> CUDA
tech_python      -> Python
```

The fixed 30-case evaluation prompts are not used as training rows. In
particular, the exact G05 prompt remains held out.

Training uses separate semantic-neighborhood prompts for the six concepts and
optimizes frozen v0.11 first-token logits plus entity-target direct logit bias
against the first tokenizer token of the explicit entity name.

Architecture:

```text
345-d semantic + lexical condition
 -> LayerNorm
 -> Linear(345 -> 64)
 -> GELU
 -> Linear(64 -> vocab)
 -> beta = 0.30
 -> first assistant token only
```

Frozen:

```text
v0.11 generation model/projection
v0.8 semantic path
v0.10.2 CPU name binding
LM Head
```

The evaluator additionally prints tokenizer decomposition for CPU, GPU, LLM,
Transformer, CUDA, and Python. For both G05 wordings it also reports each
entity first token's base logit, direct bias, base rank, and combined rank.

Run:

```powershell
git checkout v0.9.1
git pull

python run_entity_target_logit_alignment_v0121.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-entity-target-logit-v0121.pt
```

Logs:

```text
results/entity_target_logit_v0121/train.log
results/entity_target_logit_v0121/eval.log
```

Primary targets:

```text
Original G05 -> CPU
Name-request G05 -> CPU
CPU first-token rank improves strongly on G05
G08 -> Transformer
G09 -> CUDA
G10 -> Python
G27 -> GPU
G28 -> CPU
```
### v0.9.1 v0.12.2: Entity Contrastive Logit Alignment

v0.12.1 showed that explicit entity supervision improved CPU rank on G05,
but CPU still did not outrank GPU and the adapter over-fired on non-entity
questions.

v0.12.2 changes the objective from absolute entity promotion to relative
entity competition.

Core objective:

```text
correct entity logit
  >
every competing entity logit + margin
```

For CPU/GPU this explicitly includes:

```text
CPU prompt : logit(CPU) >= logit(GPU) + margin
GPU prompt : logit(GPU) >= logit(CPU) + margin
```

The loss combines:

```text
contrastive margin loss over the 6 entity first-token logits
+ 0.5 * entity classification loss
+ gate loss
+ small direct-bias L2 regularization
```

A learned entity gate suppresses the direct logit intervention on non-entity
questions.

```text
entity question     -> gate ~ 1
non-entity question -> gate ~ 0
```

Non-entity gate examples include debugging, comparison, topic-change, repeat,
fatigue, thanks, and short-answer requests. Exact fixed 30-case prompts remain
excluded from training, including G05.

Run:

```powershell
git checkout v0.9.1
git pull

python run_entity_contrastive_logit_alignment_v0122.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt
```

Logs:

```text
results/entity_contrastive_logit_v0122/train.log
results/entity_contrastive_logit_v0122/eval.log
```

The evaluator reports the gate value for all 30 benchmark cases and, for both
G05 variants, the CPU/GPU base logits, effective biases, final logits, ranks,
and CPU-GPU margins.

Primary targets:

```text
Original G05: CPU final logit > GPU final logit
Name-request G05: CPU final logit > GPU final logit
G05 generation -> CPU
non-entity gate low on error / compare / topic / repeat
G08 -> Transformer
G09 -> CUDA
G10 -> Python
G27 -> GPU
G28 -> CPU
```
### v0.9.1 v0.12.3: CPU-GPU Local Margin Refinement

v0.12.2 successfully moved G05 in the correct direction:

```text
Original G05 CPU-GPU final margin: -1.6826
Name-request G05 final margin    : -1.1942
```

but CPU still did not outrank GPU.

v0.12.3 therefore starts from the trained v0.12.2 checkpoint and performs a
local refinement only on CPU/GPU-neighborhood prompts.

Direct final-logit objective:

```text
CPU neighborhood:
  final_logit(CPU) >= final_logit(GPU) + 1.0

GPU neighborhood:
  final_logit(GPU) >= final_logit(CPU) + 1.0
```

The exact G05 benchmark prompts remain excluded from training.

Training policy:

```text
v0.12.2 adapter       : initialized from checkpoint, trainable
v0.12.2 entity gate   : frozen
v0.11 generation      : frozen
v0.8 semantic path    : frozen
v0.10.2 name binding  : frozen
learning rate         : 1e-4
target margin         : 1.0
classification weight : 0.25
preservation weight   : 0.20
```

To limit regression, replay prompts covering LLM, Transformer, CUDA, Python
and non-entity/control cases preserve the v0.12.2 effective direct-bias
outputs with an MSE preservation term.

Run:

```powershell
git checkout v0.9.1
git pull

python run_cpu_gpu_local_margin_v0123.py
```

Checkpoint:

```text
model/model-gpu-v0.9.1-cpu-gpu-local-margin-v0123.pt
```

Logs:

```text
results/cpu_gpu_local_margin_v0123/train.log
results/cpu_gpu_local_margin_v0123/eval.log
```

The evaluator compares v0.12.2 and v0.12.3 G05 final CPU-GPU margins directly
and reruns the full fixed 30-case benchmark.

Primary targets:

```text
Original G05: CPU-GPU final margin > 0
Name-request G05: CPU-GPU final margin > 0
G05 generation -> CPU
G27 -> GPU
G28 -> CPU
G08/G09/G10 retained
non-entity behavior retained
```
### Current Best Status after v0.12.3

Current best semantic/entity-generation checkpoint:

```text
model/model-gpu-v0.9.1-entity-contrastive-logit-v0122.pt
```

Status:

```text
v0.12.2 Entity Contrastive Logit Alignment
  -> CURRENT BEST

v0.12.3 CPU-GPU Local Margin Refinement
  -> FAILED EXPERIMENT
  -> retained for reproducibility and analysis
  -> not recommended as the default checkpoint
```

Reason:

v0.12.3 improved its local training/validation objective, but it generalized
in the wrong direction on the held-out G05 prompts. Compared with v0.12.2,
the CPU-GPU final margin became substantially worse and G28 regressed to a
GPU-first answer.

Observed G05 comparison:

```text
Original G05
  v0.12.2 margin : -1.6826
  v0.12.3 margin : -4.4354
  delta          : -2.7527

Name-request G05
  v0.12.2 margin : -1.1942
  v0.12.3 margin : -4.4082
  delta          : -3.2140
```

Therefore, subsequent experiments should start from v0.12.2 unless a later
experiment explicitly establishes a better held-out result.
### v0.12.2 Semantic Neighborhood Diagnostic

After v0.12.3 failed to generalize to G05, the next step is diagnostic only.
No training is performed and v0.12.2 remains the current-best checkpoint.

The diagnostic measures where both G05 variants sit in the frozen 345-d
semantic + lexical condition space relative to the CPU/GPU local prompts that
were used by the failed v0.12.3 experiment.

Measurements:

```text
cosine similarity
Euclidean distance
CPU centroid similarity/distance
GPU centroid similarity/distance
nearest CPU row
nearest GPU row
full CPU/GPU nearest-neighbor ordering
v0.12.2 entity gate value
v0.12.2 CPU-GPU final-logit margin for every reference row
```

The failed v0.12.3 checkpoint is not loaded. Its prompt set is used only as a
collection of reference points in the frozen v0.12.2 representation space.

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_neighborhood_diagnostic_v0122.py
```

Log:

```text
results/semantic_neighborhood_v0122/diagnostic.log
```

Interpretation target:

```text
If G05 is closer to GPU local rows than expected:
  -> the human-designed CPU neighborhood does not match the learned space.

If G05 is close to CPU local rows but their CPU-GPU logit behavior differs:
  -> the representation is locally similar but the output map is non-smooth
     or directionally inconsistent.
```
### v0.12.2 Local Logit Surface / Jacobian Diagnostic

This is a diagnostic-only experiment. v0.12.2 remains the current-best checkpoint.

Define the local CPU/GPU decision function in the frozen 345-d semantic + lexical
condition space:

```text
D(z) = final_logit(CPU) - final_logit(GPU)
```

For each G05 variant, the prompt-specific base LM logits are held fixed and the
diagnostic differentiates only through:

```text
gate(z) * direct_logit_adapter(z)
```

It computes:

```text
grad_z D(z)
gradient norm
directional derivative toward nearest CPU row
directional derivative toward CPU centroid
directional derivative toward nearest GPU row
directional derivative toward GPU centroid
```

It also performs finite-step probes at 1%, 5%, 10%, 25%, 50%, and 100% of each
direction vector, comparing the observed change in D(z) with the first-order
Jacobian prediction.

The failed v0.12.3 checkpoint is not loaded.

Run:

```powershell
git checkout v0.9.1
git pull

python run_local_logit_surface_diagnostic_v0122.py
```

Log:

```text
results/local_logit_surface_v0122/diagnostic.log
```

Primary interpretation:

```text
CPU direction raises D and GPU direction lowers D
  -> local output geometry is aligned; the remaining issue lies elsewhere.

CPU direction does not raise D
  -> semantic neighborhood and output-map gradient are locally inconsistent.

Jacobian sign and finite-step behavior disagree
  -> strong local nonlinearity / curvature in the output map.
```
### v0.12.2 Direct-Bias Gain Sweep

This is a no-training sensitivity experiment on the current-best v0.12.2 checkpoint.

Only the scale of the existing first-token direct semantic/entity correction is changed:

```text
final_logits = base_logits + gamma * gate * direct_bias
```

All model, semantic-adapter, name-binding, entity-adapter, and gate weights remain fixed.

Gain values:

```text
gamma = 0.50, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00
```

For every gamma, the evaluator runs the full fixed 30-case benchmark and both G05 wordings.

It reports:

```text
Semantic / Strict / Fluency / Legacy rates
Original G05 CPU-GPU final margin
Name-request G05 CPU-GPU final margin
CPU/GPU first-token ranks
whether the generated answer explicitly contains CPU
G08 Transformer
G09 CUDA
G10 Python
G27 GPU
G28 CPU
```

The purpose is to test whether v0.12.2 has the correct semantic correction direction but insufficient gain.

Run:

```powershell
git checkout v0.9.1
git pull

python run_direct_bias_gain_sweep_v0122.py
```

Log:

```text
results/direct_bias_gain_sweep_v0122/sweep.log
```

Primary interpretation:

```text
If a moderate gamma flips G05 to CPU while preserving the benchmark:
  -> v0.12.2 direction is sound and direct-bias gain is the main remaining issue.

If G05 requires a very large gamma and other cases regress first:
  -> simple global gain is insufficient; correction must become concept-specific.
```
### v0.12.2 G05 Top-K Logit Diagnostic

This is a no-training diagnostic on the current-best v0.12.2 checkpoint.

The previous gain sweep showed that CPU can outrank GPU at high gain while
still failing to become the generated first token. This diagnostic identifies
the actual vocabulary tokens that remain above CPU.

For both G05 variants and each gain:

```text
gamma = 1.00, 1.50, 2.00, 2.50, 3.00
```

the script prints the top-10 vocabulary entries with:

```text
rank
token id
decoded token string
base LM logit
raw direct bias
gamma-scaled effective bias
final logit
```

It also lists every top-k token still outranking CPU and decomposes its gap:

```text
final gap vs CPU
base-logit gap vs CPU
effective-bias gap vs CPU
```

This distinguishes whether the blocker is mainly inherited from the base LM
or is being reinforced by the semantic/entity direct-bias path.

Run:

```powershell
git checkout v0.9.1
git pull

python run_g05_topk_logit_diagnostic_v0122.py
```

Log:

```text
results/g05_topk_logits_v0122/diagnostic.log
```

Primary interpretation:

```text
CPU < competitor because base_gap dominates
  -> base LM prior is the main blocker.

CPU < competitor because bias_gap dominates
  -> direct semantic/entity adapter is promoting the wrong competitor.

CPU becomes rank 1 but generated reply still differs
  -> inspect token decoding / generation-step implementation.
```
### v0.12.4 Conditional CPU-Semantic Top-Competitor Suppression

This is a no-training safety sweep built on the current-best v0.12.2 checkpoint.

The frozen semantic head provides the CPU condition:

```text
P(tech_cpu)
```

Only the first-token logits of the G05 blockers identified by the Top-K
diagnostic are suppressed:

```text
多数の
特集
エ
学習
読み
```

The first-token formula is:

```text
final_logits
=
base_logits
+ entity_gate * v0.12.2_direct_bias
- lambda * P(tech_cpu) * blocker_mask
```

Therefore suppression becomes strong only when the frozen semantic model
classifies the prompt as CPU-like. Other concepts should receive little
suppression because their P(tech_cpu) is lower.

No weights are changed in this experiment. The sweep uses:

```text
lambda = 0, 2, 4, 6, 8, 10
direct gain = 1.0
```

For every lambda the script reruns the fixed 30-case benchmark plus both G05
variants and reports P(tech_cpu), CPU/GPU ranks, top-5 first-token logits,
generated replies, and G08/G09/G10/G27/G28 regression status.

Run:

```powershell
git checkout v0.9.1
git pull

python run_conditional_cpu_suppression_v0124.py
```

Log:

```text
results/conditional_cpu_suppression_v0124/sweep.log
```

Interpretation:

```text
If a moderate lambda makes G05 generate CPU while G08/G09/G10/G27/G28 remain stable:
  -> conditional suppression is a viable next architecture.

If non-CPU cases regress because P(tech_cpu) is not selective enough:
  -> replace raw P(tech_cpu) with a learned or margin-based CPU-specific gate.
```
### CPU/GPU Definition Correction v0.9.1

The current repository data now contains correct CPU definitions, but the existing
`model/model-gpu-v0.9-soft-intent.pt` checkpoint was trained before the latest
CPU-definition symmetry additions. This experiment retrains only the v0.9
soft-intent projection while keeping the v0.8 base model and intent head frozen.

New CPU-direct training paraphrases include:

```text
CPUとは
CPUとは何ですか。
CPUって何ですか。
CPUを簡単に説明して。
CPUの定義を教えて。
Central Processing Unitとは何ですか。
```

The corresponding answers consistently define CPU as a general-purpose / control-
oriented processor that executes diverse instructions. GPU remains associated with
large-scale parallel computation.

New files:

```text
audit_cpu_gpu_training_data_v091.py
evaluate_cpu_definition_projection_v091.py
run_cpu_definition_correction_v091.py
```

The runner performs:

```text
1. CPU/GPU training-data audit
2. projection-only v0.9 retraining
3. deterministic CPU/GPU direct-definition evaluation
```

The original checkpoint is preserved. The corrected experiment writes:

```text
model/model-gpu-v0.9.1-soft-intent-cpu-definition.pt
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_cpu_definition_correction_v091.py
```

Combined log:

```text
results/cpu_definition_correction_v091/run.log
```

Then test interactively with:

```powershell
python chat_v09.py --projection model/model-gpu-v0.9.1-soft-intent-cpu-definition.pt
```

Recommended first probe:

```text
CPUとは
```

If the direct CPU definition is corrected but G05 still fails, return to the
v0.12.2 semantic/entity path. If even the corrected projection still maps
`CPUとは` to the GPU definition, the remaining prior is inside the frozen v0.8
base model and a base/SFT retraining experiment is justified.
### G05 Re-evaluation after CPU Definition Correction

This no-training diagnostic re-evaluates G05 using the corrected v0.9.1
CPU-definition projection checkpoint under deterministic single-turn conditions.

Generation settings:

```text
temperature = 0
history turns = 0
```

Probes include:

```text
CPUとは
CPUとは何ですか
GPUとは
CPUとGPUの違いは
Original G05
Name-request G05
```

For each probe, the evaluator prints the generated reply, a simple CPU/GPU-like
classification, token count, and the top five frozen v0.8 intent probabilities.

Run:

```powershell
git checkout v0.9.1
git pull

python run_g05_after_cpu_definition_v091.py
```

Log:

```text
results/g05_after_cpu_definition_v091/evaluation.log
```

Interpretation:

```text
If direct CPU/GPU definitions are correct and G05 is also CPU-like:
  -> the earlier G05 failure was largely caused by CPU-definition training bias.

If direct CPU/GPU definitions are correct but G05 remains GPU-like:
  -> the residual problem lies in intent/semantic-to-generation routing for the G05 wording.
```
### v0.9.2 Implicit CPU Intent Correction

This experiment targets the remaining G05 failure after CPU-definition retraining.
The corrected v0.9.1 projection already answers direct CPU/GPU definition prompts correctly,
but the frozen v0.8 intent head still routes implicit CPU descriptions toward `tech_gpu`
or even `debug_error`.

Architecture:

```text
v0.8 base LM                 frozen
v0.8 intent head             fine-tuned
v0.9.1 CPU-definition projection  frozen
```

Training uses only curated implicit CPU/GPU paraphrases and protected replay.
The exact fixed G05 prompts are explicitly excluded.

Target conditions:

```text
implicit CPU rows: tech_cpu > tech_gpu
implicit CPU rows: tech_cpu > debug_error
GPU rows         : tech_gpu > tech_cpu
```

A small distillation term keeps the corrected head close to the original v0.8 intent head
outside the targeted boundary.

New checkpoint:

```text
model/model-gpu-v0.9.2-intent-head-implicit-cpu.pt
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_implicit_cpu_intent_v092.py
```

The runner trains only the intent head, then compares old vs new probabilities on:

```text
CPU direct
GPU direct
Original G05
Name-request G05
implicit CPU paraphrase
implicit GPU paraphrase
debug/error replay
```

It also generates replies through the frozen corrected v0.9.1 projection under
`temperature=0` and `history=0`.

Log:

```text
results/implicit_cpu_intent_v092/run.log
```
### v0.9.3 Intent-Projection Realignment

After v0.9.2, the corrected intent head changes the 24-dimensional intent
probability distribution. The older v0.9.1 projection was trained against
the original v0.8 intent-head distribution, so technical generation can become
misaligned even when intent probabilities move in the correct direction.

This experiment realigns the projection:

```text
v0.8 pairwise-best LM       frozen
v0.9.2 corrected Intent Head frozen
v0.9.3 Soft-Intent Projection trainable
```

The existing `train_sft_v09.py` is reused with the corrected v0.9.2 head.
The old projection is preserved.

New checkpoint:

```text
model/model-gpu-v0.9.3-soft-intent-realigned.pt
```

New files:

```text
evaluate_intent_projection_realignment_v093.py
run_intent_projection_realignment_v093.py
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_intent_projection_realignment_v093.py
```

The evaluation uses the same corrected v0.9.2 intent head with both:

```text
OLD: model/model-gpu-v0.9.1-soft-intent-cpu-definition.pt
NEW: model/model-gpu-v0.9.3-soft-intent-realigned.pt
```

and compares deterministic generation on:

```text
CPU direct
GPU direct
CPU/GPU contrast
Original G05
Name-request G05
implicit CPU paraphrase
implicit GPU paraphrase
debug/error replay
Transformer replay
Python replay
```

Log:

```text
results/intent_projection_realignment_v093/run.log
```

Interpretation:

```text
If CPU/GPU paraphrase generation recovers under the NEW projection:
  -> the v0.9.2 head / v0.9.1 projection mismatch was real.

If G05 intent remains GPU/error-dominant and generation remains wrong:
  -> further intent-head boundary work is still required after realignment.
```
### v0.9.4 Strong Intent -> Generation Coupling

v0.9.3 showed that simply retraining the same weak additive projection does not
change generation. v0.9.4 therefore changes the coupling architecture itself.

Architecture:

```text
v0.9.2 intent probabilities (24)
          |
          v
Linear 24 -> 64 -> GELU -> Linear 64 -> 512
          |
          +--> gamma (256)
          +--> beta  (256)
          |
          v
after Transformer Block 3:
x' = x * (1 + 0.5*tanh(gamma)) + 0.5*tanh(beta)
          |
          v
Blocks 4-6 -> final norm -> LM head
```

The v0.8 language model and v0.9.2 corrected intent head remain frozen.
Only the FiLM coupling is trainable. The final FiLM layer is zero-initialized,
so the initial model is an exact no-op relative to the frozen base model.

New checkpoint:

```text
model/model-gpu-v0.9.4-strong-intent-film.pt
```

New files:

```text
strong_intent_coupling_v094.py
train_strong_intent_coupling_v094.py
evaluate_strong_intent_coupling_v094.py
run_strong_intent_coupling_v094.py
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_strong_intent_coupling_v094.py
```

The evaluation includes the fixed 30-case benchmark and focused probes for:

```text
CPU / GPU direct definitions
CPU/GPU contrast
Original G05
Name-request G05
implicit CPU/GPU paraphrases
Transformer
Python
debug/error
```

Log:

```text
results/strong_intent_coupling_v094/run.log
```

Interpretation:

```text
If Transformer/Python/CPU/GPU generation improves while intent probabilities stay fixed:
  -> the primary bottleneck was weak intent-to-generation coupling.

If intent is correct but FiLM generation is still wrong:
  -> the frozen base LM/output prior is dominating even stronger hidden modulation.

If G05 intent remains GPU/error-dominant while other technical generation improves:
  -> coupling is improved, but G05 still requires further intent-boundary work.
```
### v0.9.5 Semantic Intent Bottleneck

v0.9.4 showed that stronger FiLM coupling can improve generation, but ambiguous
24-dimensional sigmoid intent probabilities remain a bottleneck. v0.9.5 adds
a supervised continuous semantic representation before generation coupling.

Architecture:

```text
prompt hidden (256)
      |
      v
LayerNorm -> Linear 256 -> 64 -> GELU
      |
      +--> semantic bottleneck z (64)
      |       |
      |       +--> learned semantic/intent tag head 64 -> 24
      |
      + frozen v0.9.2 intent probabilities (24)
      |
      v
concat: z64 + learned semantic24 + frozen intent24 = 112
      |
      v
MLP 112 -> 128 -> 512
      |
      +--> FiLM gamma(256)
      +--> FiLM beta(256)
      |
      v
inject after Transformer Block 3
```

The v0.8 LM and v0.9.2 intent head are frozen. Only the semantic bottleneck,
its 24-label supervision head, and the FiLM coupling are trainable.

The auxiliary semantic head is supervised by the intent/tag labels already
attached to the augmented SFT rows. This prevents the 64-dimensional bottleneck
from becoming an unconstrained LM-only latent vector.

New checkpoint:

```text
model/model-gpu-v0.9.5-semantic-intent-bottleneck.pt
```

New files:

```text
semantic_intent_bottleneck_v095.py
train_semantic_intent_bottleneck_v095.py
evaluate_semantic_intent_bottleneck_v095.py
run_semantic_intent_bottleneck_v095.py
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_intent_bottleneck_v095.py
```

The evaluator prints both the frozen v0.9.2 intent probabilities and the new
learned semantic-tag probabilities for each focused probe. This allows direct
inspection of whether the bottleneck separates CPU/GPU/Transformer/Python more
cleanly than the original intent head.

Log:

```text
results/semantic_intent_bottleneck_v095/run.log
```

Interpretation:

```text
If learned semantic tags correct G05/Transformer ambiguity and generation improves:
  -> intent-representation ambiguity was a primary bottleneck.

If learned semantic tags improve but generation does not:
  -> semantic representation is better, but hidden-to-output coupling remains weak.

If learned semantic tags remain ambiguous:
  -> the frozen prompt hidden representation itself is insufficient and deeper
     semantic encoder adaptation is required.
```
### v0.9.6 Semantic Encoder Adaptation

v0.9.5 showed that adding a supervised bottleneck after the frozen v0.8
prompt representation did not improve held-out semantic generalization.
v0.9.6 therefore adapts the late Transformer representation itself.

Architecture:

```text
Embedding + Blocks 1-3        frozen
             |
             v
Blocks 4-6                  trainable
             |
             v
FinalNorm                   trainable
             |\
             | +--> Semantic tag head (24)  trainable
             |
             +--> frozen LM head -> generation
```

The semantic tag head supervises the adapted prompt hidden state directly.
The LM head stays frozen so improvements must come from moving the hidden
representation into a better semantic/generation region rather than changing
the output vocabulary geometry.

Exact prompts from the fixed 30-case benchmark are removed from the augmented
training rows before the train/validation split. This includes exact G05.

Default optimization:

```text
Blocks 4-6 + FinalNorm LR : 3e-6
Semantic-head LR          : 3e-4
Semantic loss weight      : 0.20
Technical repeat          : 2
Replay repeat             : 2
```

New checkpoint:

```text
model/model-gpu-v0.9.6-semantic-encoder-adapted.pt
```

New files:

```text
semantic_encoder_adaptation_v096.py
train_semantic_encoder_adaptation_v096.py
evaluate_semantic_encoder_adaptation_v096.py
run_semantic_encoder_adaptation_v096.py
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_encoder_adaptation_v096.py
```

The evaluator tests the adapted language model directly, without the v0.9
intent projection/FiLM path. It reports the fixed 30-case benchmark plus:

```text
CPU direct
GPU direct
CPU/GPU contrast
Original G05
G05 wording without the question ending
Name-request G05
CPU/GPU paraphrases
Transformer
Python
Error
```

It also prints semantic-head probabilities from the adapted hidden state.

Log:

```text
results/semantic_encoder_adaptation_v096/run.log
```

Interpretation:

```text
If G05 semantic probability moves to tech_cpu and direct generation becomes CPU:
  -> the frozen v0.8 late representation was the main bottleneck.

If semantic probability improves but generation remains GPU-like:
  -> hidden semantics improved, but the frozen LM-head/base output prior still dominates.

If both remain GPU-like:
  -> Blocks 4-6 adaptation at this learning rate/supervision is insufficient;
     a stronger encoder adaptation or additional contrastive semantic objective is needed.
```
### v0.9.7 Output Alignment

v0.9.6 showed a key mismatch: some prompts became CPU-like in the adapted
semantic hidden representation, while the frozen LM head still generated
GPU-like answers. v0.9.7 aligns the output layer to the adapted hidden space.

Starting point:

```text
model/model-gpu-v0.9.6-semantic-encoder-adapted.pt
```

Trainable components:

```text
Blocks 1-3       frozen
Blocks 4-6       trainable   LR 1e-6
FinalNorm        trainable   LR 1e-6
Semantic head    trainable   LR 1e-4
LM head          trainable   LR 1e-6
```

The LM head is anchored to its v0.9.6 starting weights with a small MSE
penalty so output geometry can follow the adapted hidden representation
without freely drifting.

Exact fixed 30-case prompts are still removed from augmented training rows
when present. The experiment continues from v0.9.6 rather than retraining
from the v0.8 base checkpoint.

New checkpoint:

```text
model/model-gpu-v0.9.7-output-aligned.pt
```

New files:

```text
output_alignment_v097.py
train_output_alignment_v097.py
evaluate_output_alignment_v097.py
run_output_alignment_v097.py
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_output_alignment_v097.py
```

Primary diagnostic:

```text
G05 no-question
  semantic head: tech_cpu > tech_gpu
  desired generation: CPU / CPU-like
```

If this probe changes from GPU-like generation to CPU-like while semantic
probabilities stay CPU-dominant, the v0.9.6 hidden-to-output misalignment
hypothesis is supported.

Log:

```text
results/output_alignment_v097/run.log
```
### v0.9.8 Semantic Entity Logit Alignment

v0.9.7 improved semantic separation enough that Original G05 reached
`tech_cpu > tech_gpu`, while generation still produced the GPU answer.
v0.9.8 therefore adds an explicit causal objective from semantic technical
states to entity output-token logits.

Entities:

```text
tech_cpu         -> CPU
tech_gpu         -> GPU
tech_llm         -> LLM
tech_transformer -> Transformer
tech_cuda        -> CUDA
tech_python      -> Python
```

For rows with exactly one technical entity tag, the target entity first-token
logit is trained to exceed all competing entity logits by a margin.

For rows whose gold answer actually starts with that entity, an additional
top-blocker margin requires the target entity token to exceed the strongest
non-target vocabulary token. This blocker objective is deliberately restricted
so ordinary descriptive answers are not all forced to start with an entity name.

Training objective:

```text
Loss = LM loss
     + 0.10 * semantic loss
     + 0.20 * entity contrastive margin loss
     + 0.05 * top-blocker margin loss
     + LM-head anchor
```

Default margins:

```text
entity-vs-entity margin : 1.0
entity-vs-top-blocker   : 0.25
```

The experiment continues from:

```text
model/model-gpu-v0.9.7-output-aligned.pt
```

and saves:

```text
model/model-gpu-v0.9.8-entity-logit-aligned.pt
```

Exact fixed 30-case benchmark prompts remain excluded from training.

New files:

```text
entity_logit_alignment_v098.py
train_entity_logit_alignment_v098.py
evaluate_entity_logit_alignment_v098.py
run_entity_logit_alignment_v098.py
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_entity_logit_alignment_v098.py
```

The focused evaluator prints:

```text
Sem     : adapted semantic probabilities
Entity  : CPU/GPU/LLM/Transformer/CUDA/Python first-token logits
Top-1   : strongest token over the entire vocabulary
Reply   : deterministic greedy generation
```

The key diagnostic is G05:

```text
Desired causal chain:

semantic state -> tech_cpu
               -> logit(CPU) > logit(GPU)
               -> CPU reaches or approaches global top-1
               -> CPU-like generation
```

If entity ordering improves but global top-1 remains an unrelated blocker,
the next problem is full-vocabulary prior suppression rather than entity
confusion. If CPU becomes global top-1 but the continuation still collapses,
the remaining issue is post-first-token continuation rather than retrieval.
### v0.9.9 Dynamic Top-Competitor Alignment

v0.9.8 successfully improved semantic-state -> entity-logit coupling, but
the correct entity token could still lose to an unrelated global vocabulary
competitor such as `多数の`, `特集`, `ムスキー`, or malformed subword tokens.

v0.9.9 directly optimizes that competition for entity-answer rows only.

```text
adapted semantic hidden
        |
        v
correct entity first-token logit
        |
        +---- compare against max(all other vocabulary logits)
        |
        v
margin loss:
target_entity > dynamic_global_competitor + margin
```

The competitor is recomputed from the current full-vocabulary logits on every
forward pass. The loss is applied only when the gold answer actually begins
with a single technical entity (CPU/GPU/LLM/Transformer/CUDA/Python), so
ordinary explanatory answers are not forced to start with an entity token.

Default settings:

```text
source checkpoint       : model/model-gpu-v0.9.8-entity-logit-aligned.pt
Blocks 4-6 LR           : 3e-7
Semantic-head LR        : 3e-5
LM-head LR              : 3e-6
Dynamic top margin      : 0.75
Dynamic top weight      : 0.15
LM-head anchor weight   : 2e-4
```

v0.9.9 also guarantees that the validation split contains at least one
eligible entity-answer row for each available technical entity when possible.
The trainer prints train/validation eligible counts and validation top-loss,
addressing the v0.9.8 diagnostic where entity/blocker validation losses could
remain zero because no eligible validation samples were present.

New checkpoint:

```text
model/model-gpu-v0.9.9-dynamic-top-competitor.pt
```

New files:

```text
dynamic_top_competitor_v099.py
train_dynamic_top_competitor_v099.py
evaluate_dynamic_top_competitor_v099.py
run_dynamic_top_competitor_v099.py
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_dynamic_top_competitor_v099.py
```

The evaluator prints the top five global vocabulary logits for each focused
probe in addition to semantic and entity logits. The critical checks are:

```text
G05 no-question:
  CPU entity should remain above GPU
  and CPU should move toward / reach global top-1.

CPU paraphrase:
  CPU should remain far above GPU
  while unrelated blockers such as the former `ムスキー` token are suppressed.

Transformer:
  Transformer should remain above CUDA among entity logits
  while the malformed global blocker should lose rank.
```

Interpretation:

```text
If correct entity becomes global top-1 on held-out paraphrases:
  -> the remaining bottleneck was global vocabulary competition.

If training entity rows satisfy the margin but held-out probes do not:
  -> the output-alignment rule does not generalize; stronger semantic/entity
     conditioning or contrastive held-out training structure is needed.

If benchmark fluency/strict scores regress:
  -> the top-competitor constraint is too aggressive and should be weakened
     before extending it further.
```
### v0.10.0 Semantic-Gated Entity Decoder

v0.9.9 showed that semantic/entity discrimination can be correct while the
ordinary 8k-vocabulary first-token competition is still dominated by unrelated
high-prior tokens. v0.10.0 therefore changes the decoding path rather than
pushing vocabulary logits harder.

Architecture:

```text
v0.9.9 adapted prompt hidden (256)       frozen
                |
                v
      LayerNorm -> Linear 256 -> 64 -> GELU
                |
        +-------+--------+
        |                |
        v                v
 answer-mode gate    entity decoder
 entity / normal    CPU/GPU/LLM/
                    Transformer/CUDA/Python
        |
        v
if gate >= threshold:
  force ONLY the selected entity's first token
        |
        v
return immediately to ordinary frozen LM decoding
```

The v0.9.9 LM and semantic head are completely frozen. Only the new gate and
entity decoder are trained. This isolates whether semantic state can route
generation without further modifying the base generator.

Training targets:

```text
gate = 1:
  rows with exactly one technical entity tag whose gold answer starts with
  that entity token

gate = 0:
  all other rows

entity target:
  one of CPU / GPU / LLM / Transformer / CUDA / Python
```

Exact fixed 30-case benchmark prompts remain excluded from training.

New checkpoint:

```text
model/model-gpu-v0.10.0-semantic-gated-entity-decoder.pt
```

New files:

```text
semantic_gated_entity_decoder_v0100.py
train_semantic_gated_entity_decoder_v0100.py
evaluate_semantic_gated_entity_decoder_v0100.py
run_semantic_gated_entity_decoder_v0100.py
```

Run:

```powershell
git checkout v0.9.1
git pull

python run_semantic_gated_entity_decoder_v0100.py
```

The evaluator reports for each focused probe:

```text
semantic probabilities
entity-mode gate probability
entity candidate probabilities
forced entity tag (or None)
final generated reply
```

Key diagnostic cases:

```text
Original G05
G05 no-question
Name-request G05
CPU paraphrase
GPU paraphrase
Transformer
Python
Error
```

Interpretation:

```text
If G05-like prompts get gate≈1, entity=CPU, and the reply becomes CPU-like:
  -> the main remaining bottleneck was ordinary vocabulary competition.

If gate≈1 but the wrong entity is selected:
  -> semantic/entity generalization remains the bottleneck.

If the correct entity is selected but continuation is malformed:
  -> first-token routing works, but post-entity LM continuation needs alignment.

If non-entity prompts are incorrectly gated:
  -> answer-mode classification needs a better decision boundary.
```

### v0.10.2: Semantic Consistency Weight Sweep

The v0.10.1 LM-head LR sweep showed a clear plateau:

```text
LM Head LR    Semantic    Strict    G05    G08    G09    G28
1e-7          26/30       26/30     MISS   MISS   PASS   PASS
3e-7          26/30       26/30     MISS   MISS   PASS   PASS
1e-6          26/30       26/30     MISS   MISS   PASS   PASS
3e-6          26/30       26/30     MISS   MISS   PASS   PASS
1e-5          25/30       25/30     MISS   MISS   PASS   MISS
```

This indicates that changing only the LM-head learning rate does not correct
the remaining G05/G08 semantic failures. v0.10.2 therefore fixes the LM-head
LR at `3e-6`, the strongest tested value before benchmark regression, and
sweeps only the semantic-consistency loss weight.

Sweep:

```text
0.10
0.20
0.35
0.50
0.75
1.00
```

Fixed conditions:

```text
semantic teacher      : v0.8 constrained, frozen
base model            : v0.8 pairwise-best
Blocks 1-3            : frozen
Blocks 4-6            : trainable @ 1e-5
FinalNorm             : trainable @ 1e-5
281 -> 256 projection : trainable @ 1e-3
consistency head      : trainable @ 1e-3
LM Head               : trainable @ 3e-6
consistency weight    : sweep value
```

Each weight starts from the same base checkpoint and writes an independent
checkpoint under:

```text
model/model-gpu-v0.10.2-consistency-weight-*.pt
```

Run:

```powershell
git checkout v0.10.2
git pull

python run_semantic_consistency_weight_sweep_v0102.py
```

Results are written to:

```text
results/semantic_consistency_weight_v0102_sweep/summary.csv
```

Primary success criteria:

```text
1. Recover G05 and/or G08.
2. Preserve G09 and G28.
3. Keep semantic and strict scores at least 26/30.
4. Prefer a lower semantic-consistency validation loss without LM regression.
```

If higher consistency weight lowers semantic loss but G05/G08 remain MISS,
the next experiment should change the semantic representation or supervision
rather than further increasing LM-head learning rate.

### v0.10.3: Semantic-to-Output Alignment

v0.10.2 reached its best fixed-benchmark result at consistency weight `0.50`:

```text
Semantic-content : 27/30
Strict composite : 27/30
G05              : MISS
G08              : MISS
G09              : PASS
G28              : PASS
```

The important diagnostic was that G05 and G08 could already have the correct
semantic concept while generation still produced the wrong lexical output.
v0.10.3 therefore keeps the best v0.10.2 optimization settings and adds a
direct semantic-concept -> output-token alignment objective.

Concept/entity mapping:

```text
tech_gpu         -> GPU
tech_cpu         -> CPU
tech_llm         -> LLM
tech_transformer -> Transformer
tech_cuda        -> CUDA
tech_python      -> Python
```

For each training row, the frozen semantic teacher selects the highest
probability technical concept. When its confidence is at least `0.70`, the
first-answer-position logits are constrained so that the corresponding entity
token exceeds the strongest competing technical entity token by a margin.

Alignment loss:

```text
L_align = max(0, margin - logit(target_entity)
                       + max(logit(other_entities)))

L_total = L_LM
        + 0.50 * L_semantic_consistency
        + 0.10 * L_align
```

Default v0.10.3 settings:

```text
Consistency weight   : 0.50
LM Head LR           : 3e-6
Blocks 4-6 LR        : 1e-5
Projection LR        : 1e-3
Consistency-head LR  : 1e-3
Alignment weight     : 0.10
Alignment margin     : 0.75
Alignment confidence : 0.70
```

New files:

```text
train_semantic_output_alignment_v0103.py
evaluate_semantic_output_alignment_v0103.py
run_semantic_output_alignment_v0103.py
```

Run:

```powershell
git fetch origin
git checkout v0.10.3
git pull origin v0.10.3

python run_semantic_output_alignment_v0103.py
```

Output:

```text
model/model-gpu-v0.10.3-semantic-output-aligned.pt
results/semantic_output_alignment_v0103/train.log
results/semantic_output_alignment_v0103/evaluation.log
```

Primary success criteria:

```text
1. G05 should move from GPU-like generation toward CPU.
2. G08 should generate Transformer cleanly.
3. G09 and G28 should remain PASS.
4. Semantic/strict score should remain >= 27/30, preferably improve.
5. Alignment loss should decrease without destabilizing LM validation loss.
```

Interpretation:

```text
semantic correct + output corrected
    -> semantic-to-generation coupling was the bottleneck.

semantic correct + output still wrong
    -> first-token entity alignment is insufficient; continuation-level
       alignment or a learned semantic decoder is needed.

benchmark regression
    -> alignment weight/margin is too strong and should be swept.
```

### v0.10.4: Semantic-to-Global-Output Alignment

v0.10.3 showed that first-token semantic/entity alignment can improve lexical
selection, but it did not fully solve generation:

```text
v0.10.3 semantic/strict : 26/30
G05                     : semantic CPU, output still GPU-like
G08                     : first token repaired to "Transformer"
G28                     : semantic CPU, but direct answer started with GPU
```

The v0.10.4 experiment keeps the best v0.10.2 consistency setting and adds two
new objectives.

1. Global-vocabulary alignment

The target semantic entity must beat the strongest competing token in the
entire vocabulary at the first answer position:

```text
L_global =
  max(0,
      margin
      - logit(target_entity)
      + max(logit(all_other_vocab_tokens)))
```

2. Short continuation alignment

The first four answer tokens receive an additional LM loss. This tests whether
the semantic correction can persist beyond only the first entity token.

The total objective is:

```text
L_total =
    L_LM
  + 0.50 * L_semantic_consistency
  + 0.05 * L_entity_alignment
  + 0.05 * L_global_alignment
  + 0.05 * L_first4_continuation
```

Default settings:

```text
Consistency weight       : 0.50
LM Head LR               : 3e-6
Blocks 4-6 LR            : 1e-5
Projection LR            : 1e-3
Consistency-head LR      : 1e-3

Entity alignment weight  : 0.05
Entity alignment margin  : 0.75
Global alignment weight  : 0.05
Global alignment margin  : 0.50
Continuation weight      : 0.05
Continuation tokens      : 4
Alignment confidence     : 0.70
```

New files:

```text
train_semantic_global_output_alignment_v0104.py
evaluate_semantic_global_output_alignment_v0104.py
run_semantic_global_output_alignment_v0104.py
```

The evaluator also adds a direct-entity check for the benchmark cases where
the answer should begin with a concrete entity:

```text
G05 -> CPU
G08 -> Transformer
G09 -> CUDA
G10 -> Python
G27 -> GPU
G28 -> CPU
```

This prevents a response such as "GPUです...CPUは汎用処理..." from being
counted as fully correct for a question whose direct answer is CPU.

Run:

```powershell
git fetch origin
git checkout v0.10.4
git pull origin v0.10.4

python run_semantic_global_output_alignment_v0104.py
```

Outputs:

```text
model/model-gpu-v0.10.4-semantic-global-output-aligned.pt
results/semantic_global_output_alignment_v0104/train.log
results/semantic_global_output_alignment_v0104/evaluation.log
```

Primary success criteria:

```text
1. G05 begins with CPU and remains semantically CPU-like.
2. G08 begins cleanly with Transformer and improves continuation.
3. G09 remains CUDA-correct.
4. G28 begins with CPU, not GPU.
5. Corrected strict score reaches at least 27/30 without fluency regression.
6. Global alignment loss decreases without destabilizing validation LM loss.
```

### v0.10.5: Selective Semantic Output Alignment

v0.10.4 proved that global vocabulary alignment can repair direct entity
selection such as G28, but applying it to every row destabilized unrelated
conversation behavior.

Observed v0.10.4 result:

```text
Semantic-content       : 17/30
Strict composite       : 16/30
Direct entity          : 5/6
Corrected strict       : 16/30

G28                     : repaired to CPU
G08                     : direct Transformer preserved
G05                     : still GPU-like despite semantic CPU

Nontechnical regressions:
topic / repeat / end / compare rows were pulled toward technical entities.
```

v0.10.5 introduces an explicit supervised alignment gate based on the
augmentation row tags.

Alignment is active when a row contains at least one technical tag:

```text
tech_gpu
tech_cpu
tech_llm
tech_transformer
tech_cuda
tech_python
```

and does not contain a blocking control/nontechnical tag:

```text
control_short
control_topic
control_repeat
control_end
debug_error
research_compare
greeting
fatigue
thanks
capital
```

This prevents prompts such as "stop the GPU topic" from being forced toward
a GPU entity merely because the word GPU appears. For gated-off rows,
entity/global/continuation alignment losses are exactly zero. Normal LM loss
and semantic-consistency loss remain active.

Conceptually:

```text
prompt
  |
  +-- technical intent ----> LM + consistency
  |                          + entity alignment
  |                          + global alignment
  |                          + short continuation alignment
  |
  +-- nontechnical intent --> LM + consistency only
```

The selective objective is:

```text
technical row:
L_total =
    L_LM
  + 0.50 * L_semantic_consistency
  + 0.05 * L_entity_alignment
  + 0.02 * L_global_alignment
  + 0.02 * L_first4_continuation

nontechnical row:
L_total =
    L_LM
  + 0.50 * L_semantic_consistency
```

Default settings:

```text
Consistency weight       : 0.50
LM Head LR               : 3e-6
Blocks 4-6 LR            : 1e-5
Projection LR            : 1e-3
Consistency-head LR      : 1e-3

Entity alignment weight  : 0.05
Entity alignment margin  : 0.75
Global alignment weight  : 0.02
Global alignment margin  : 0.50
Continuation weight      : 0.02
Continuation tokens      : 4
Alignment confidence     : 0.70
```

New files:

```text
train_selective_semantic_output_alignment_v0105.py
evaluate_selective_semantic_output_alignment_v0105.py
run_selective_semantic_output_alignment_v0105.py
```

Run:

```powershell
git fetch origin
git checkout v0.10.5
git pull origin v0.10.5

python run_selective_semantic_output_alignment_v0105.py
```

Outputs:

```text
model/model-gpu-v0.10.5-selective-semantic-output-aligned.pt
results/selective_semantic_output_alignment_v0105/train.log
results/selective_semantic_output_alignment_v0105/evaluation.log
```

Primary success criteria:

```text
1. Recover nontechnical behavior lost in v0.10.4.
2. Preserve G28 direct CPU correction.
3. Preserve G08 direct Transformer correction.
4. Improve G05 toward direct CPU generation.
5. Direct-entity rate remains >= 5/6.
6. Corrected strict score recovers toward the v0.10.2 27/30 level.
7. No broad GPU/Transformer leakage into topic/repeat/end/compare prompts.
```

### v0.10.6: Selective Alignment + Boundary Replay

v0.10.5 restored most of the nontechnical behavior lost in v0.10.4:

```text
Semantic-content : 26/30
Strict composite : 26/30
Fluency          : 30/30
Direct entity    : 4/6
Corrected strict : 25/30
```

The remaining failures are concentrated in already-known boundary cases:
CPU reverse identification (G05), Transformer completion (G08), Python/CUDA
separation (G10), short-control behavior (G12), and CPU/GPU direct selection
(G28).

v0.10.6 deliberately keeps the v0.10.5 architecture and optimization settings
unchanged. The controlled change is training data only:

```text
Targeted Boundary v1       : ON
Targeted Boundary v2       : ON
Protected Boundary Replay  : ON
Balanced Control Replay    : OFF
```

The existing Boundary v2 data specifically reinforces:
- CPU reverse-identification from "central/diverse instruction" descriptions.
- Complete Transformer identification from Attention descriptions.
- short vs repeat control separation.
- end vs topic control separation.

Protected replay reinforces:
- Python vs CUDA category separation.
- CPU/GPU relation selection.
- topic/end capabilities that should not regress.

Selective alignment remains gated exactly as in v0.10.5:
technical-tag rows receive entity/global/continuation alignment unless blocked
by control/debug/research tags. Nontechnical rows receive LM + semantic
consistency only.

Unchanged optimization settings:

```text
Consistency weight       : 0.50
LM Head LR               : 3e-6
Blocks 4-6 LR            : 1e-5
Projection LR            : 1e-3
Consistency-head LR      : 1e-3

Entity alignment weight  : 0.05
Entity alignment margin  : 0.75
Global alignment weight  : 0.02
Global alignment margin  : 0.50
Continuation weight      : 0.02
Continuation tokens      : 4
Alignment confidence     : 0.70
```

New files:

```text
train_selective_boundary_replay_v0106.py
evaluate_selective_boundary_replay_v0106.py
run_selective_boundary_replay_v0106.py
```

Run:

```powershell
git fetch origin
git checkout v0.10.6
git pull origin v0.10.6

python run_selective_boundary_replay_v0106.py
```

Outputs:

```text
model/model-gpu-v0.10.6-selective-boundary-replay.pt
results/selective_boundary_replay_v0106/train.log
results/selective_boundary_replay_v0106/evaluation.log
```

Primary success criteria:

```text
1. G05 moves to direct CPU generation.
2. G08 keeps direct Transformer and improves semantic completion.
3. G10 stops mixing Python with CUDA.
4. G12 short-control behavior recovers.
5. G28 begins with CPU while preserving semantic correctness.
6. Nontechnical G13-G22 behavior remains recovered.
7. Corrected strict score reaches or exceeds 27/30.
```

### v0.10.7: GPU Template Ablation

v0.10.6 improved several boundary cases, but G05 still generated a memorized
GPU-like sentence even though the semantic prediction strongly favored CPU.

The suspected generation prior was the repeated training phrase:

```text
多数の計算を並列に処理する...
```

v0.10.7 is a controlled data ablation. Architecture, losses, learning rates,
selective gating, Targeted Boundary v2, and Protected Boundary Replay are kept
unchanged from v0.10.6. The only intended change is wording in the training
sources.

The token/word `多数` is removed from all three active training sources:

```text
data/conversation-ja.txt
data/instruction-ja.txt
augment_sft_v07.py
```

Representative rewrites:

```text
多数の計算を並列に処理する
  -> 同種の計算を並列に処理する

GPUは多数の計算を同時並行で処理する
  -> GPUは同種の計算を同時並行で処理する

多数の演算を同時に進める
  -> 多くの演算を同時に進める

多数の異なる命令を扱う
  -> さまざまな命令を扱う
```

The purpose is to test whether G05 is caused mainly by an overlearned lexical
GPU template rather than by semantic recognition itself.

New files:

```text
train_gpu_template_ablation_v0107.py
evaluate_gpu_template_ablation_v0107.py
run_gpu_template_ablation_v0107.py
```

Run:

```powershell
git fetch origin
git checkout v0.10.7
git pull origin v0.10.7

python run_gpu_template_ablation_v0107.py
```

Outputs:

```text
model/model-gpu-v0.10.7-gpu-template-ablation.pt
results/gpu_template_ablation_v0107/train.log
results/gpu_template_ablation_v0107/evaluation.log
```

Primary success criterion:

```text
If G05 changes from the memorized GPU-like answer toward CPU while the
semantic prediction remains CPU, this supports the hypothesis that the
failure was driven by lexical/template bias in the training corpus.
```

### v0.10.8: Base-Prior Ablation — Rebuild v0.8 from Pretraining

v0.10.7 removed `多数` from the current SFT/alignment sources, but G05 still
generated a GPU-like phrase containing `多数`. This indicates that the phrase
can survive in the frozen/reused v0.8 base checkpoint.

v0.10.8 therefore moves the ablation earlier in the training history and
rebuilds the v0.8 language model from pretraining.

Controlled pipeline:

```text
general-ja + data-nagato
conversation-ja
instruction-ja
        |
        | runtime lexical ablation:
        |   多数 -> 多く
        v
train_mixed_v08_clean_prior.py
        |
        v
model/model-gpu-v0.8-pretrain-clean.pt
        |
        | SFT sources already contain zero "多数"
        v
train_sft_v08.py
        |
        v
model/model-gpu-v0.8-chat-clean.pt
        |
        v
evaluate_generalization_v07.py
```

The architecture, tokenizer, pretraining sample count, curriculum mixture,
SFT hyperparameters, technical repeat, and replay repeat remain aligned with
the v0.8 experiment. Existing checkpoints are not overwritten.

The clean-pretraining script removes `多数` from *all* text sources after
loading them, including local `general-ja.txt` and `data-nagato.txt`.
This prevents the word from being reintroduced through the general corpus.

The v0.7 BPE tokenizer is intentionally kept fixed. A token may still exist in
the vocabulary, but the rebuilt v0.8 model receives no pretraining/SFT exposure
to the string `多数` in this experiment.

New files:

```text
train_mixed_v08_clean_prior.py
run_base_prior_ablation_v0108.py
```

Run the full controlled rebuild:

```powershell
git fetch origin
git checkout v0.10.8
git pull origin v0.10.8

python run_base_prior_ablation_v0108.py
```

Outputs:

```text
model/model-gpu-v0.8-pretrain-clean.pt
model/model-gpu-v0.8-chat-clean.pt
model/model-gpu-v0.8-intent-head-clean.pt

results/base_prior_ablation_v0108/pretrain.log
results/base_prior_ablation_v0108/sft.log
results/base_prior_ablation_v0108/evaluation.log
```

Primary question:

```text
Does G05 stop generating the memorized
"多数の計算を並列に処理する..."
GPU template when v0.8 itself is rebuilt without exposure to "多数"?
```

Interpretation:

```text
G05 improves toward CPU
  -> strong evidence that the old v0.8 lexical prior caused the failure.

G05 still emits 多数
  -> the source is deeper than the rebuilt v0.8 text exposure
     (for example tokenizer/generation dynamics or another data path).

G05 removes 多数 but remains GPU-like
  -> lexical memorization and semantic-to-generation binding are
     separate contributing factors.
```

### v0.10.9: Clean Intent Diagnostic

v0.10.8 rebuilt v0.8 from pretraining without exposure to the lexical cue
`多数`. The old phrase disappeared from G05, but the answer still selected
GPU semantics for a CPU reverse-identification prompt.

v0.10.9 performs no training. It diagnoses whether the rebuilt clean v0.8
intent representation already knows that G05 is CPU/general-purpose.

Inputs:

```text
model/model-gpu-v0.8-chat-clean.pt
model/model-gpu-v0.8-intent-head-clean.pt
```

The experiment has two parts:

```text
1. Full 30-case multi-label intent diagnosis
2. Focused G05 CPU/GPU probability diagnosis
```

The focused G05 report prints:

```text
tech_cpu
tech_gpu
CPU-GPU margin
property_general
property_parallel
general-parallel margin
generated answer
```

Interpretation:

```text
tech_cpu > tech_gpu
AND property_general > property_parallel
BUT generation is GPU-like
    -> representation is correct;
       representation-to-generation binding is the bottleneck.

tech_gpu >= tech_cpu
    -> reverse-identification is still a representation/training problem.
```

New files:

```text
diagnose_clean_g05_intent_v0109.py
run_clean_intent_diagnostic_v0109.py
```

Run:

```powershell
git fetch origin
git checkout v0.10.9
git pull origin v0.10.9

python run_clean_intent_diagnostic_v0109.py
```

Outputs:

```text
results/clean_intent_diagnostic_v0109/intent_full.log
results/clean_intent_diagnostic_v0109/g05_focus.log
```



### v0.11.0: Clean Semantic-to-Generation Binding

v0.10.9 showed that G05 is already represented as CPU/general-purpose internally, but generation still starts with GPU. v0.11.0 therefore changes only the representation-to-generation bridge.

Design:

```text
clean v0.8 hidden
   -> frozen clean intent head (24 probabilities)
   -> low-rank IntentEntityLogitBinding
   -> vocabulary logit bias
   -> applied ONLY at the first generated token
```

The v0.8 backbone and intent head remain frozen. The adapter is trained only on technical rows whose reference answer begins with the correct entity name. Exact prompts from the fixed 30-case benchmark are excluded.

Technical gate:

```text
exactly one technical-intent probability >= 0.50
    -> first-token binding ON
zero or multiple technical intents >= 0.50
    -> binding OFF, base generation unchanged
```

This keeps nontechnical generation isolated from the new binding path and avoids the broad-output regression seen in earlier global-alignment experiments.

New files:

```text
intent_entity_logit_binding_v0110.py
train_intent_entity_logit_binding_v0110.py
evaluate_intent_entity_logit_binding_v0110.py
run_intent_entity_logit_binding_v0110.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.0
git pull origin v0.11.0

python run_intent_entity_logit_binding_v0110.py
```

Primary criterion: G05 should start with `CPU` while nontechnical cases remain unchanged. Direct-entity cases G08/G09/G10/G27/G28 are secondary regression guards.
\n

### v0.11.1: CPU-GPU Relational Binding

v0.11.0 showed that the clean intent head can prefer CPU for G05 while generation still starts with GPU. v0.11.1 adds directional CPU-GPU relations instead of treating CPU and GPU only as independent attribute bundles.

Directional relation labels:

```text
cpu_controls_gpu
gpu_controlled_by_cpu
cpu_assigns_work_gpu
gpu_executes_for_cpu
```

Examples of new relation supervision:

```text
CPU -> controls / assigns work -> GPU
GPU -> is controlled by / executes work for -> CPU
```

The model learns these relations from paraphrased training rows. The exact fixed G05 benchmark prompt remains excluded.

Architecture:

```text
clean v0.8 hidden
   -> frozen clean intent head (24 dims)
   -> trainable directional relation head (4 dims)
   -> [intent probabilities + relation probabilities]
   -> low-rank vocabulary-logit binding
   -> first-token CPU/GPU correction
```

The clean v0.8 backbone and clean intent head remain frozen. Only the relation head and relational binding adapter are trainable.

New/updated files:

```text
augment_sft_v07.py
cpu_gpu_relational_binding_v0111.py
train_cpu_gpu_relational_binding_v0111.py
evaluate_cpu_gpu_relational_binding_v0111.py
run_cpu_gpu_relational_binding_v0111.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.1
git pull origin v0.11.1

python run_cpu_gpu_relational_binding_v0111.py
```

Primary success criterion:

```text
G05 should start with CPU, while G27 and G28 remain correct.
```

The evaluator also prints the four learned relation probabilities for every fixed benchmark case so the effect can be inspected directly.

### v0.11.2: CPU/GPU Role Binding

v0.11.1 showed that four directional relation labels were too fine-grained for the current model/data scale. The relation head overfit quickly and produced high CPU-GPU relation probabilities even on unrelated prompts.

v0.11.2 therefore compresses the relation structure into two roles:

```text
controller -> CPU
executor   -> GPU
```

Conceptual structure:

```text
CPU
  - controller
  - general-purpose processing
  - diverse instruction execution
  - system coordination

GPU
  - executor / accelerator
  - parallel computation
  - homogeneous workloads
  - receives work from CPU
```

Architecture:

```text
clean v0.8 hidden
   -> frozen clean intent head
   -> trainable 2-dim role head
        controller / executor
   -> [intent probabilities + role probabilities]
   -> low-rank first-token logit binding
```

Two changes distinguish v0.11.2 from v0.11.1:

1. Non-CPU/GPU negative rows train the role head toward `controller=0, executor=0`, preventing constant relation activation.
2. Output learning uses a direct CPU-vs-GPU margin rather than full-vocabulary cross entropy:

```text
CPU target: logit(CPU) > logit(GPU) + margin
GPU target: logit(GPU) > logit(CPU) + margin
```

The evaluator reports, for every fixed case:

```text
CPU intent probability
GPU intent probability
controller probability
executor probability
role-binding ON/OFF
CPU-GPU logit gap before binding
CPU-GPU bias gap
CPU-GPU logit gap after binding
```

The exact fixed benchmark prompts are excluded from training.

New files:

```text
cpu_gpu_role_binding_v0112.py
train_cpu_gpu_role_binding_v0112.py
evaluate_cpu_gpu_role_binding_v0112.py
run_cpu_gpu_role_binding_v0112.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.2
git pull origin v0.11.2

python run_cpu_gpu_role_binding_v0112.py
```

Primary success criterion:

```text
G05 -> CPU
G27 -> GPU
G28 -> CPU
```

while preserving the 30-case nontechnical regression set.

### v0.11.3: Direct CPU/GPU Logit Margin Binding

v0.11.2 established that G05 already has the correct relative semantic direction:

```text
CPU intent > GPU intent
controller > executor
```

but the base first-token LM prior still strongly favors GPU:

```text
G05 base CPU-GPU logit gap = -3.099
```

v0.11.3 therefore stops adding new semantic labels. It freezes the clean base model, clean intent head, and v0.11.2 role head, then learns only one signed CPU-vs-GPU logit-gap correction.

Architecture:

```text
frozen clean intent probabilities
       +
frozen controller/executor probabilities
       ↓
DirectGapBinding
       ↓
single signed delta

positive delta -> favors CPU
negative delta -> favors GPU

CPU logit += delta / 2
GPU logit -= delta / 2
```

This changes only the CPU/GPU first-token gap. No other vocabulary logits are modified.

Training objective:

```text
CPU target:
    CPU_logit - GPU_logit >= +1.0

GPU target:
    GPU_logit - CPU_logit >= +1.0
```

The inference gate is relative rather than absolute:

```text
CPU correction ON when:
    CPU intent > GPU intent
    CPU-GPU intent margin >= 0.10
    controller > executor
    controller-executor margin >= 0.05

GPU correction ON when:
    GPU intent > CPU intent
    GPU-CPU intent margin >= 0.10
    executor > controller
    executor-controller margin >= 0.05
```

This means G05 should be eligible even though its controller probability is below 0.55, while ambiguous comparison cases such as G27/G28 remain protected by the relative-margin gate.

The evaluator reports:

```text
CPU/GPU intent probabilities
controller/executor probabilities
direct-binding ON/OFF
base CPU-GPU logit gap
learned signed delta
final CPU-GPU logit gap
```

New files:

```text
cpu_gpu_direct_margin_binding_v0113.py
train_cpu_gpu_direct_margin_binding_v0113.py
evaluate_cpu_gpu_direct_margin_binding_v0113.py
run_cpu_gpu_direct_margin_binding_v0113.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.3
git pull origin v0.11.3

python run_cpu_gpu_direct_margin_binding_v0113.py
```

Primary criterion:

```text
G05 final CPU-GPU gap >= +1.0
G05 output starts with CPU
G27 remains GPU
G28 remains CPU
```

### v0.11.4: Direct CPU/GPU Margin Strength Sweep

v0.11.3 proved that the direct signed gap correction works in the correct direction. For G05:

```text
base CPU-GPU gap = -3.099
learned delta     = +2.694
final gap         = -0.405
```

The remaining issue is correction strength, not semantic direction.

v0.11.4 keeps the v0.11.3 architecture unchanged and sweeps only:

```text
max_delta
learning rate
target signed margin
```

Sweep configurations:

```text
d16_lr2e3_m15 : max_delta=16, lr=2e-3, margin=1.5
d20_lr2e3_m15 : max_delta=20, lr=2e-3, margin=1.5
d20_lr3e3_m20 : max_delta=20, lr=3e-3, margin=2.0
d24_lr3e3_m20 : max_delta=24, lr=3e-3, margin=2.0
```

The training script now accepts:

```text
--target-margin
```

Selection criterion:

```text
G05 -> CPU
G27 -> GPU
G28 -> CPU
```

If multiple configurations pass, the sweep recommends the checkpoint with the smallest G05 margin overshoot above +1.0.

Run:

```powershell
git fetch origin
git checkout v0.11.4
git pull origin v0.11.4

python run_cpu_gpu_direct_margin_sweep_v0114.py
```

Logs are written under:

```text
results/cpu_gpu_direct_margin_sweep_v0114/
```

### v0.11.5: First-Token Competitor Diagnostic

v0.11.4 showed that G05 can be moved from a CPU-GPU gap of -3.099 to nearly neutral, but the generated answer changed to `DPU...` rather than `CPU...`. This indicates that the real decision is not only CPU versus GPU; CPU must win against the full vocabulary.

v0.11.5 adds a diagnostic only. No training is performed.

It inspects G05 using the strongest near-neutral v0.11.4 checkpoint:

```text
model/model-gpu-v0.11.4-d16_lr2e3_m15.pt
```

The diagnostic prints:

```text
- CPU/GPU intent probabilities
- controller/executor probabilities
- whether direct binding is active
- signed binding delta
- tokenization of CPU, GPU, and DPU
- base first-token top 20 logits
- after-binding first-token top 20 logits
- exact rank/logit of CPU, GPU, and the first DPU token
- winner before and after binding
- CPU-versus-winner final logit gap
```

New files:

```text
diagnose_first_token_competitors_v0115.py
run_first_token_competitor_diagnostic_v0115.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.5
git pull origin v0.11.5

python run_first_token_competitor_diagnostic_v0115.py
```

The next correction should be based on the actual top competing token rather than increasing CPU-GPU correction strength blindly.

### v0.11.6: Hardest-Competitor Margin Binding

v0.11.5 showed that G05 is not a CPU-vs-GPU-only problem. After v0.11.4 correction, the first-token ranking became:

```text
1. D
2. G
3. CPU
```

Therefore v0.11.6 trains against the full vocabulary rather than only the CPU/GPU pair.

For each CPU/GPU training row, the correct first token is selected:

```text
CPU target -> token "CPU"
GPU target -> first token "G"
```

The adapter predicts a non-negative boost for that target only. The loss is:

```text
target_logit + boost >= max(all other vocabulary logits) + margin
```

with default margin 0.5.

Architecture:

```text
frozen clean intent probabilities
+
frozen controller/executor probabilities
        ↓
HardestCompetitorBoost
        ↓
non-negative target boost
        ↓
boost only the selected correct first token
```

No competing token is explicitly suppressed, and no unrelated vocabulary logit is modified.

New files:

```text
hardest_competitor_binding_v0116.py
train_hardest_competitor_binding_v0116.py
evaluate_hardest_competitor_binding_v0116.py
run_hardest_competitor_binding_v0116.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.6
git pull origin v0.11.6

python run_hardest_competitor_binding_v0116.py
```

Primary criterion:

```text
G05 -> CPU
G27 -> GPU
G28 -> CPU
```

The evaluator also prints the target boost, the base winner, the after-binding winner, and the target/winner logits for every activated case.

### v0.11.7: Safe Hardest-Competitor Binding

v0.11.6 successfully fixed G05 by boosting the correct CPU first token above the full vocabulary, but exposed two safety issues:

```text
1. The boost initialized at 6.0 because max_boost * sigmoid(0) = 12 * 0.5.
2. G09 (CUDA) incorrectly activated the GPU binding because tech_gpu was high even though tech_cuda was the stronger technical intent.
```

v0.11.7 keeps the full-vocabulary hardest-competitor objective and fixes both issues.

#### Near-zero boost initialization

The final layer is initialized so that:

```text
initial boost ~= 0.20
```

instead of 6.0.

For max_boost=12, the output bias is initialized to the inverse-sigmoid value corresponding to 0.20/12.

#### Technical top-intent safety gate

The six technical intent labels are ranked:

```text
tech_gpu
tech_cpu
tech_llm
tech_transformer
tech_cuda
tech_python
```

Binding is allowed only when the top technical intent is exactly the target CPU/GPU concept.

CPU correction requires:

```text
top technical intent == tech_cpu
CPU > GPU by at least 0.10
controller > executor by at least 0.05
```

GPU correction requires:

```text
top technical intent == tech_gpu
GPU > CPU by at least 0.10
executor > controller by at least 0.05
```

Therefore a CUDA prompt such as G09 should remain unmodified when `tech_cuda` is the strongest technical intent.

New files:

```text
safe_hardest_competitor_binding_v0117.py
train_safe_hardest_competitor_binding_v0117.py
evaluate_safe_hardest_competitor_binding_v0117.py
run_safe_hardest_competitor_binding_v0117.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.7
git pull origin v0.11.7

python run_safe_hardest_competitor_binding_v0117.py
```

Primary criteria:

```text
G05 -> CPU
G09 -> no GPU binding
G27 -> GPU
G28 -> CPU
initial boost ~= 0.20
```

### v0.11.8: Multi-Concept Safe Hardest-Competitor Binding

v0.11.7 fixed the CPU/GPU binding problem while preserving CUDA safety. v0.11.8 generalizes the same first-token hardest-competitor mechanism to six technical concepts:

```text
tech_gpu         -> GPU
tech_cpu         -> CPU
tech_llm         -> LLM
tech_transformer -> Transformer
tech_cuda        -> CUDA
tech_python      -> Python
```

The adapter is conditioned on:

```text
frozen intent probabilities
+
frozen controller/executor probabilities
+
target concept one-hot
```

and learns a non-negative boost for the selected concept's first token.

Training objective:

```text
target first-token logit + boost
    >= max(all other vocabulary logits) + 0.5
```

The boost still starts near 0.20.

#### Safety gate

CPU/GPU keep the v0.11.7 relative intent + role gate.

For LLM / Transformer / CUDA / Python, binding is allowed only when:

```text
1. that concept is the highest technical intent,
2. its probability is at least 0.70,
3. the canonical answer name is not already present in the prompt.
```

The third condition protects comparison/relation prompts such as:

```text
LLMとTransformerは同じ意味ですか。
```

from being forced to start with one of the compared entity names.

This conservative gate is intended to fix high-confidence missing-entity cases such as CUDA while avoiding new cross-concept regressions.

New files:

```text
multi_concept_safe_binding_v0118.py
train_multi_concept_safe_binding_v0118.py
evaluate_multi_concept_safe_binding_v0118.py
run_multi_concept_safe_binding_v0118.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.8
git pull origin v0.11.8

python run_multi_concept_safe_binding_v0118.py
```

Primary checks:

```text
G05 -> CPU
G09 -> CUDA if high-confidence tech_cuda
G27 -> GPU
G28 -> CPU
G30 comparison remains protected
```

### v0.11.9: Semantic Intent Repair / Calibration

v0.11.8 stabilized multi-concept generation binding, but the remaining failures were primarily technical-intent recognition errors:

```text
G07 LLM         : tech_llm confidence too low
G08 Transformer : tech_cuda ranked above tech_transformer
G10 Python      : tech_python correct but below the 0.70 gate
```

v0.11.9 leaves both the base language model and the v0.8 intent head frozen. It adds a small residual calibrator only over the six technical intent logits:

```text
tech_gpu
tech_cpu
tech_llm
tech_transformer
tech_cuda
tech_python
```

Architecture:

```text
frozen v0.8 intent logits
        ↓
extract six technical logits
        ↓
TechnicalIntentCalibrator
        ↓
six calibrated technical logits
        ↓
softmax ranking/confidence used only for the safety gate
```

The frozen v0.11.8 binding adapter still receives the original intent probabilities and role probabilities. Calibration therefore changes only concept selection and gate confidence, not the learned generation boost mapping.

Training uses 24 technical paraphrases with no exact overlap with the fixed 30-case benchmark. The loss combines six-way cross entropy with a small confidence-margin term targeting 0.75 probability for the correct technical concept.

Safety behavior remains:

```text
CPU/GPU:
  calibrated top concept must be CPU/GPU
  original CPU/GPU relative intent margin still applies
  controller/executor role margin still applies

LLM / Transformer / CUDA / Python:
  calibrated top probability >= 0.70
  canonical answer name must not already appear in the prompt
```

New files:

```text
technical_intent_calibrator_v0119.py
train_technical_intent_calibrator_v0119.py
evaluate_calibrated_multi_concept_binding_v0119.py
run_technical_intent_calibration_v0119.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.9
git pull origin v0.11.9

python run_technical_intent_calibration_v0119.py
```

Primary checks:

```text
G05 -> CPU remains correct
G07 -> LLM
G08 -> Transformer
G09 -> CUDA remains correct
G10 -> Python
G27 -> GPU
G28 -> CPU
G30 comparison remains protected
```

### v0.11.10: Selective Intent Repair

v0.11.9 showed that a free six-way calibrator can over-correct already-correct raw intents and can create false technical activations on nontechnical prompts.

v0.11.10 changes the strategy from global recalibration to selective repair.

Principles:

```text
1. Keep the raw technical top label by default.
2. Allow label replacement only when the raw top-vs-second margin is small.
3. Use the repair model to raise confidence for an already-correct raw winner.
4. Train an explicit technical-scope gate with nontechnical negative prompts.
5. Keep the frozen v0.11.8 generation binding unchanged.
```

The repair module contains:

```text
six raw technical logits
        ├─ residual six-way repair
        └─ binary technical-scope head
```

Default selective-repair conditions:

```text
raw top-second margin <= 0.20
repaired top differs from raw top
repaired top confidence >= 0.55
```

For LLM / Transformer / CUDA / Python binding:

```text
technical scope >= 0.70
chosen concept confidence >= 0.70
canonical name not already present in prompt
```

CPU/GPU continue to use the original pairwise intent and controller/executor role gates so G05/G27/G28 remain protected.

New files:

```text
selective_intent_repair_v01110.py
train_selective_intent_repair_v01110.py
evaluate_selective_intent_repair_v01110.py
run_selective_intent_repair_v01110.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.10
git pull origin v0.11.10

python run_selective_intent_repair_v01110.py
```

Primary checks:

```text
G05 -> CPU remains correct
G07 -> raw LLM is preserved and confidence may be lifted
G08 -> Transformer can be selectively repaired if ambiguity is small
G09 -> CUDA remains correct
G10 -> raw Python is preserved and confidence may be lifted
G21 -> technical scope should block false Transformer activation
G27 -> GPU
G28 -> CPU
G30 -> comparison remains protected
```

### v0.11.11: Selective Repair Threshold Sweep

v0.11.10 showed that the selective-repair architecture is safer than the global calibrator, but the remaining failures are sensitive to threshold choices rather than model structure.

v0.11.11 freezes all learned components and sweeps only three thresholds:

```text
repair_margin_threshold
  0.20 / 0.25 / 0.30

scope_threshold
  0.50 / 0.60 / 0.70

binding_confidence_threshold
  0.45 / 0.55 / 0.70
```

This gives 27 combinations.

No retraining is performed. The sweep uses:

```text
frozen clean v0.8 base model
frozen clean v0.8 intent head
frozen v0.11.2 role head
frozen v0.11.8 multi-concept binding
frozen v0.11.10 selective repair
```

Mandatory safety guards:

```text
G05 -> CPU
G09 -> CUDA
G27 -> GPU
G28 -> CPU
G21 -> binding OFF
G30 -> protected comparison, binding OFF
```

Among configurations that pass all guards, selection prefers:

```text
1. Number of repaired target cases among G07 / G08 / G10
2. Strict score
3. Semantic-content score
4. Tracked direct score
```

Outputs:

```text
results/selective_repair_threshold_sweep_v01111/threshold_sweep.csv
results/selective_repair_threshold_sweep_v01111/threshold_sweep_detail.csv
```

Run:

```powershell
git fetch origin
git checkout v0.11.11
git pull origin v0.11.11

python run_selective_repair_threshold_sweep_v01111.py
```

The script prints all 27 configurations and a final best configuration.

### v0.11.12: LLM-specific Scope Rescue

v0.11.11 established the best general thresholds:

```text
repair_margin_threshold      = 0.20
scope_threshold              = 0.50
binding_confidence_threshold = 0.45
```

These settings fixed G08 and G10 while preserving all mandatory safety guards, leaving G07 LLM as the main technical binding failure.

v0.11.12 keeps the v0.11.11 thresholds fixed and adds one narrow LLM-specific rescue rule.

The rescue activates only when:

```text
raw top          == tech_llm
repair top       == tech_llm
raw top-second margin <= 0.20
chosen confidence >= 0.70
canonical "LLM" not already present in the prompt
```

The raw-margin requirement is a safety constraint. It allows G07, whose LLM raw margin is small, while excluding unrelated prompts such as topic/compare cases whose raw LLM margin is much larger.

Normal technical scope remains unchanged for all other concepts.

New files:

```text
evaluate_llm_scope_rescue_v01112.py
run_llm_scope_rescue_v01112.py
```

No retraining is required.

Run:

```powershell
git fetch origin
git checkout v0.11.12
git pull origin v0.11.12

python run_llm_scope_rescue_v01112.py
```

Primary checks:

```text
G05 -> CPU
G07 -> LLM with rescue=YES
G08 -> Transformer
G09 -> CUDA
G10 -> Python
G21 -> binding OFF
G27 -> GPU
G28 -> CPU
G30 -> binding OFF / comparison protected
```

### v0.11.13: LLM Hardest-Competitor Boost Sweep

v0.11.12 successfully activated LLM rescue for G07, but the learned v0.11.8 boost (+4.963) was still not large enough to move the first token from the base winner to the LLM token.

v0.11.13 freezes all intent, scope, repair, and binding logic and sweeps only an extra multiplier on the LLM-rescue boost:

```text
1.00 / 1.25 / 1.50 / 2.00 / 2.50 / 3.00
```

The multiplier is applied only when the narrow v0.11.12 LLM rescue condition is true. All other concepts keep their original v0.11.8 boost unchanged.

Fixed thresholds:

```text
repair margin       = 0.20
scope threshold     = 0.50
binding confidence  = 0.45
LLM rescue margin   = 0.20
LLM rescue confidence = 0.70
```

Mandatory guards:

```text
G05 -> CPU
G09 -> CUDA
G10 -> Python
G21 -> binding OFF
G27 -> GPU
G28 -> CPU
G30 -> binding OFF
```

The sweep selects the smallest multiplier that makes G07 start with LLM while all guards still pass.

Run:

```powershell
git fetch origin
git checkout v0.11.13
git pull origin v0.11.13

python run_llm_hardest_competitor_boost_sweep_v01113.py
```

Output:

```text
results/llm_hardest_competitor_boost_sweep_v01113/boost_sweep.csv
```

### v0.11.14: Canonical Entity Prefix Binding

v0.11.13 showed that doubling the LLM rescue boost is enough to make the first token start with the LLM prefix, but the continuation still drifted:

```text
LLMARTです。
```

This means semantic selection and first-token binding were working, while canonical entity completion was not.

v0.11.14 extends binding from only the first token to the complete tokenized canonical entity prefix.

Canonical targets:

```text
tech_gpu         -> GPU
tech_cpu         -> CPU
tech_llm         -> LLM
tech_transformer -> Transformer
tech_cuda        -> CUDA
tech_python      -> Python
```

Behavior:

```text
step 0:
  use the existing learned v0.11.8 boost
  use x2.00 only for the narrow LLM rescue path

step 1..N while canonical prefix is incomplete:
  boost the expected canonical token only as much as necessary
  to exceed the current hardest vocabulary competitor by 0.50

after the canonical entity is complete:
  return immediately to normal LM generation
```

The continuation correction is therefore dynamic and minimal rather than a fixed forced-token value.

All v0.11.11 thresholds and v0.11.12 LLM rescue safety conditions remain fixed.

New files:

```text
evaluate_canonical_entity_prefix_binding_v01114.py
run_canonical_entity_prefix_binding_v01114.py
```

No retraining is required.

Run:

```powershell
git fetch origin
git checkout v0.11.14
git pull origin v0.11.14

python run_canonical_entity_prefix_binding_v01114.py
```

Primary target:

```text
G07 should begin with exactly "LLM" rather than "LLMART".
```

Safety checks:

```text
G05 -> CPU
G09 -> CUDA
G10 -> Python
G21 -> binding OFF
G27 -> GPU
G28 -> CPU
G30 -> comparison protected / binding OFF
```

### v0.11.15: Post-Entity Boundary Binding

v0.11.14 confirmed that canonical entity completion itself was already correct. For G07, the tokenizer produced:

```text
LLM -> ['LL', 'M']
```

and the second token `M` was already the strongest token without any extra prefix boost. The remaining failure was the token immediately after the completed entity:

```text
LLM + ART -> LLMART
```

v0.11.15 adds a one-step boundary rule immediately after a bound canonical entity is completed.

Boundary rule:

```text
for exactly one token after the entity:
  block vocabulary tokens whose decoded piece begins directly with
  ASCII alphanumeric or underscore

then:
  choose the highest remaining normal LM token

after that one boundary step:
  return to ordinary LM generation
```

This does not force a fixed suffix such as `です`. It only prevents direct ASCII continuation from merging with the canonical entity into a different identifier-like word.

The rule applies only when semantic binding is active. Existing scope, repair, prefix, and safety gates remain unchanged.

New files:

```text
evaluate_post_entity_boundary_binding_v01115.py
run_post_entity_boundary_binding_v01115.py
```

No retraining is required.

Run:

```powershell
git fetch origin
git checkout v0.11.15
git pull origin v0.11.15

python run_post_entity_boundary_binding_v01115.py
```

Primary target:

```text
G07: remove the "ART" continuation after LLM while preserving the exact LLM prefix.
```

Safety checks remain:

```text
G05 -> CPU
G09 -> CUDA
G10 -> Python
G21 -> binding OFF
G27 -> GPU
G28 -> CPU
G30 -> protected comparison / binding OFF
```

### v0.11.16: Semantic Continuation Repair

v0.11.15 completed the semantic-to-entity binding path and achieved 7/7 direct entity accuracy. The remaining technical failures were no longer entity-selection failures:

```text
G08 -> "Transformerです."       entity correct, Attention/content missing
G27 -> "GPUは同種の計算を蛍ります." entity correct, parallel-compute content broken
```

v0.11.16 therefore repairs the continuation after the **actual generated entity**, not the internal chosen semantic label. This matters because G08 and G27 already generate the correct entity even when the internal selected concept differs.

Semantic continuation anchors:

```text
GPU         -> は大量の並列計算を得意...
CPU         -> は汎用処理や制御を担当...
LLM         -> は文章を学習して生成する言語モデル...
Transformer -> はAttentionを中心に使うモデル構造...
CUDA        -> はNVIDIA GPUで汎用計算を行う技術...
Python      -> は読みやすい汎用プログラミング言語...
```

The repair is deliberately narrow:

```text
1. Generate a v0.11.15 baseline.
2. Detect whether the actual reply begins with a canonical technical entity.
3. If the baseline already passes semantic-content evaluation, do nothing.
4. If it is semantically incomplete, guide at most 8 continuation tokens.
5. For each guided token, add only the minimum boost required to beat the
   current hardest competitor by 0.35.
6. Return to normal greedy LM generation immediately after the guided prefix.
```

This is an experimental semantic-readout repair, not retraining.

New files:

```text
evaluate_semantic_continuation_repair_v01116.py
run_semantic_continuation_repair_v01116.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.16
git pull origin v0.11.16

python run_semantic_continuation_repair_v01116.py
```

Primary targets:

```text
G08 -> Transformer + Attention/content
G27 -> GPU + parallel-compute content
```

Safety target:

```text
G21/G30 must not activate continuation repair.
Already-correct v0.11.15 technical replies should remain unchanged.
```

### v0.11.17: Transformer Continuation Category Repair

v0.11.16 improved semantic-content and strict accuracy to 25/30 while preserving 7/7 direct entity accuracy and G21/G30 safety.

The remaining technical failure was G08:

```text
TransformerはAttentionを中心に使うための技術です。
```

The generated entity and Attention relation were correct, but the continuation drifted to the wrong category because v0.11.16 stopped guidance after 8 tokens, before the anchor reached `モデル構造`.

v0.11.17 keeps the entire v0.11.16 mechanism unchanged except for one narrow rule:

```text
if actual generated entity == Transformer
and baseline semantic-content == MISS:
    guide the complete Transformer continuation anchor
else:
    keep the v0.11.16 maximum of 8 guided tokens
```

Transformer anchor:

```text
はAttentionを中心に使うモデル構造
```

The same hardest-competitor margin remains:

```text
continuation margin = 0.35
```

No retraining is required.

New files:

```text
evaluate_transformer_continuation_category_repair_v01117.py
run_transformer_continuation_category_repair_v01117.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.17
git pull origin v0.11.17

python run_transformer_continuation_category_repair_v01117.py
```

Primary target:

```text
G08 -> Transformer + Attention + model/structure category
```

Expected preservation targets:

```text
G01 and G27 remain repaired by the v0.11.16 GPU continuation.
G05/G07/G09/G10/G28 remain unchanged.
G21/G30 continuation repair remains OFF.
```

### v0.11.18: Conversational Intent Repair

v0.11.17 reached 26/30 semantic-content and strict accuracy with 7/7 direct technical entities and preserved technical safety. The remaining failures were all conversational/control intents:

```text
G11 short
G13 topic
G15 repeat
G21 compare
```

v0.11.18 leaves the complete technical pipeline unchanged and adds a narrow conversational router.

The router does **not** match the benchmark prompts verbatim. It uses generic Japanese cues for four control intents:

```text
short   -> 一言 / 二言 / 短く / 簡潔 / 長い説明は不要
topic   -> 別件 / 別の話題 / 別のテーマ / 話題・テーマを変える
repeat  -> 理解できない / 分からない / もう一度 / 別の言い方 / 言い換え
compare -> 比べる / 比較 / 公平 / 差を検証
```

Repair activates only when:

```text
1. the full v0.11.17 baseline is semantic MISS, and
2. exactly one conversational route wins from the generic cues.
```

The routed response is generated through the same hardest-competitor guidance idea with margin 0.35.

Anchors:

```text
short   -> はい。簡潔に答えます。
topic   -> いいですよ。別の話題についてどうぞ。
repeat  -> もちろんです。分かりやすく説明します。
compare -> 同じ条件と評価指標をそろえて比較します。
```

No retraining is required.

New files:

```text
evaluate_conversational_intent_repair_v01118.py
run_conversational_intent_repair_v01118.py
```

Run:

```powershell
git fetch origin
git checkout v0.11.18
git pull origin v0.11.18

python run_conversational_intent_repair_v01118.py
```

Primary targets:

```text
G11 -> short response
G13 -> topic-change response
G15 -> repeat/re-explain response
G21 -> fair comparison response
```

Safety target:

```text
G05/G07/G08/G09/G10/G27/G28 technical behavior unchanged.
G30 protected LLM-vs-Transformer comparison must not enter conversational repair.
```

### v1.0.1: Held-out Generalization Test

v.1.0.0 freezes the 30-case benchmark-complete pipeline.

v.1.0.1 adds a completely new 60-case held-out paraphrase evaluation without modifying the model or repair logic.

Coverage:

```text
12 intents x 5 paraphrases = 60 cases

gpu / cpu / llm / transformer / cuda / python
short / topic / repeat / compare
error / end
```

The new prompts are different from the original 30-case benchmark.

Metrics:

```text
Baseline semantic accuracy
Final semantic accuracy
Per-intent accuracy
Conversational repair precision
Conversational repair recall
False activation rate
```

The evaluator runs both:

```text
v0.11.17 baseline
v0.11.18 final conversational-repair pipeline
```

so the contribution and safety of conversational repair can be measured directly.

New files:

```text
evaluate_heldout_generalization_v1001.py
run_heldout_generalization_v1001.py
```

Run:

```powershell
git fetch origin
git checkout v.1.0.1
git pull origin v.1.0.1

python run_heldout_generalization_v1001.py
```

Output:

```text
results/heldout_generalization_v1001/evaluation.log
```

### v1.0.2: Held-out Evaluator Fix

v1.0.1 exposed an evaluator bug: the conversational repair path still depended on the original 30-case `CASES` table. Because all held-out prompts were intentionally new, the repair path was never allowed to activate.

v1.0.2 removes that dependency from held-out evaluation.

New held-out trigger:

```text
1. Generate the v0.11.17 baseline.
2. Score the baseline with the held-out case's own required/forbidden groups.
3. If the held-out baseline is PASS, keep it unchanged.
4. If it is MISS, call the generic conversational cue router directly.
5. If exactly one route wins, generate the corresponding guided response.
6. Score the final reply using the same held-out requirements.
```

This allows repair precision, recall, and false-activation rate to be measured correctly on unseen prompts.

New files:

```text
evaluate_heldout_generalization_v1002.py
run_heldout_generalization_v1002.py
```

Run:

```powershell
git fetch origin
git checkout v.1.0.2
git pull origin v.1.0.2

python run_heldout_generalization_v1002.py
```

Output:

```text
results/heldout_generalization_v1002/evaluation.log
```

### v1.0.3: Technical Semantic Generalization Diagnostic

v1.0.2 showed that the remaining held-out bottleneck is mainly technical semantic generalization.

v1.0.3 does not change model behavior. It diagnoses the 30 held-out technical cases:

```text
gpu / cpu / llm / transformer / cuda / python
5 cases each = 30 technical cases
```

For each case it records:

```text
expected concept/entity
raw technical intent
raw margin
repaired intent
repair confidence
technical scope
chosen concept
chosen confidence
binding ON/OFF
actual generated entity
semantic PASS/MISS
```

Failures are classified into three stages:

```text
recognition:
    chosen concept != expected concept

binding:
    chosen concept == expected,
    but expected canonical entity is not generated

generation:
    expected entity is generated,
    but required semantic content is still missing
```

Outputs:

```text
results/technical_generalization_v1003/diagnostic.log
results/technical_generalization_v1003/diagnostic.csv
```

Run:

```powershell
git fetch origin
git checkout v.1.0.3
git pull origin v.1.0.3

python run_technical_generalization_diagnostic_v1003.py
```

The diagnostic should identify whether the next improvement should target semantic recognition, entity binding, or post-entity generation.

### v1.0.4: Repair Override Policy Sweep

v1.0.3 showed that 20/21 technical held-out failures were recognition failures, while binding/generation were largely healthy.

v1.0.4 isolates the **repair decision policy** from generation and compares multiple override rules using frozen prompt features.

Datasets:

```text
Held-out technical : 30 cases
Original technical guard : 10 cases
```

Policies:

```text
P0 current
P1 confidence-only thresholds
P2 scope + confidence thresholds
P3 disagreement + confidence thresholds
P4 concept-specific thresholds
```

Metrics:

```text
chosen-concept accuracy
override count
helpful overrides
wrong overrides
original technical guard accuracy
```

The sweep selects the best candidate that preserves all original technical selections when possible. No generation policy is changed yet; this branch is diagnostic only.

New files:

```text
repair_override_policy_sweep_v1004.py
run_repair_override_policy_sweep_v1004.py
```

Run:

```powershell
git fetch origin
git checkout v.1.0.4
git pull origin v.1.0.4

python run_repair_override_policy_sweep_v1004.py
```

Outputs:

```text
results/repair_override_policy_sweep_v1004/policy_sweep.log
results/repair_override_policy_sweep_v1004/policy_sweep.csv
```

The winning policy should be validated in real generation in the next step before replacing the current runtime decision rule.

### v1.0.5: CUDA-only Safe Override

v1.0.4 showed that a broad repair override improves held-out recognition but causes unacceptable regressions. The cleanest signal was CUDA: several held-out CUDA prompts were repaired correctly by the repair head while the runtime still trusted the raw GPU label.

v1.0.5 therefore introduces a narrow CUDA-only override for evaluation:

```text
repair_top == tech_cuda
raw_top != tech_cuda
scope >= 0.70
repair_conf >= 0.50
```

When active, generation is guided to:

```text
CUDAはNVIDIA GPUで汎用計算を行うための技術です。
```

The branch evaluates three guards:

```text
1. Original 30-case benchmark
2. Held-out technical 30
3. Held-out total 60, including the v1.0.2 conversational held-out repair
```

New files:

```text
evaluate_cuda_only_safe_override_v1005.py
run_cuda_only_safe_override_v1005.py
```

Run:

```powershell
git fetch origin
git checkout v.1.0.5
git pull origin v.1.0.5

python run_cuda_only_safe_override_v1005.py
```

Output:

```text
results/cuda_only_safe_override_v1005/evaluation.log
```

This branch is intentionally narrow: LLM repair is not overridden because v1.0.4 showed mixed correct and incorrect LLM repairs.

### v1.0.6: LLM Semantic Recognition Repair

v1.0.5 solved the held-out CUDA set with a narrow safe override. The next bottleneck was LLM recognition, where held-out accuracy remained 0/5 and broad repair overrides were unsafe.

v1.0.6 adds a separate lightweight **LLM semantic detector** trained on new paraphrases that are distinct from the five held-out LLM prompts.

Training data:

```text
30 LLM-positive prompts
30 non-LLM negative prompts

Negative coverage:
GPU / CPU / Transformer / CUDA / Python
+ conversational controls
```

Architecture:

```text
Frozen base model final hidden state (256)
        ↓
MLP 256 -> 32 -> 1
        ↓
LLM probability
```

The detector threshold is selected on a held-back validation split with a preference for zero false positives.

Runtime order:

```text
1. Existing v0.11.17 technical pipeline
2. CUDA-only safe override from v1.0.5
3. LLM detector
4. If LLM detector fires, guide:
   "LLMは文章を学習して生成する言語モデルです。"
5. Conversational repair remains a later fallback in integrated evaluation
```

CUDA decisions take precedence over the LLM detector.

New files:

```text
train_llm_semantic_detector_v1006.py
evaluate_llm_semantic_repair_v1006.py
run_llm_semantic_repair_v1006.py
```

Run:

```powershell
git fetch origin
git checkout v.1.0.6
git pull origin v.1.0.6

python run_llm_semantic_repair_v1006.py
```

The runner first trains:

```text
model/model-gpu-v1.0.6-llm-detector.pt
```

and then evaluates:

```text
Original integrated 30-case guard
Held-out technical 30
Held-out LLM 5
Held-out total 60
```

Output:

```text
results/llm_semantic_repair_v1006/evaluation.log
```

The v1.0.1 held-out LLM prompts are not used as detector training examples.

### v1.0.7: LLM Detector Hard-Negative Repair

v1.0.6 improved held-out LLM recognition from 0/5 to 2/5, but produced false positives on a capital question and a Python question.

v1.0.7 hardens the detector with additional non-LLM examples focused on the observed false-positive regions:

```text
capital / geography
Python / programming language
Transformer / model-structure prompts
model-comparison prompts
```

The five v1.0.1 held-out LLM prompts are still not used for gradient training.

Training now uses:

```text
30 LLM-positive prompts
existing non-LLM negatives
14 additional hard-negative training prompts
6 held-back hard-negative guard prompts
```

Threshold selection explicitly prefers:

```text
zero false positives on the combined validation-negative
+ hard-negative guard set
then maximum positive recall
```

New files:

```text
train_llm_semantic_detector_v1007.py
run_llm_hard_negative_repair_v1007.py
```

The existing v1.0.6 integrated evaluator is reused with the new checkpoint.

Run:

```powershell
git fetch origin
git checkout v.1.0.7
git pull origin v.1.0.7

python run_llm_hard_negative_repair_v1007.py
```

Checkpoint:

```text
model/model-gpu-v1.0.7-llm-detector-hard-negative.pt
```

Output:

```text
results/llm_hard_negative_repair_v1007/evaluation.log
```

Primary acceptance target:

```text
Original integrated benchmark = 30/30
Held-out LLM >= 2/5
No new LLM false-positive regression
CUDA held-out remains 5/5
```

### v1.0.8: LLM Threshold Safety Sweep

v1.0.7 restored the original integrated benchmark to 30/30 and kept held-out LLM at 2/5, but one held-out Python prompt still triggered a false LLM override at probability 0.587. One original repeat prompt also triggered an unnecessary LLM override at probability 0.576.

v1.0.8 keeps the v1.0.7 detector frozen and sweeps only the runtime threshold:

```text
0.50
0.55
0.58
0.60
0.62
0.65
0.70
```

Each threshold is evaluated on:

```text
Original integrated 30
Held-out technical 30
Held-out LLM 5
Held-out CUDA 5
Held-out total 60
Original false LLM overrides
Held-out false LLM overrides
```

Selection priority:

```text
1. Original = 30/30
2. Zero false LLM overrides
3. CUDA = 5/5
4. Highest held-out LLM accuracy
5. Highest held-out total accuracy
6. Higher threshold for extra safety
```

New files:

```text
llm_threshold_safety_sweep_v1008.py
run_llm_threshold_safety_sweep_v1008.py
```

Run:

```powershell
git fetch origin
git checkout v.1.0.8
git pull origin v.1.0.8

python run_llm_threshold_safety_sweep_v1008.py
```

Outputs:

```text
results/llm_threshold_safety_sweep_v1008/threshold_sweep.log
results/llm_threshold_safety_sweep_v1008/threshold_sweep.csv
```

The expected candidate from v1.0.7 observations is near 0.60, but v1.0.8 selects from measured results rather than assuming it.

### v1.0.9: Python Semantic Recognition Repair

v1.0.8 fixed the LLM detector threshold at 0.70 with zero observed false LLM overrides while preserving:

```text
Original integrated : 30/30
Held-out LLM        : 2/5
Held-out CUDA       : 5/5
```

The next weakest technical class was Python at 1/5.

v1.0.9 adds a separate Python semantic detector trained on new paraphrases that do not reuse the five held-out Python prompts.

Architecture:

```text
Frozen base hidden state (256)
        ↓
MLP 256 -> 32 -> 1
        ↓
Python probability
```

Training data covers Python-positive paraphrases and negatives from:

```text
LLM
CUDA
CPU / GPU
Transformer
other programming languages
generic programming questions
conversation / general QA
```

Runtime priority:

```text
1. Existing technical pipeline
2. CUDA-only safe override
3. LLM detector at fixed threshold 0.70
4. Python detector
5. Conversational fallback
```

CUDA and LLM decisions take precedence over Python.

New files:

```text
train_python_semantic_detector_v1009.py
evaluate_python_semantic_repair_v1009.py
run_python_semantic_repair_v1009.py
```

Run:

```powershell
git fetch origin
git checkout v.1.0.9
git pull origin v.1.0.9

python run_python_semantic_repair_v1009.py
```

Checkpoint:

```text
model/model-gpu-v1.0.9-python-detector.pt
```

Output:

```text
results/python_semantic_repair_v1009/evaluation.log
```

Primary acceptance target:

```text
Original integrated = 30/30
Held-out Python > 1/5
Held-out CUDA = 5/5
Python false positives = 0
```

### v1.0.10: CPU/GPU Hardware Semantic Disambiguation

v1.0.9 improved held-out Python from 1/5 to 3/5 while preserving the original integrated benchmark at 30/30 and CUDA at 5/5.

The remaining hardware confusion was concentrated in CPU and GPU. v1.0.10 adds a dedicated three-class semantic detector:

```text
CPU
GPU
OTHER
```

Using an explicit OTHER class prevents unrelated prompts from being forced into a CPU/GPU decision.

Architecture:

```text
Frozen base hidden state (256)
        ↓
MLP 256 -> 48 -> 3
        ↓
CPU / GPU / OTHER probabilities
```

Runtime priority:

```text
1. Existing technical pipeline
2. CUDA-only safe override
3. LLM detector at fixed threshold 0.70
4. Python detector
5. CPU/GPU hardware detector
6. Conversational fallback
```

New files:

```text
train_cpu_gpu_semantic_detector_v1010.py
evaluate_cpu_gpu_semantic_repair_v1010.py
run_cpu_gpu_semantic_repair_v1010.py
```

Run:

```powershell
git fetch origin
git checkout v.1.0.10
git pull origin v.1.0.10

python run_cpu_gpu_semantic_repair_v1010.py
```

Checkpoint:

```text
model/model-gpu-v1.0.10-cpu-gpu-detector.pt
```

Output:

```text
results/cpu_gpu_semantic_repair_v1010/evaluation.log
```

Primary acceptance target:

```text
Original integrated = 30/30
GPU > 2/5
CPU > 2/5
Python >= 3/5
CUDA = 5/5
Hardware false positives = 0
```

### v1.0.11: Hardware Threshold Safety Sweep

v1.0.10 improved held-out GPU from 2/5 to 4/5, but the CPU/GPU detector produced two held-out false-positive GPU overrides on LLM prompts.

v1.0.11 keeps the hardware detector frozen and sweeps only its runtime confidence threshold:

```text
0.75
0.85
0.90
0.92
0.93
0.94
0.95
```

Each threshold is evaluated on:

```text
Original integrated 30
Held-out technical 30
Held-out GPU 5
Held-out CPU 5
Held-out Python 5
Held-out CUDA 5
Held-out total 60
Original hardware false positives
Held-out hardware false positives
```

Selection priority:

```text
1. Original = 30/30
2. Zero hardware false positives
3. CUDA = 5/5
4. Python >= 3/5
5. Highest GPU accuracy
6. Highest CPU accuracy
7. Highest total accuracy
8. Higher threshold for extra safety
```

New files:

```text
hardware_threshold_safety_sweep_v1011.py
run_hardware_threshold_safety_sweep_v1011.py
```

Run:

```powershell
git fetch origin
git checkout v.1.0.11
git pull origin v.1.0.11

python run_hardware_threshold_safety_sweep_v1011.py
```

Outputs:

```text
results/hardware_threshold_safety_sweep_v1011/threshold_sweep.log
results/hardware_threshold_safety_sweep_v1011/threshold_sweep.csv
```

The expected safe region is around 0.93, because the observed false-positive GPU confidences were 0.899 and 0.915 while the useful held-out GPU overrides were 0.934 and 0.966. The sweep selects from measured results rather than assuming this threshold.

### v1.0.12: CPU Hard-Case Recognition Repair

v1.0.11 fixed the CPU/GPU hardware threshold at 0.93, preserving GPU 4/5 with zero observed hardware false positives. CPU remained at 2/5 because the remaining CPU failures were not threshold misses; their hardware labels were GPU or OTHER.

v1.0.12 adds a separate CPU semantic detector trained on new CPU paraphrases and hard negatives.

Architecture:

```text
Frozen base hidden state (256)
        ↓
MLP 256 -> 32 -> 1
        ↓
CPU probability
```

Runtime priority:

```text
1. Existing technical pipeline
2. CUDA-only safe override
3. LLM detector at 0.70
4. Python detector
5. CPU/GPU hardware detector at 0.93
6. CPU hard-case detector
7. Conversational fallback
```

The CPU detector only runs when the higher-priority CUDA / LLM / Python / hardware overrides are inactive.

New files:

```text
train_cpu_semantic_detector_v1012.py
evaluate_cpu_semantic_repair_v1012.py
run_cpu_semantic_repair_v1012.py
```

Run:

```powershell
git fetch origin
git checkout v.1.0.12
git pull origin v.1.0.12

python run_cpu_semantic_repair_v1012.py
```

Checkpoint:

```text
model/model-gpu-v1.0.12-cpu-detector.pt
```

Output:

```text
results/cpu_semantic_repair_v1012/evaluation.log
```

Primary acceptance target:

```text
Original integrated = 30/30
CPU > 2/5
GPU >= 4/5
Python >= 3/5
CUDA = 5/5
CPU false positives = 0
```

### v1.1.0: LLM Hard-Case Second-Stage Repair

v1.0.12 preserved the original integrated benchmark at 30/30 and improved held-out technical accuracy to 20/30. The weakest remaining technical class was LLM at 2/5.

v1.1.0 adds a second-stage LLM detector trained on indirect descriptions of language-model behavior. The original LLM detector remains fixed at threshold 0.70.

Architecture:

```text
Frozen base hidden state (256)
        ↓
MLP 256 -> 32 -> 1
        ↓
second-stage LLM probability
```

The second-stage detector is evaluated only after all higher-priority safe overrides remain inactive:

```text
1. Existing technical pipeline
2. CUDA-only safe override
3. Primary LLM detector at 0.70
4. Python detector
5. CPU/GPU hardware detector at 0.93
6. CPU hard-case detector
7. LLM second-stage detector
8. Conversational fallback
```

Training positives emphasize indirect LLM descriptions such as next-token prediction, text generation, context-based continuation, and natural-language response generation. Hard negatives include Transformer architecture, Python, CPU/GPU, CUDA, generic model questions, and conversational prompts.

New files:

```text
train_llm_second_stage_detector_v110.py
evaluate_llm_second_stage_repair_v110.py
run_llm_second_stage_repair_v110.py
```

Run:

```powershell
git fetch origin
git checkout v1.1.0
git pull origin v1.1.0

python run_llm_second_stage_repair_v110.py
```

Checkpoint:

```text
model/model-gpu-v1.1.0-llm-second-stage.pt
```

Output:

```text
results/llm_second_stage_repair_v110/evaluation.log
```

Primary acceptance target:

```text
Original integrated = 30/30
LLM > 2/5
GPU >= 4/5
CPU >= 3/5
Python >= 3/5
CUDA = 5/5
LLM second-stage false positives = 0
```

### v1.1.1: Transformer Semantic Recognition Repair

v1.1.0 improved held-out LLM from 2/5 to 4/5 while preserving the original integrated benchmark at 30/30. Transformer remained at 3/5.

v1.1.1 adds a dedicated Transformer semantic detector trained on new paraphrases describing Self-Attention, Multi-Head Attention, Encoder/Decoder blocks, sequence modeling, and Attention-based architectures.

Architecture:

```text
Frozen base hidden state (256)
        ↓
MLP 256 -> 32 -> 1
        ↓
Transformer probability
```

Runtime priority:

```text
1. Existing technical pipeline
2. CUDA-only safe override
3. Primary LLM detector at 0.70
4. Python detector
5. CPU/GPU hardware detector at 0.93
6. CPU hard-case detector
7. LLM second-stage detector
8. Transformer semantic detector
9. Conversational fallback
```

Transformer hard negatives include:

```text
LLM
generic neural-network/model questions
GPU / CPU / CUDA
Python
general conversation
```

New files:

```text
train_transformer_semantic_detector_v111.py
evaluate_transformer_semantic_repair_v111.py
run_transformer_semantic_repair_v111.py
```

Run:

```powershell
git fetch origin
git checkout v1.1.1
git pull origin v1.1.1

python run_transformer_semantic_repair_v111.py
```

Checkpoint:

```text
model/model-gpu-v1.1.1-transformer-detector.pt
```

Output:

```text
results/transformer_semantic_repair_v111/evaluation.log
```

Primary acceptance target:

```text
Original integrated = 30/30
Transformer > 3/5
LLM >= 4/5
GPU >= 4/5
CPU >= 3/5
Python >= 3/5
CUDA = 5/5
Transformer false positives = 0
```

### v1.1.2: Hybrid LLM-Transformer Relation Guard

v1.1.1 raised held-out Transformer accuracy to 5/5 and technical accuracy to 24/30, but the original G30 relation question ("LLMとTransformerは同じ意味ですか。") was incorrectly overwritten by the Transformer-only anchor, reducing original strict accuracy to 29/30.

v1.1.2 adds a semantic relation guard for prompts that mention both LLM and Transformer and ask about identity, difference, relation, or comparison.

Relation examples include terms such as:

```text
同じ
同一
違い
異なる
関係
関連
比較
同義
```

When the guard fires, the single-concept Transformer override is suppressed and the prompt is routed to a hybrid anchor:

```text
LLMとTransformerは同じ意味ではありません。
TransformerはLLMで利用されるモデル構造の一つです。
```

No model retraining is required. All existing detector checkpoints and thresholds remain unchanged.

Runtime priority:

```text
1. LLM-Transformer relation guard
2. CUDA-only safe override
3. Primary LLM detector
4. Python detector
5. CPU/GPU hardware detector
6. CPU hard-case detector
7. LLM second-stage detector
8. Transformer semantic detector
9. Conversational fallback
```

New files:

```text
evaluate_llm_transformer_relation_guard_v112.py
run_llm_transformer_relation_guard_v112.py
```

Run:

```powershell
git fetch origin
git checkout v1.1.2
git pull origin v1.1.2

python run_llm_transformer_relation_guard_v112.py
```

Output:

```text
results/llm_transformer_relation_guard_v112/evaluation.log
```

Primary acceptance target:

```text
Original semantic = 30/30
Original strict = 30/30
Held-out Transformer = 5/5
Held-out LLM >= 4/5
Held-out technical >= 24/30
Held-out total >= 45/60
```

### v1.1.3: Deterministic Relation Response

v1.1.2 restored original strict accuracy to 30/30 with the LLM-Transformer relation guard, but the guided decoder corrupted part of the hybrid relation sentence even though semantic scoring passed.

v1.1.3 removes generative decoding from this deterministic relation path.

Before:

```text
relation guard
    -> generate_guided_response(...)
    -> possible decoding corruption
```

Now:

```text
relation guard
    -> return RELATION_ANCHOR directly
```

The fixed relation response is:

```text
LLMとTransformerは同じ意味ではありません。
TransformerはLLMで利用されるモデル構造の一つです。
```

No checkpoint or threshold is changed. This isolates a generation-fidelity issue from semantic routing and prevents the model from corrupting an already-determined factual relation response.

New files:

```text
evaluate_deterministic_relation_response_v113.py
run_deterministic_relation_response_v113.py
```

Run:

```powershell
git fetch origin
git checkout v1.1.3
git pull origin v1.1.3

python run_deterministic_relation_response_v113.py
```

Output:

```text
results/deterministic_relation_response_v113/evaluation.log
```

Primary acceptance target:

```text
G30 relation text is exact and uncorrupted
Original semantic = 30/30
Original strict = 30/30
Held-out Transformer = 5/5
Held-out technical >= 24/30
Held-out total >= 45/60
```

### v1.1.4: Fresh Generalization Test v2

v1.1.3 is frozen for this experiment. No detector retraining, no threshold tuning, and no new repair policy is introduced.

A new evaluation-only set of 120 prompts is added:

```text
12 intents x 10 prompts

GPU
CPU
LLM
Transformer
CUDA
Python
short
topic
repeat
compare
error
end
```

The purpose is to measure whether the v1.1.3 improvements generalize to previously unused prompts instead of continuing to tune against the original 60 held-out cases.

New files:

```text
fresh_generalization_cases_v114.py
evaluate_fresh_generalization_v114.py
run_fresh_generalization_v114.py
```

Run:

```powershell
git fetch origin
git checkout v1.1.4
git pull origin v1.1.4

python run_fresh_generalization_v114.py
```

Outputs:

```text
results/fresh_generalization_v114/fresh_v2.log
results/fresh_generalization_v114/fresh_v2.csv
```

The CSV records each fresh prompt, final response, pass/fail result, missing semantic requirements, conversational route, and which semantic overrides fired.

Important evaluation rule:

```text
Do not train or tune on these 120 prompts before recording the first result.
```

The first run is the clean generalization measurement for the frozen v1.1.3 system.

### v1.1.5: Unified Semantic Router Diagnostic

Fresh Generalization Test v2 measured 65.0% total accuracy and 55.0% technical accuracy on 120 previously unused prompts. From this point, the Fresh v2 set is treated as a development set rather than a pristine test set.

v1.1.5 does not change generation behavior. It collects all semantic detector outputs for the 60 technical Fresh v2 prompts and compares them simultaneously.

Diagnostic candidate scores:

```text
GPU         = hardware GPU probability
CPU         = max(hardware CPU probability, CPU hard-case probability)
LLM         = max(primary LLM probability, second-stage LLM probability)
Transformer = Transformer detector probability
CUDA        = CUDA repair probability x repair scope
Python      = Python detector probability
```

For every technical prompt the diagnostic records:

```text
gold class
top-1 class / score
top-2 class / score
top-1 margin
gold-class score
all six candidate scores
primary + second-stage LLM scores
CPU binary score
hardware CPU/GPU/OTHER probabilities
CUDA repair probability
repair scope
```

It also prints a six-class confusion matrix and per-class top-1 rates.

Important: these detector heads were trained independently, so their raw outputs are not calibrated probabilities across classes. Raw argmax is therefore diagnostic only and is not deployed as a new routing policy.

New files:

```text
unified_semantic_router_diagnostic_v115.py
run_unified_semantic_router_diagnostic_v115.py
```

Run:

```powershell
git fetch origin
git checkout v1.1.5
git pull origin v1.1.5

python run_unified_semantic_router_diagnostic_v115.py
```

Outputs:

```text
results/unified_semantic_router_diagnostic_v115/diagnostic.log
results/unified_semantic_router_diagnostic_v115/router_scores.csv
results/unified_semantic_router_diagnostic_v115/confusion.csv
```

The next decision should be based on whether score calibration can separate the correct concept from competing detector heads, or whether a single learned six-class semantic router should replace the serial override chain.

### v1.2.0: Unified Hidden-State Semantic Router

v1.1.5 showed that raw detector-score argmax reaches only 51.7% on the Fresh-v2 technical development set, while score calibration improves to about 60%. This suggests that detector outputs contain useful information but are not sufficiently comparable across independently trained heads.

v1.2.0 tests a cleaner architecture:

```text
Prompt
  ↓
Frozen base LLM
  ↓
last hidden state (256)
  ↓
Unified 6-class semantic router
  ↓
GPU / CPU / LLM / Transformer / CUDA / Python
```

No serial detector priority is used in this experiment.

Two router architectures are evaluated:

```text
Linear : 256 -> 6

MLP:
256 -> 64 -> GELU -> Dropout(0.10) -> 6
```

Evaluation uses the Fresh-v2 technical 60 prompts as a development set with 5-fold stratified cross-validation. Each fold contains two examples per class. Three random seeds are ensembled inside each fold.

Important:

```text
Fresh-v2 is no longer treated as a pristine final test set.
The v1.2.0 result is cross-validated development performance.
A new untouched Fresh-v3 set will be required before claiming final generalization.
```

New files:

```text
unified_hidden_router_cv_v120.py
run_unified_hidden_router_cv_v120.py
```

Run:

```powershell
git fetch origin
git checkout v1.2.0
git pull origin v1.2.0

python run_unified_hidden_router_cv_v120.py
```

Outputs:

```text
results/unified_hidden_router_cv_v120/cv.log
results/unified_hidden_router_cv_v120/summary.csv
results/unified_hidden_router_cv_v120/linear_oof_predictions.csv
results/unified_hidden_router_cv_v120/linear_confusion.csv
results/unified_hidden_router_cv_v120/mlp_oof_predictions.csv
results/unified_hidden_router_cv_v120/mlp_confusion.csv
```

Reference points:

```text
Raw detector-score argmax : 51.7%
Calibrated score router   : about 60%
```

If hidden-state cross-validation is clearly above the calibration reference, v1.2.x should continue with the unified hidden-state router architecture rather than adding more serial concept-specific detectors.

### v1.2.1: Expanded Unified Semantic Router Dataset

v1.2.0 achieved 63.3% five-fold CV accuracy with both Linear and MLP hidden-state routers on the 60-prompt Fresh-v2 technical development set. The main remaining weakness was GPU/CPU separation, while CUDA and Python were substantially stronger.

v1.2.1 expands semantic router training data to:

```text
6 classes x 50 prompts = 300 prompts

GPU
CPU
LLM
Transformer
CUDA
Python
```

Each class contains five semantic families with ten prompts per family. The dataset intentionally includes hard contrasts such as:

```text
GPU vs CPU
GPU vs CUDA
CPU vs GPU
LLM vs Transformer
Transformer vs LLM
CUDA vs GPU
Python vs model/hardware concepts
```

To reduce paraphrase leakage, cross-validation is family-held-out rather than random-prompt CV:

```text
5 folds
1 unseen semantic family per class per fold
60 test prompts per fold
240 training prompts per fold
```

Both router architectures are evaluated again:

```text
Linear : 256 -> 6
MLP    : 256 -> 64 -> GELU -> Dropout -> 6
```

After cross-validation, each architecture is also trained on all 300 expanded prompts and checked against the 60 technical Fresh-v2 prompts as a development transfer test.

New files:

```text
semantic_router_dataset_v121.py
unified_hidden_router_cv_v121.py
run_unified_hidden_router_cv_v121.py
```

Run:

```powershell
git fetch origin
git checkout v1.2.1
git pull origin v1.2.1

python run_unified_hidden_router_cv_v121.py
```

Outputs:

```text
results/unified_hidden_router_cv_v121/cv.log
results/unified_hidden_router_cv_v121/summary.csv
results/unified_hidden_router_cv_v121/linear_cv_predictions.csv
results/unified_hidden_router_cv_v121/linear_cv_confusion.csv
results/unified_hidden_router_cv_v121/linear_fresh_v2_confusion.csv
results/unified_hidden_router_cv_v121/mlp_cv_predictions.csv
results/unified_hidden_router_cv_v121/mlp_cv_confusion.csv
results/unified_hidden_router_cv_v121/mlp_fresh_v2_confusion.csv
```

Reference:

```text
v1.2.0 hidden-state CV: 63.3%
```

The main question for v1.2.1 is whether broader semantic training data improves GPU/CPU recognition without sacrificing the strong CUDA, Python, and Transformer separation already observed.

### v1.2.2: Semantic Family Invariance Training

v1.2.1 showed a strong development-transfer gain on Fresh-v2 (Linear 81.7%) but only 53.0% family-held-out CV. This indicates that the router can learn seen semantic families but still struggles to generalize across unseen families within the same concept.

v1.2.2 adds an explicit invariance objective:

```text
Frozen base hidden state (256)
        ↓
Semantic projection 256 -> 64
        ↓
6-class router
```

Training loss:

```text
L = CrossEntropy + lambda * SupervisedContrastiveLoss
```

The supervised contrastive term pulls examples from the same semantic class together across different families and pushes examples from different classes apart.

Lambda sweep:

```text
0.00
0.10
0.25
0.50
1.00
```

Evaluation remains 5-fold family-held-out CV on the 300-prompt v1.2.1 dataset. Each fold holds out one complete semantic family from every class.

A full-data Fresh-v2 transfer check is also retained to ensure that stronger invariance does not destroy the previously observed transfer performance.

New files:

```text
semantic_family_invariance_cv_v122.py
run_semantic_family_invariance_v122.py
```

Run:

```powershell
git fetch origin
git checkout v1.2.2
git pull origin v1.2.2

python run_semantic_family_invariance_v122.py
```

Outputs:

```text
results/semantic_family_invariance_v122/invariance.log
results/semantic_family_invariance_v122/summary.csv
results/semantic_family_invariance_v122/lambda_*_cv_confusion.csv
results/semantic_family_invariance_v122/lambda_*_fresh_v2_confusion.csv
```

Reference:

```text
v1.2.1 Linear family-CV : 53.0%
v1.2.1 Linear Fresh-v2  : 81.7%
```

The main question is whether lambda > 0 raises family-held-out CV while preserving the strong CUDA, Python, and Transformer transfer performance.
### v1.2.3: Base Representation Adaptation

v1.2.2 improved family-held-out CV only slightly, from 53.0% to 55.0%, with the best supervised-contrastive weight at 0.10. Fresh-v2 transfer remained strong at 81.7%. This suggests that the frozen final hidden representation is still the main bottleneck for semantic-family invariance.

v1.2.3 therefore adapts the late base representation instead of only training an external router.

Trainable path:

    Embedding + Blocks 1-4 : frozen
    Blocks 5-6             : low-LR trainable
    FinalNorm              : trainable
    Semantic projection    : 256 -> 64
    6-class router         : trainable

Loss:

    L = CrossEntropy
      + 0.10 * SupervisedContrastiveLoss
      + alpha * RepresentationPreservationLoss

The preservation term is MSE between the adapted final hidden state and the original clean-base final hidden state.

Preservation sweep:

    alpha = 0.0, 0.1, 0.5, 1.0

Learning rates:

    Blocks 5-6 + FinalNorm : 1e-5
    Projection + Router    : 8e-4

For efficiency, Embedding + Blocks 1-4 are executed once and cached. Training then operates on cached sequence states.

Evaluation:

    300 prompts
    6 classes
    5-fold family-held-out CV
    2 seeds per fold

The experiment also reports mean hidden drift:

    drift = mean(1 - cosine(adapted_hidden, original_hidden))

New files:

    base_representation_adaptation_cv_v123.py
    run_base_representation_adaptation_v123.py

Run:

    git fetch origin
    git checkout v1.2.3
    git pull origin v1.2.3
    python run_base_representation_adaptation_v123.py

Outputs:

    results/base_representation_adaptation_v123/adaptation.log
    results/base_representation_adaptation_v123/summary.csv
    results/base_representation_adaptation_v123/preservation_*_cv_confusion.csv

Reference:

    v1.2.2 best family-CV : 55.0%

The main decision is whether adapting Blocks 5-6 materially improves unseen-family recognition while keeping hidden-state drift small.
### v1.2.4: Late-Block Adaptation Diagnostic + LR Sweep

v1.2.3 produced 53.0% family-held-out CV for every preservation weight and reported hidden drift as 0.000000. v1.2.4 checks whether adaptation is actually occurring before changing architecture again.

Base LR sweep:

    1e-5
    3e-5
    1e-4
    3e-4

Fixed settings:

    Blocks 5-6 + FinalNorm : trainable
    Embedding + Blocks 1-4 : frozen/cached
    Projection + Router LR : 8e-4
    Loss                   : CE + 0.10*SupCon + 0.10*preservation
    Evaluation             : 5-fold family-held-out CV

For every fold/seed, the diagnostic records:

    Block5 relative parameter delta
    Block6 relative parameter delta
    FinalNorm relative parameter delta
    Block5 average gradient norm
    Block6 average gradient norm
    FinalNorm average gradient norm
    hidden drift = mean(1 - cosine(adapted, original))
    family-held-out accuracy

Hidden drift and parameter deltas are printed with high precision so that very small adaptation is not rounded to zero.

New files:

    late_block_adaptation_diagnostic_v124.py
    run_late_block_adaptation_diagnostic_v124.py

Run:

    git fetch origin
    git checkout v1.2.4
    git pull origin v1.2.4
    python run_late_block_adaptation_diagnostic_v124.py

Outputs:

    results/late_block_adaptation_diagnostic_v124/diagnostic.log
    results/late_block_adaptation_diagnostic_v124/summary.csv
    results/late_block_adaptation_diagnostic_v124/lr_*_diagnostics.csv
    results/late_block_adaptation_diagnostic_v124/lr_*_cv_confusion.csv

Decision rule:

    non-zero gradients + increasing parameter deltas with LR -> adaptation is active
    deltas increase but accuracy stays flat               -> late blocks are insufficient
    accuracy > 55%                                        -> continue late-block adaptation
### v1.2.5: Trainable Late-Block Fix

v1.2.4 proved that Blocks 5-6 and FinalNorm were not actually adapting: all gradient norms and parameter deltas were exactly zero across the full LR sweep.

Root cause:

    the clean base model was frozen with requires_grad=False
    deepcopy() preserved requires_grad=False in Block5, Block6 and FinalNorm

v1.2.5 explicitly re-enables gradients after copying the late layers:

    Block5    : requires_grad=True
    Block6    : requires_grad=True
    FinalNorm : requires_grad=True

The experiment then repeats the same LR sweep and diagnostics as v1.2.4:

    1e-5
    3e-5
    1e-4
    3e-4

Measured signals:

    family-held-out CV accuracy
    hidden drift
    Block5 relative parameter delta
    Block6 relative parameter delta
    FinalNorm relative parameter delta
    Block5 gradient norm
    Block6 gradient norm
    FinalNorm gradient norm

New files:

    trainable_late_block_fix_v125.py
    run_trainable_late_block_fix_v125.py

Run:

    git fetch origin
    git checkout v1.2.5
    git pull origin v1.2.5
    python run_trainable_late_block_fix_v125.py

Outputs:

    results/trainable_late_block_fix_v125/diagnostic.log
    results/trainable_late_block_fix_v125/summary.csv
    results/trainable_late_block_fix_v125/lr_*_diagnostics.csv
    results/trainable_late_block_fix_v125/lr_*_cv_confusion.csv

Primary success condition:

    gradient norms > 0
    parameter deltas > 0

Accuracy is then compared against the v1.2.2 frozen-base reference of 55.0%.
### v1.2.6: Layer-wise Semantic Separability Diagnostic

v1.2.5 confirmed that late-block adaptation was genuinely active, but even the best LR reached only 54.7% family-held-out CV, below the 55.0% frozen-base reference.

v1.2.6 therefore measures semantic separability at every representation stage of the clean base model:

    Embedding
    Block1
    Block2
    Block3
    Block4
    Block5
    Block6
    FinalNorm

For each stage, the final-token hidden vector is extracted from the completely frozen base model and evaluated with the same Linear 256 -> 6 probe under 5-fold family-held-out CV.

Dataset:

    300 prompts
    6 classes
    5 semantic families/class

Classes:

    GPU
    CPU
    LLM
    Transformer
    CUDA
    Python

Outputs include per-stage overall accuracy, macro recall, per-class recall, and confusion matrices.

New files:

    layerwise_semantic_separability_v126.py
    run_layerwise_semantic_separability_v126.py

Run:

    git fetch origin
    git checkout v1.2.6
    git pull origin v1.2.6
    python run_layerwise_semantic_separability_v126.py

Outputs:

    results/layerwise_semantic_separability_v126/diagnostic.log
    results/layerwise_semantic_separability_v126/summary.csv
    results/layerwise_semantic_separability_v126/*_confusion.csv

Interpretation:

    earlier layer clearly better than Block6/FinalNorm -> later processing erases separability
    all layers weak for GPU/CPU                     -> base representation learning is the bottleneck

References:

    v1.2.2 frozen final representation : 55.0%
    v1.2.5 adapted late-block best     : 54.7%
### v1.2.7: Multi-Layer Semantic Fusion

v1.2.6 found that Block1 had the best single-layer family-held-out separability at 60.7%, while Block4 and FinalNorm preserved different class-specific strengths. v1.2.7 tests whether those distributed signals can be combined.

Candidate layers:

    Block1
    Block4
    FinalNorm

Fusion is simple concatenation. Evaluated feature sets:

    Block1
    Block4
    FinalNorm
    Block1 + Block4
    Block1 + FinalNorm
    Block4 + FinalNorm
    Block1 + Block4 + FinalNorm

Each feature set is evaluated with both:

    Linear router
    MLP router (input -> 128 -> 6)

Evaluation remains 5-fold family-held-out CV on the 300-prompt semantic dataset, with the base LM completely frozen.

New files:

    multilayer_semantic_fusion_v127.py
    run_multilayer_semantic_fusion_v127.py

Run:

    git fetch origin
    git checkout v1.2.7
    git pull origin v1.2.7
    python run_multilayer_semantic_fusion_v127.py

Outputs:

    results/multilayer_semantic_fusion_v127/fusion.log
    results/multilayer_semantic_fusion_v127/summary.csv
    results/multilayer_semantic_fusion_v127/*_confusion.csv

References:

    v1.2.6 Block1 Linear : 60.7%
    v1.2.6 Block4 Linear : 55.0%
    v1.2.6 FinalNorm     : 53.3%

Primary decision:

    fusion > 60.7% -> routing benefits from semantic signals distributed across layers
    fusion <= 60.7% -> Block1 remains the preferred routing tap point

GPU recall is tracked separately because v1.2.6 showed weak GPU separability at every individual layer.
### v1.2.8: Class-Specific Layer Gating

v1.2.7 showed that simple multi-layer concatenation does not beat Block1 Linear (60.7%), but GPU recognition is stronger at Block4 with an MLP specialist. v1.2.8 therefore uses class-dependent routing instead of global fusion.

Architecture:

    Block1 -> Linear 6-class general router
    Block4 -> MLP GPU-vs-NonGPU specialist

Policy:

    use Block1 general prediction by default
    if Block4 GPU specialist confidence >= threshold and general prediction is not GPU
    override prediction to GPU

GPU specialist:

    input 256
    hidden 64
    output 2
    positive-class weight 5.0

Threshold sweep:

    0.50
    0.60
    0.70
    0.80
    0.85
    0.90
    0.95

Evaluation:

    300 prompts
    6 classes
    5-fold family-held-out CV
    3 seeds/fold

Reported diagnostics:

    overall accuracy
    macro recall
    per-class recall
    override count
    helpful overrides
    harmful overrides
    correct GPU overrides
    false GPU overrides
    override precision

New files:

    class_specific_layer_gating_v128.py
    run_class_specific_layer_gating_v128.py

Run:

    git fetch origin
    git checkout v1.2.8
    git pull origin v1.2.8
    python run_class_specific_layer_gating_v128.py

Outputs:

    results/class_specific_layer_gating_v128/gating.log
    results/class_specific_layer_gating_v128/threshold_sweep.csv
    results/class_specific_layer_gating_v128/oof_scores.csv
    results/class_specific_layer_gating_v128/threshold_*_confusion.csv

Reference:

    v1.2.7 Block1 Linear : 60.7%

Important: threshold selection is development tuning on the same 300-prompt dataset. A future untouched Fresh-v3 set is required for unbiased final validation.
### v1.2.9: GPU Semantic Failure Analysis

v1.2.8 showed that a Block4 GPU-vs-NonGPU specialist did not provide a reliable gating signal. The mean GPU specialist score was only slightly higher on GPU prompts than on non-GPU prompts, and every threshold reduced overall accuracy.

v1.2.9 therefore stops adding routing policy and analyzes where the GPU concept is failing.

Analysis 1: multiclass GPU failure modes by layer

    Block1
    Block2
    Block3
    Block4
    Block5
    Block6
    FinalNorm

For each layer, a 6-class Linear probe is evaluated with 5-fold family-held-out CV. GPU prompts are then broken down by predicted class:

    GPU -> GPU
    GPU -> CPU
    GPU -> LLM
    GPU -> Transformer
    GPU -> CUDA
    GPU -> Python

Analysis 2: GPU family failure modes

The 50 GPU prompts are broken down by their five semantic families so that each family can be traced to its dominant confusion class at each layer.

Analysis 3: GPU pairwise separability

Pairwise Linear probes measure:

    GPU vs CPU
    GPU vs LLM
    GPU vs Transformer
    GPU vs CUDA
    GPU vs Python

for every layer using the same family-held-out principle.

New files:

    gpu_semantic_failure_analysis_v129.py
    run_gpu_semantic_failure_analysis_v129.py

Run:

    git fetch origin
    git checkout v1.2.9
    git pull origin v1.2.9
    python run_gpu_semantic_failure_analysis_v129.py

Outputs:

    results/gpu_semantic_failure_analysis_v129/analysis.log
    results/gpu_semantic_failure_analysis_v129/gpu_multiclass_by_layer.csv
    results/gpu_semantic_failure_analysis_v129/gpu_family_failure_modes.csv
    results/gpu_semantic_failure_analysis_v129/gpu_pairwise_separability.csv
    results/gpu_semantic_failure_analysis_v129/gpu_pairwise_best_stage.csv

Interpretation:

    low pairwise accuracy -> true semantic boundary problem
    high pairwise accuracy + poor 6-class GPU recall -> pairwise information exists but no stable multiclass GPU region

The purpose of v1.2.9 is diagnostic only; it does not deploy a new routing policy.
### v1.3.0: GPU-CPU Semantic Boundary Training

v1.2.9 identified GPU-vs-CPU as the weakest GPU semantic boundary. Best pairwise family-held-out accuracy was only 57.0%, while GPU-vs-CUDA reached 96.0% and GPU-vs-LLM/Transformer were around 80%.

v1.3.0 therefore trains the Block1 representation directly for the GPU-vs-CPU boundary instead of adding another routing policy.

Adapted component:

    Block1

Frozen components:

    Embedding
    Blocks 2-6
    FinalNorm
    LM head

Boundary training data:

    GPU + CPU samples from the 300-prompt semantic dataset
    fold training uses only the four seen semantic families
    the fifth GPU family and fifth CPU family remain held out

Loss:

    Binary CrossEntropy
    + 0.25 * supervised contrastive loss
    + 0.10 * Block1 representation preservation loss

Block1 LR sweep:

    1e-5
    3e-5
    1e-4

Evaluation has two stages for every fold:

    1. Direct GPU-vs-CPU binary accuracy on held-out GPU/CPU families
    2. A fresh 6-class Linear probe trained on the adapted Block1 features

This distinguishes a true improvement to the GPU-CPU semantic boundary from a narrow binary fix that damages broader semantic routing.

New files:

    gpu_cpu_boundary_training_v130.py
    run_gpu_cpu_boundary_training_v130.py

Run:

    git fetch origin
    git checkout v1.3.0
    git pull origin v1.3.0
    python run_gpu_cpu_boundary_training_v130.py

Outputs:

    results/gpu_cpu_boundary_training_v130/boundary.log
    results/gpu_cpu_boundary_training_v130/summary.csv
    results/gpu_cpu_boundary_training_v130/lr_*_six_class_confusion.csv

References:

    v1.2.9 GPU-vs-CPU pairwise : 57.0%
    v1.2.7 Block1 6-class      : 60.7%

Target:

    raise GPU-vs-CPU held-out accuracy toward 75-80%
    while retaining useful 6-class family-held-out performance

Fresh-v2 is already a development set. Final validation still requires a future untouched Fresh-v3 set.
### v1.3.1: GPU-CPU Family Cross-Generalization Matrix

v1.3.0 showed that directly adapting Block1 for GPU-vs-CPU did not improve held-out family generalization. v1.3.1 therefore diagnoses whether the problem is a weak concept boundary or family-specific wording overlap.

GPU families:

    parallel
    matrix
    graphics
    throughput
    contrast_cpu

CPU families:

    control
    instruction
    general
    sequential
    contrast_gpu

Representation:

    frozen Block1 final-token hidden state

Probe:

    Linear 256 -> 2

Three 5x5 matrices are produced:

    A. within-pair 5-fold CV
       Train/test inside one GPU-family + CPU-family pair.

    B. leave-target-pair-out generalization
       Train on the other four GPU families and four CPU families, then test the fully unseen target GPU-family + CPU-family pair.

    C. source-pair-only -> all other families
       Train only on one GPU-family + CPU-family pair, then test on all remaining GPU/CPU families.

For every matrix, overall accuracy, GPU recall, and CPU recall are written separately.

Additional output ranks target-family vulnerability using mean leave-pair-out accuracy.

New files:

    gpu_cpu_family_cross_generalization_v131.py
    run_gpu_cpu_family_cross_generalization_v131.py

Run:

    git fetch origin
    git checkout v1.3.1
    git pull origin v1.3.1
    python run_gpu_cpu_family_cross_generalization_v131.py

Outputs:

    results/gpu_cpu_family_cross_generalization_v131/matrix.log
    results/gpu_cpu_family_cross_generalization_v131/within_pair_accuracy.csv
    results/gpu_cpu_family_cross_generalization_v131/leave_pair_out_accuracy.csv
    results/gpu_cpu_family_cross_generalization_v131/source_pair_to_others_accuracy.csv
    results/gpu_cpu_family_cross_generalization_v131/*_gpu_recall.csv
    results/gpu_cpu_family_cross_generalization_v131/*_cpu_recall.csv
    results/gpu_cpu_family_cross_generalization_v131/family_pair_details.csv
    results/gpu_cpu_family_cross_generalization_v131/target_family_vulnerability.csv

Interpretation:

    low within-pair -> the family definitions themselves overlap
    high within-pair + low leave-pair-out -> wording/family overfitting
    low source-pair-to-others -> that family pair is a poor teaching basis for the general GPU-vs-CPU boundary

The purpose is diagnostic only; no routing policy or base-model weights are changed.
### v1.3.2: Family-Invariant GPU-CPU Representation Learning

v1.3.1 showed a strong diagnostic split: within-family GPU/CPU pairs were 100% separable, but leave-target-pair-out generalization averaged only 39.0%. This indicates family-specific local separation without a transferable global GPU-vs-CPU concept boundary.

v1.3.2 learns a semantic adapter on top of frozen Block1 representations.

Architecture:

    frozen Block1 hidden (256)
        -> residual semantic adapter 256 -> 128 -> 256
        -> GPU/CPU binary head

The residual adapter starts at the original Block1 representation.

Training objective:

    Binary CrossEntropy
    + 0.10 * class compactness
    + 0.50 * GPU/CPU centroid separation
    + 1.00 * family-invariance loss
    + 0.25 * representation-preservation loss

Primary evaluation:

    25 target GPU-family x CPU-family leave-pair-out tests

For each target pair, the adapter sees the other four GPU families and four CPU families, while the target GPU and CPU families remain unseen.

Secondary evaluation:

    5-fold 6-class family-held-out routing

A new 6-class Linear probe is trained on adapted representations to verify that GPU/CPU boundary learning does not destroy broader semantic routing.

References:

    v1.3.1 leave-pair-out mean : 39.0%
    v1.2.7 Block1 6-class      : 60.7%

Targets:

    leave-pair-out mean >= 60%
    keep 6-class accuracy near the 60.7% Block1 reference

New files:

    family_invariant_gpu_cpu_v132.py
    run_family_invariant_gpu_cpu_v132.py

Run:

    git fetch origin
    git checkout v1.3.2
    git pull origin v1.3.2
    python run_family_invariant_gpu_cpu_v132.py

Outputs:

    results/family_invariant_gpu_cpu_v132/family_invariant.log
    results/family_invariant_gpu_cpu_v132/leave_pair_out_accuracy.csv
    results/family_invariant_gpu_cpu_v132/leave_pair_out_details.csv
    results/family_invariant_gpu_cpu_v132/six_class_confusion.csv
    results/family_invariant_gpu_cpu_v132/summary.csv

If leave-pair-out remains low despite the family-invariant objective, the next hypothesis is that frozen Block1 does not contain enough transferable GPU/CPU structure and semantic supervision must reach base-model representation learning.
### v1.3.3: Semantic-Supervised Base Representation Training

v1.3.2 showed that an adapter on top of frozen Block1 representations reduced leave-pair-out GPU/CPU generalization from 39.0% to 30.6%. v1.3.3 therefore moves semantic supervision into base representation formation.

Trainable components:

    Token embedding
    Positional embedding (if enabled)
    Block1
    Block2

Frozen components:

    Blocks3-6
    FinalNorm
    LM Head

Training objective:

    LM next-token loss
    + 0.35 * GPU/CPU semantic classification loss
    + 0.10 * family-invariance anchor loss

The LM objective is evaluated through the complete model, so gradients flow through the frozen later blocks into Block2, Block1, and the embeddings while the later parameters themselves remain unchanged.

Evaluation 1: exact 25-pair leave-target-pair-out matrix

    For every GPU-family x CPU-family target pair, that pair is removed before semantic/base training.
    A fresh Linear GPU-vs-CPU probe is then trained on the seen adapted Block1 features and tested on the unseen target pair.

Evaluation 2: standard 5-fold semantic-family holdout

    GPU-vs-CPU pairwise Linear probe
    6-class Linear routing probe

References:

    v1.3.1 leave-pair-out mean : 39.0%
    v1.2.9 GPU-vs-CPU pairwise : 57.0%
    v1.2.7 Block1 6-class      : 60.7%

Initial targets:

    leave-pair-out mean >= 55-60%
    GPU-vs-CPU pairwise >= 70%
    6-class >= 58-60%

New files:

    semantic_supervised_base_training_v133.py
    run_semantic_supervised_base_training_v133.py

Run:

    git fetch origin
    git checkout v1.3.3
    git pull origin v1.3.3
    python run_semantic_supervised_base_training_v133.py

Outputs:

    results/semantic_supervised_base_v133/training.log
    results/semantic_supervised_base_v133/leave_pair_out_accuracy.csv
    results/semantic_supervised_base_v133/six_class_confusion.csv
    results/semantic_supervised_base_v133/summary.csv

Fresh-v2 is already development data. Final validation still requires an untouched Fresh-v3 set.
### v1.3.4: Semantic Dataset Redesign Diagnostic

v1.3.3 showed that moving semantic supervision into Embedding/Block1/Block2 still did not improve GPU-vs-CPU cross-family generalization. v1.3.4 therefore tests the dataset ontology itself before changing the model again.

Problem identified in the previous dataset:

    GPU families mixed workload, application, performance behavior, and CPU contrast.
    CPU families mixed control role, instruction behavior, general-purpose usage, sequential behavior, and GPU contrast.

Those family sets were not aligned semantic axes, so holding out one family per class also changed the semantic dimension being tested.

v1.3.4 adds an aligned GPU/CPU dataset with the same five axes on both classes:

    definition
    architecture
    workload
    comparison
    application

Each class has 10 prompts per axis:

    2 classes x 5 axes x 10 prompts = 100 prompts

Base model weights remain completely frozen. The diagnostic uses the Block1 final-token hidden representation and a Linear 256 -> 2 probe.

Evaluations:

    1. Old dataset paired-index leave-family-out baseline
    2. Redesigned within-axis 5-fold CV
    3. Redesigned leave-one-axis-out
    4. Redesigned mixed-axis stratified 5-fold CV

Interpretation:

    redesigned leave-axis-out >> old leave-family-out
        -> ontology mismatch was a major source of apparent generalization failure

    high within-axis / mixed-axis but low leave-axis-out
        -> Block1 still depends on semantic-axis-specific cues rather than a shared GPU-vs-CPU abstraction

New files:

    semantic_gpu_cpu_aligned_v134.py
    semantic_dataset_redesign_diagnostic_v134.py
    run_semantic_dataset_redesign_diagnostic_v134.py

Run:

    git fetch origin
    git checkout v1.3.4
    git pull origin v1.3.4
    python run_semantic_dataset_redesign_diagnostic_v134.py

Outputs:

    results/semantic_dataset_redesign_v134/diagnostic.log
    results/semantic_dataset_redesign_v134/comparison.csv
    results/semantic_dataset_redesign_v134/old_leave_family_out.csv
    results/semantic_dataset_redesign_v134/redesigned_within_axis.csv
    results/semantic_dataset_redesign_v134/redesigned_leave_axis_out.csv

This experiment intentionally changes only the semantic ontology/evaluation dataset, not the base model or routing policy.
### v1.3.5: Comparison-Axis Redesign / Hard Contrast Diagnostic

v1.3.4 showed that ontology alignment improved GPU/CPU leave-one-axis-out generalization to 62.0%, but the comparison axis remained a clear outlier (30% within-axis, 35% leave-axis-out).

v1.3.5 redesigns only the comparison axis using matched hard-contrast pairs.

Design rule:

    same topic family
    nearly identical question syntax
    opposite computational criterion
    expected answer flips only between GPU and CPU

Examples of paired criteria:

    high-throughput homogeneous parallel arithmetic vs low-latency branch-heavy control
    large matrix multiplication vs OS/interrupt control
    many lightweight arithmetic units vs fewer high-function cores
    data-parallel workload vs branch/control-dependent workload

Dataset:

    10 matched GPU/CPU pairs
    20 prompts total

Base model weights remain completely frozen. Evaluation uses the Block1 final-token hidden state with a Linear 256 -> 2 probe.

Evaluations:

    1. old v1.3.4 comparison-axis within-axis 5-fold CV
    2. hard matched-pair leave-one-pair-out CV
    3. train on aligned non-comparison axes, transfer to old comparison
    4. train on aligned non-comparison axes, transfer to hard comparison

Interpretation:

    hard-pair >> old-comparison
        -> old comparison wording/ontology caused much of the failure

    hard-pair good but non-comparison transfer low
        -> comparison relation remains a separate representation problem

    both hard-pair and transfer low
        -> frozen Block1 is weak at relational GPU/CPU comparison

New files:

    semantic_gpu_cpu_comparison_hard_v135.py
    comparison_axis_redesign_diagnostic_v135.py
    run_comparison_axis_redesign_diagnostic_v135.py

Run:

    git fetch origin
    git checkout v1.3.5
    git pull origin v1.3.5
    python run_comparison_axis_redesign_diagnostic_v135.py

Outputs:

    results/comparison_axis_redesign_v135/diagnostic.log
    results/comparison_axis_redesign_v135/summary.csv
    results/comparison_axis_redesign_v135/hard_pair_results.csv
### v1.3.6: Aligned Semantic Dataset v2

v1.3.5 showed that symmetric comparison wording substantially improved GPU/CPU comparison behavior. v1.3.6 generalizes that lesson to all six semantic classes.

Classes:

    GPU
    CPU
    LLM
    Transformer
    CUDA
    Python

Shared ontology axes:

    definition
    architecture
    function
    comparison
    application

Every class contains 10 prompts per axis:

    6 classes x 5 axes x 10 prompts = 300 prompts

The base model remains completely frozen. Evaluation uses the Block1 final-token hidden representation with a Linear 256 -> 6 probe.

Evaluations:

    1. Mixed-axis stratified 5-fold CV
       Every fold receives examples from every class and every semantic axis.

    2. Leave-one-axis-out
       One complete semantic axis is held out across all six classes.

The second evaluation is the main cross-axis generalization test because the ontology is now aligned across all classes.

New files:

    semantic_aligned_v2_v136.py
    aligned_semantic_dataset_v2_eval_v136.py
    run_aligned_semantic_dataset_v2_eval_v136.py

Run:

    git fetch origin
    git checkout v1.3.6
    git pull origin v1.3.6
    python run_aligned_semantic_dataset_v2_eval_v136.py

Outputs:

    results/aligned_semantic_dataset_v2_v136/evaluation.log
    results/aligned_semantic_dataset_v2_v136/summary.csv
    results/aligned_semantic_dataset_v2_v136/leave_axis_breakdown.csv

This experiment changes the semantic ontology and evaluation dataset only; the base model and routing policy are unchanged.
### v1.3.7: Six-Class Symmetric Comparison Dataset

v1.3.6 raised six-class mixed-axis accuracy to 82.3% and leave-one-axis-out accuracy to 75.0%, but the comparison axis remained the weakest at 55.0%. v1.3.7 redesigns comparison for all six classes using symmetric reciprocal contrast pairs.

Classes:

    GPU
    CPU
    LLM
    Transformer
    CUDA
    Python

Comparison construction:

    complete graph across all 6 classes
    15 unordered class pairs
    2 semantic topics per pair
    reciprocal prompt for each class in the pair
    30 reciprocal topics
    60 comparison prompts
    exactly 10 comparison prompts per class

Each reciprocal pair uses the same sentence form:

    target semantic description
    contrasted semantic description
    only the intended class direction is reversed

The remaining four axes (definition, architecture, function, application) are reused unchanged from v1.3.6.

Evaluations:

    1. v1.3.6 old comparison within-axis CV
    2. v1.3.7 symmetric comparison within-axis CV
    3. non-comparison axes -> old comparison transfer
    4. non-comparison axes -> symmetric comparison transfer
    5. full v1.3.7 mixed-axis stratified 5-fold CV
    6. full v1.3.7 leave-one-axis-out

All evaluations use the same completely frozen base model and the Block1 final-token hidden representation with a Linear 256 -> 6 probe.

References:

    v1.3.6 mixed-axis           : 82.3%
    v1.3.6 leave-one-axis-out   : 75.0%
    v1.3.6 comparison axis      : 55.0%

New files:

    semantic_symmetric_comparison_v137.py
    semantic_aligned_v3_v137.py
    symmetric_comparison_eval_v137.py
    run_symmetric_comparison_eval_v137.py

Run:

    git fetch origin
    git checkout v1.3.7
    git pull origin v1.3.7
    python run_symmetric_comparison_eval_v137.py

Outputs:

    results/symmetric_comparison_v137/evaluation.log
    results/symmetric_comparison_v137/summary.csv
    results/symmetric_comparison_v137/comparison_per_class.csv

The experiment changes only semantic comparison-data design. Base-model weights and routing policy remain unchanged.
### v1.3.8: Layer-wise Relational Comparison Diagnostic

v1.3.7 showed that symmetric six-class reciprocal comparison prompts collapsed at Block1 (5.0% within-axis accuracy), while non-comparison axes remained much stronger. v1.3.8 tests whether relational direction emerges in deeper Transformer layers.

Dataset:

    v1.3.7 symmetric comparison
    60 prompts
    6 classes
    10 prompts/class

Stages:

    Embedding
    Block1
    Block2
    Block3
    Block4
    Block5
    Block6
    FinalNorm

At every stage, the final-token hidden vector is evaluated with the same Linear 256 -> 6 probe under 5-fold class-stratified CV.

Base model weights remain completely frozen.

Interpretation:

    deeper layers >> Block1
        -> relational direction is represented later than concept identity

    all layers near chance
        -> reciprocal contrast / negation relation is not stably represented by the current base model

New files:

    layerwise_relational_comparison_v138.py
    run_layerwise_relational_comparison_v138.py

Run:

    git fetch origin
    git checkout v1.3.8
    git pull origin v1.3.8
    python run_layerwise_relational_comparison_v138.py

Outputs:

    results/layerwise_relational_comparison_v138/diagnostic.log
    results/layerwise_relational_comparison_v138/summary.csv
    results/layerwise_relational_comparison_v138/*_confusion.csv

Reference:

    v1.3.7 Block1 symmetric comparison within-axis : 5.0%
### v1.3.9: Relational Semantic Representation Diagnostic

v1.3.8 showed that six-class reciprocal comparison remains near chance across all layers (best Block3 = 20.0%). v1.3.9 decomposes the problem into concept identity and role direction.

Dataset:

    v1.3.7 symmetric reciprocal comparison
    60 prompts
    6 classes
    15 unordered class pairs

At every stage (Embedding, Block1-6, FinalNorm), independent Linear heads predict:

    Target class      : 6-way
    Contrast class    : 6-way
    Unordered pair    : 15-way

Derived diagnostics:

    unordered pair reconstructed from target+contrast heads
    strict ordered target+contrast accuracy
    direction accuracy conditioned on the unordered pair being recognized

Interpretation:

    high unordered-pair + low ordered-role
        -> both concepts are represented, but target/contrast direction is lost

    low unordered-pair
        -> even concept-pair identity is not stably recoverable from the final-token hidden vector

This directly tests whether semantic data should be represented as structured concept + role/relation information instead of a single undifferentiated vector.

New files:

    relational_semantic_representation_v139.py
    run_relational_semantic_representation_v139.py

Run:

    git fetch origin
    git checkout v1.3.9
    git pull origin v1.3.9
    python run_relational_semantic_representation_v139.py

Outputs:

    results/relational_semantic_representation_v139/diagnostic.log
    results/relational_semantic_representation_v139/summary.csv
### v1.4.0: Explicit Relation-Direction Head Diagnostic

v1.3.9 showed a strong asymmetry: unordered concept-pair identity was recoverable at roughly 66-75%, while ordered target/contrast accuracy stayed near zero to the low teens. v1.4.0 therefore factorizes relational semantics explicitly.

At every representation stage, two independent Linear heads predict:

    Unordered concept pair : 15-way
    Direction              : 2-way

Direction uses a canonical ordering of the two concepts:

    0 = target is canonical first concept
    1 = target is canonical second concept

The final semantic decision is reconstructed as:

    unordered pair + direction -> target concept + contrast concept

Diagnostics:

    pair accuracy
    explicit direction accuracy
    reconstructed target accuracy
    reconstructed contrast accuracy
    strict ordered target+contrast accuracy
    direction accuracy conditioned on pair being correct
    gold pair + predicted direction oracle
    predicted pair + gold direction oracle

These oracle decompositions identify whether pair identity or direction is the dominant bottleneck.

Stages:

    Embedding
    Block1
    Block2
    Block3
    Block4
    Block5
    Block6
    FinalNorm

New files:

    explicit_relation_direction_v140.py
    run_explicit_relation_direction_v140.py

Run:

    git fetch origin
    git checkout v1.4.0
    git pull origin v1.4.0
    python run_explicit_relation_direction_v140.py

Outputs:

    results/explicit_relation_direction_v140/diagnostic.log
    results/explicit_relation_direction_v140/summary.csv

Interpretation:

    direction > 50% with strong pair accuracy
        -> explicit structured concept-pair + role encoding can recover relational semantics

    direction near 50%
        -> role direction is not linearly encoded in the frozen representation
### v1.4.1: Token-wise Relation Direction Diagnostic

v1.4.0 showed that unordered concept-pair identity is recoverable, but explicit target/contrast direction remains far below chance from a single final-token hidden vector. v1.4.1 tests whether direction information is distributed across token states rather than concentrated at the final token.

Stages:

    Block1
    Block2
    Block3
    Block4
    Block5
    Block6
    FinalNorm

Pooling representations:

    final_token
    mean_all
    max_all
    mean_first_half
    mean_second_half
    mean_first_second_concat

Tasks:

    Direction head : 2-way
    Pair head      : 15-way
    Reconstructed target from predicted pair + predicted direction

Reference:

    v1.4.0 best final-token direction : 21.7% at Block3
    binary chance                     : 50.0%

Interpretation:

    token-wise pooling > 50% direction
        -> relation direction exists across token states but is lost by final-token compression

    all pooling <= 50%
        -> the frozen base representation lacks a stable linearly decodable target/contrast direction signal

New files:

    tokenwise_relation_direction_v141.py
    run_tokenwise_relation_direction_v141.py

Run:

    git fetch origin
    git checkout v1.4.1
    git pull origin v1.4.1
    python run_tokenwise_relation_direction_v141.py

Outputs:

    results/tokenwise_relation_direction_v141/diagnostic.log
    results/tokenwise_relation_direction_v141/summary.csv
### v1.4.2: Role-Position Disentanglement Diagnostic

v1.4.1 found direction accuracy above chance with first-half pooling (peak 63.3%), but the prompt template always placed the target description before the contrast description. v1.4.2 removes that confound by independently varying semantic role and surface order.

Dataset construction:

    30 semantic comparison topics
    2 target roles per topic
    2 surface orders per target role
    120 prompts total

Surface orders:

    target-first
    contrast-first

Role label:

    canonical-first concept is target
    canonical-second concept is target

Because each target role appears in both surface orders, Role and Position are no longer equivalent.

Evaluations:

    mixed-order 5-fold Role CV
    mixed-order Position CV
    mixed-order Pair 15-way CV
    reconstructed Target accuracy
    target-first -> contrast-first transfer
    contrast-first -> target-first transfer

Representations:

    final_token
    mean_all
    mean_first_half
    mean_second_half
    mean_first_second_concat

across Block1-6 and FinalNorm.

Interpretation:

    mixed Role high + cross-position Role near 50%
        -> apparent direction signal is mainly a surface-position shortcut

    cross-position Role clearly above 50%
        -> target/contrast role is represented independently of word order

Reference:

    v1.4.1 best first-half direction : 63.3%
    binary chance                    : 50.0%

New files:

    semantic_role_position_v142.py
    role_position_disentanglement_v142.py
    run_role_position_disentanglement_v142.py

Run:

    git fetch origin
    git checkout v1.4.2
    git pull origin v1.4.2
    python run_role_position_disentanglement_v142.py

Outputs:

    results/role_position_disentanglement_v142/diagnostic.log
    results/role_position_disentanglement_v142/summary.csv
### v1.4.3: Explicit Role Marker / Span Diagnostic

v1.4.2 showed that Concept Pair and surface Position are strongly represented, while semantic Target/Contrast Role remains weak. v1.4.3 therefore factorizes the prompt into explicit semantic components instead of relying on a single whole-prompt vector.

Structured inputs:

    Concept span 1
    Concept span 2
    Natural-language relation phrase

The relation phrase uses neutral positional wording such as:

    first concept is intended, second concept is not
    first concept is not intended, second concept is intended

No class name or TARGET/CONTRAST label token is inserted into the model input.

Feature combinations:

    span1
    span2
    relation
    span12
    span1_relation
    span2_relation
    span12_relation

Tasks:

    Role 2-way
    Pair 15-way
    Reconstructed Target

Evaluations:

    mixed-order 5-fold CV
    target-first -> contrast-first transfer
    contrast-first -> target-first transfer

Reference:

    v1.4.2 best mixed Role CV       : 21.7%
    v1.4.2 best cross-position Role : 39.2%
    binary chance                   : 50.0%

Interpretation:

    span12_relation >> span12
        -> explicit Concept + Relation factorization restores role binding

    relation-only dominates
        -> the role rule is carried mainly by the explicit relation phrase; concept pair is still needed to reconstruct the target class

New files:

    semantic_explicit_role_span_v143.py
    explicit_role_span_diagnostic_v143.py
    run_explicit_role_span_diagnostic_v143.py

Run:

    git fetch origin
    git checkout v1.4.3
    git pull origin v1.4.3
    python run_explicit_role_span_diagnostic_v143.py

Outputs:

    results/explicit_role_span_v143/diagnostic.log
    results/explicit_role_span_v143/summary.csv
### v1.4.4: Structured Semantic Composition

v1.4.3 showed that simply concatenating Concept spans and a relation vector with a Linear head did not recover semantic role binding. v1.4.4 replaces learned role classification with explicit structured composition.

Independent predictions:

    Concept1 : 6-way from span1
    Concept2 : 6-way from span2
    Target Position : 2-way from the natural-language relation phrase

Deterministic binding:

    if TargetPosition == FIRST:
        Target = Concept1
    else:
        Target = Concept2

This directly tests the structured semantic representation:

    Semantic = Concepts + Relation/TargetPosition + explicit binding

Evaluations:

    mixed-order 5-fold CV
    reconstructed Target accuracy
    gold concepts + predicted position oracle
    predicted concepts + gold position oracle
    target-first -> contrast-first transfer
    contrast-first -> target-first transfer

Stages:

    Block1
    Block2
    Block3
    Block4
    Block5
    Block6
    FinalNorm

Reference:

    v1.4.3 best reconstructed Target CV : 16.7%
    six-class chance                    : 16.7%

Interpretation:

    large Target improvement
        -> missing operation was explicit role binding rather than concept recognition

    gold concepts oracle high, gold position oracle low
        -> concept classification is the bottleneck

    gold position oracle high, gold concepts oracle low
        -> target-position relation decoding is the bottleneck

New files:

    structured_semantic_composition_v144.py
    run_structured_semantic_composition_v144.py

Run:

    git fetch origin
    git checkout v1.4.4
    git pull origin v1.4.4
    python run_structured_semantic_composition_v144.py

Outputs:

    results/structured_semantic_composition_v144/diagnostic.log
    results/structured_semantic_composition_v144/summary.csv
### v1.4.5: Relation Phrase Paraphrase Generalization

v1.4.4 achieved 98.3% reconstructed Target accuracy with structured Concept1 / Concept2 / TargetPosition composition. However, its cross-position split was confounded because one TargetPosition label could disappear from training. v1.4.5 replaces that split with leave-one-relation-family-out evaluation.

Relation data:

    5 wording families
    2 labels: FIRST target / SECOND target
    2 paraphrases per label per family
    20 relation phrases total

Families:

    direct
    select
    ordinal
    contrast
    referent

Every train/test split contains both semantic labels. Only wording family is held out.

Evaluation:

    leave one relation wording family out
    predict TargetPosition from unseen relation phrasing
    combine with Concept1/Concept2 predictions
    deterministic binding reconstructs Target Concept

Stages:

    Block1
    Block2
    Block3
    Block4
    Block5
    Block6
    FinalNorm

Reference:

    v1.4.4 reconstructed Target CV : 98.3%
    binary relation chance         : 50.0%

New files:

    semantic_relation_paraphrase_v145.py
    relation_paraphrase_generalization_v145.py
    run_relation_paraphrase_generalization_v145.py

Run:

    git fetch origin
    git checkout v1.4.5
    git pull origin v1.4.5
    python run_relation_paraphrase_generalization_v145.py

Outputs:

    results/relation_paraphrase_generalization_v145/diagnostic.log
    results/relation_paraphrase_generalization_v145/summary.csv

---

## License

This project is licensed under the **Apache License 2.0**.

Copyright © Hirofumi Inomata.

