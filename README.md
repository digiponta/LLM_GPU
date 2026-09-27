# LLM_GPU

CUDA/PyTorch GPU version of the homemade LLM project.

This repository is based on the architecture and end-to-end flow proven in
`digiponta/LLM` branch `v0.3`. The original educational virtual-GPU runtime
and hand-written backward propagation are replaced by real PyTorch tensors,
CUDA kernels, and PyTorch autograd.

## Architecture

```text
Japanese corpus
    |
character tokenizer
    |
Embedding
    |
Transformer blocks
    |-- LayerNorm
    |-- single-head causal Self-Attention
    |-- residual connection
    |-- LayerNorm
    |-- FFN (Linear -> GELU -> Linear)
    |-- residual connection
    |
final LayerNorm
    |
lm_head
    |
Cross Entropy
    |
PyTorch autograd
    |
AdamW
    |
CUDA GPU
```

## Files

```text
tokenizer.py          character tokenizer compatible with the original project
dataset.py            next-token dataset / uniform sampled windows
model.py              CUDA-capable Transformer language model
train.py              GPU training loop with progress and ETA
train_corpus.py       corpus pretraining entry point
train_conversation.py v0.5 conversational fine-tuning from the v0.4 checkpoint
chat.py               multi-turn conversational interface
evaluate_chat.py      deterministic conversational regression evaluation
infer.py              interactive raw GPU inference
check_gpu.py          CUDA/PyTorch diagnostic
prepare_wikipedia.py  Wikipedia dump -> plain text helper
requirements.txt      Python dependencies
```

## Installation

Install a CUDA-enabled PyTorch build suitable for the NVIDIA driver on the PC,
then install the project requirements.

```powershell
python -m pip install -r requirements.txt
python check_gpu.py
```

A successful setup reports:

```text
CUDA available : True
Device         : cuda
GPU            : NVIDIA ...
VRAM           : ... GiB
```

## Training data

By default the trainer looks for:

```text
data/general-ja.txt
data/data-nagato.txt
```

For convenience, if those files do not exist in this repository, it also
looks for the existing original-project files:

```text
../LLM/data/general-ja.txt
../LLM/data/data-nagato.txt
```

## Train

```powershell
python train_corpus.py
```

Default v0.4 long GPU run:

```text
context length : 64
d_model        : 64
layers         : 2
FFN dimension  : 256
attention heads: 1
batch size     : 64
samples        : 120,000,000
epochs         : 3
learning rate  : 5e-4
```

This sample count is estimated from the measured v0.3 benchmark on an
RTX 3070 Ti: 20,000 samples x 3 epochs took about 6 seconds. Linear scaling
gives approximately 120,000,000 samples x 3 epochs for a ten-hour run.

Actual runtime will vary with GPU clocks, thermals, system load, and I/O.
v0.4 computes sample positions on demand instead of allocating a huge Python
list, and DataLoader shuffling is disabled because the dataset itself uses a
deterministic pseudo-random corpus traversal.

Checkpoint:

```text
model/model-gpu-v0.4.pt
```

## Inference

```powershell
python infer.py
```

Generation uses temperature, top-k sampling, repetition penalty, and a bounded
context window.

## Relationship to the original v0.3

The original repository verified the complete path:

```text
corpus -> tokenizer -> model forward -> loss -> backward
       -> optimizer -> checkpoint -> reload -> inference
```

LLM_GPU preserves that path while moving tensor computation and gradient
calculation to a physical CUDA GPU.


---

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
