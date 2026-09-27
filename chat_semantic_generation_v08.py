# chat_semantic_generation_v08.py

from __future__ import annotations

from chat_semantic_generation_v062 import (
    AI_PREFIX,
    USER_PREFIX,
    build_prompt,
    compute_semantic_bias,
    generate_reply,
)

DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_BASE_MODEL = "model/model-gpu-v0.8-chat-pairwise-best.pt"
DEFAULT_SEMANTIC_ADAPTER = "model/model-gpu-v0.9.1-semantic-adapter-v08.pt"
DEFAULT_INTEGRATION = "model/model-gpu-v0.9.1-semantic-generation-v08.pt"
