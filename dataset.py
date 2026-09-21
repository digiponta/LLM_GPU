# dataset.py
#
# PyTorch dataset utilities for autoregressive next-token prediction.

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset):
    def __init__(
        self,
        token_ids: Sequence[int],
        context_length: int,
        max_samples: Optional[int] = None,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be > 0.")
        if len(token_ids) <= context_length:
            raise ValueError("Not enough tokens for one training sample.")

        self.token_ids = list(token_ids)
        self.context_length = context_length
        total = len(self.token_ids) - context_length

        if max_samples is None or max_samples >= total:
            self.positions = list(range(total))
        elif max_samples <= 0:
            raise ValueError("max_samples must be > 0 or None.")
        elif max_samples == 1:
            self.positions = [0]
        else:
            # Uniform coverage over the entire concatenated corpus.
            self.positions = [
                round(i * (total - 1) / (max_samples - 1))
                for i in range(max_samples)
            ]

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int):
        start = self.positions[index]
        end = start + self.context_length
        x = torch.tensor(
            self.token_ids[start:end],
            dtype=torch.long,
        )
        y = torch.tensor(
            self.token_ids[start + 1:end + 1],
            dtype=torch.long,
        )
        return x, y


def load_text_files(
    filenames: Sequence[str],
    separator: str = "\n\n",
) -> str:
    texts: List[str] = []
    for filename in filenames:
        with open(filename, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return separator.join(texts)
