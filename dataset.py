# dataset.py
#
# PyTorch dataset utilities for autoregressive next-token prediction.
#
# v0.4:
#   - Sample positions are calculated on demand.
#   - No giant Python list is created for very large training runs.
#   - max_samples may exceed the number of unique corpus windows; in that
#     case windows are revisited through a deterministic permutation.

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset):
    def __init__(
        self,
        token_ids: Sequence[int],
        context_length: int,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be > 0.")
        if len(token_ids) <= context_length:
            raise ValueError("Not enough tokens for one training sample.")

        self.token_ids = list(token_ids)
        self.context_length = context_length
        self.total_positions = len(self.token_ids) - context_length
        self.seed = int(seed)

        if max_samples is None:
            self.sample_count = self.total_positions
        else:
            if max_samples <= 0:
                raise ValueError("max_samples must be > 0 or None.")
            self.sample_count = int(max_samples)

        # Choose a stride coprime with total_positions so modular traversal
        # covers every possible corpus window before repeating.
        self._permutation_stride = self._choose_coprime_stride(
            self.total_positions
        )
        self._offset = self.seed % self.total_positions

    @staticmethod
    def _choose_coprime_stride(total: int) -> int:
        if total <= 1:
            return 1

        # Start near the golden-ratio fraction of the corpus length.
        stride = max(1, int(total * 0.6180339887498949))
        if stride >= total:
            stride = total - 1

        while math.gcd(stride, total) != 1:
            stride -= 1
            if stride <= 0:
                stride = 1
                break

        return stride

    def __len__(self) -> int:
        return self.sample_count

    def _position_for_index(self, index: int) -> int:
        if index < 0:
            index += self.sample_count
        if index < 0 or index >= self.sample_count:
            raise IndexError("dataset index out of range")

        # Memory-free pseudo-shuffled traversal. If sample_count is larger
        # than total_positions, the permutation naturally repeats.
        return (
            self._offset
            + index * self._permutation_stride
        ) % self.total_positions

    def __getitem__(self, index: int):
        start = self._position_for_index(index)
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
