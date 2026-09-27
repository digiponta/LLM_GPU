# tokenizer_bpe.py
#
# LLM_GPU v0.7 byte-level BPE tokenizer.
#
# The v0.6 character tokenizer is intentionally left untouched for
# reproducibility. v0.7 uses this tokenizer for new checkpoints.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, normalizers, pre_tokenizers, trainers
from tokenizers.models import BPE


class BPETokenizer:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(
        self,
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ):
        self.target_vocab_size = int(vocab_size)
        self.min_frequency = int(min_frequency)
        self.backend: Optional[HFTokenizer] = None
        self.fitted = False

    @staticmethod
    def _new_backend() -> HFTokenizer:
        tokenizer = HFTokenizer(BPE(unk_token=BPETokenizer.UNK_TOKEN))
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True,
        )
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer

    @staticmethod
    def _chunks(
        texts: Sequence[str],
        chunk_chars: int = 100_000,
    ) -> Iterable[str]:
        for text in texts:
            if not text:
                continue
            for start in range(0, len(text), chunk_chars):
                yield text[start:start + chunk_chars]

    def fit_texts(self, texts: List[str]) -> None:
        backend = self._new_backend()
        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=[
                self.PAD_TOKEN,
                self.UNK_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
            ],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            max_token_length=32,
            show_progress=True,
        )
        backend.train_from_iterator(
            self._chunks(texts),
            trainer=trainer,
        )
        self.backend = backend
        self.fitted = True

    def fit(self, text: str) -> None:
        self.fit_texts([text])

    def _require_backend(self) -> HFTokenizer:
        if not self.fitted or self.backend is None:
            raise RuntimeError("BPE tokenizer has not been fitted.")
        return self.backend

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        backend = self._require_backend()
        ids = backend.encode(
            text,
            add_special_tokens=False,
        ).ids
        result: List[int] = []
        if add_bos:
            result.append(self.bos_id)
        result.extend(ids)
        if add_eos:
            result.append(self.eos_id)
        return result

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        backend = self._require_backend()
        return backend.decode(
            [int(x) for x in token_ids],
            skip_special_tokens=skip_special_tokens,
        )

    @property
    def vocab_size(self) -> int:
        return self._require_backend().get_vocab_size()

    def _token_id(self, token: str) -> int:
        value = self._require_backend().token_to_id(token)
        if value is None:
            raise RuntimeError(f"Special token missing: {token}")
        return int(value)

    @property
    def pad_id(self) -> int:
        return self._token_id(self.PAD_TOKEN)

    @property
    def unk_id(self) -> int:
        return self._token_id(self.UNK_TOKEN)

    @property
    def bos_id(self) -> int:
        return self._token_id(self.BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self._token_id(self.EOS_TOKEN)

    def pad(self, token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        return token_ids + [self.pad_id] * (max_length - len(token_ids))

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        rows = [
            self.encode(text, add_bos=add_bos, add_eos=add_eos)
            for text in texts
        ]
        if max_length is not None:
            rows = [self.pad(row, max_length) for row in rows]
        return rows

    def save(self, filename: str) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._require_backend().save(str(path))

    @classmethod
    def load(cls, filename: str):
        tokenizer = cls()
        tokenizer.backend = HFTokenizer.from_file(str(filename))
        tokenizer.fitted = True
        tokenizer.target_vocab_size = tokenizer.backend.get_vocab_size()
        return tokenizer


# Keep the same import name used by training/chat code.
Tokenizer = BPETokenizer
