# tokenizer.py
#
# Character-level tokenizer compatible with the homemade LLM project.

import json
from pathlib import Path
from typing import Dict, List, Optional


class Tokenizer:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.fitted = False
        for token in (
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ):
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return token_id

    def fit(self, text: str) -> None:
        for char in sorted(set(text)):
            self._add_token(char)
        self.fitted = True

    def fit_texts(self, texts: List[str]) -> None:
        self.fit("".join(texts))

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        if not self.fitted:
            raise RuntimeError("Tokenizer vocabulary has not been fitted.")

        result: List[int] = []
        if add_bos:
            result.append(self.bos_id)
        result.extend(self.token_to_id.get(char, self.unk_id) for char in text)
        if add_eos:
            result.append(self.eos_id)
        return result

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        special = {
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        }
        output = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), self.UNK_TOKEN)
            if skip_special_tokens and token in special:
                continue
            output.append(token)
        return "".join(output)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

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
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {"token_to_id": self.token_to_id, "fitted": self.fitted},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.token_to_id = {
            token: int(token_id)
            for token, token_id in data["token_to_id"].items()
        }
        tokenizer.id_to_token = {
            token_id: token
            for token, token_id in tokenizer.token_to_id.items()
        }
        tokenizer.fitted = data.get("fitted", True)
        return tokenizer
