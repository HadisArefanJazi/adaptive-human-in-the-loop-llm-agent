from __future__ import annotations

import json
import math
from importlib.resources import files
from typing import Iterable as iterable

from .records import immutable_record


stopwords = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "for",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
}


def tokenize(text: str) -> list[str]:
    cleaned = "".join(
        character
        if ("a" <= character <= "z" or "0" <= character <= "9" or character == "#")
        else " "
        for character in text.casefold()
    )
    return [token for token in cleaned.split() if token not in stopwords]


class document_record(immutable_record):
    doc_id: str
    title: str
    text: str
    answer: str

    __match_args__ = (
        "doc_id",
        "title",
        "text",
        "answer",
    )

    def __init__(
        self,
        doc_id: str,
        title: str,
        text: str,
        answer: str,
    ) -> None:
        object.__setattr__(self, "doc_id", doc_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "answer", answer)


class retrieved_document(immutable_record):
    document: document_record
    score: float

    __match_args__ = (
        "document",
        "score",
    )

    def __init__(
        self,
        document: document_record,
        score: float,
    ) -> None:
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "score", score)


class bm25_retriever:
    """compact local bm25 retriever used only when retrieve is selected."""

    def __init__(
        self,
        documents: iterable[document_record],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("at least one document is required")
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self._tokens: list[list[str]] = []
        self._term_frequencies: list[dict[str, int]] = []

        total_length = 0
        for document in self.documents:
            tokens = tokenize(f"{document.title} {document.text}")
            self._tokens.append(tokens)
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self._term_frequencies.append(frequencies)
            total_length += len(tokens)
        self._average_length = total_length / len(self.documents)

        document_frequency: dict[str, int] = {}
        for tokens in self._tokens:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        document_count = len(self.documents)
        self._idf = {
            term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    @classmethod
    def from_package_data(cls) -> "bm25_retriever":
        path = files("adaptive_hitl_agent.data").joinpath("knowledge_base.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(document_record(**item) for item in payload)

    def _score(self, query_tokens: list[str], index: int) -> float:
        frequencies = self._term_frequencies[index]
        length = len(self._tokens[index])
        score = 0.0
        for term in query_tokens:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / self._average_length
            )
            score += (
                self._idf.get(term, 0.0)
                * frequency
                * (self.k1 + 1)
                / denominator
            )
        return score

    def retrieve(self, query: str, top_k: int = 2) -> list[retrieved_document]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_tokens = tokenize(query)
        scored = [
            retrieved_document(document=document, score=self._score(query_tokens, index))
            for index, document in enumerate(self.documents)
        ]
        scored.sort(key=lambda item: (-item.score, item.document.doc_id))
        return scored[:top_k]
