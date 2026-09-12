from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9#]+")
STOPWORDS = {
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
    return [
        token
        for token in TOKEN_RE.findall(text.casefold())
        if token not in STOPWORDS
    ]


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    answer: str


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float


class BM25Retriever:
    """Compact local BM25 retriever used only when RETRIEVE is selected."""

    def __init__(
        self,
        documents: Iterable[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("At least one document is required")
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self._tokens: list[list[str]] = []
        self._term_frequencies: list[Counter[str]] = []

        total_length = 0
        for document in self.documents:
            tokens = tokenize(f"{document.title} {document.text}")
            self._tokens.append(tokens)
            self._term_frequencies.append(Counter(tokens))
            total_length += len(tokens)
        self._average_length = total_length / len(self.documents)

        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))

        document_count = len(self.documents)
        self._idf = {
            term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    @classmethod
    def from_package_data(cls) -> "BM25Retriever":
        path = files("adaptive_hitl_agent.data").joinpath("knowledge_base.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(Document(**item) for item in payload)

    def _score(self, query: str, index: int) -> float:
        frequencies = self._term_frequencies[index]
        length = len(self._tokens[index])
        score = 0.0
        for term in tokenize(query):
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

    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedDocument]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        scored = [
            RetrievedDocument(document=document, score=self._score(query, index))
            for index, document in enumerate(self.documents)
        ]
        scored.sort(key=lambda item: (-item.score, item.document.doc_id))
        return scored[:top_k]
