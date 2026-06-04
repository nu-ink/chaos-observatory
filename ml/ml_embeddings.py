from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ArticleInput:
    article_id: str
    source: str | None
    title: str | None
    url: str | None
    published_at: str | None
    text: str | None
    summary: str | None = None


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return normalize_whitespace(value)


def build_embedding_text(article: ArticleInput) -> str:
    title = clean_text(article.title)
    source = clean_text(article.source)
    summary = clean_text(article.summary)
    body = clean_text(article.text)

    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if source:
        parts.append(f"Source: {source}")
    if summary:
        parts.append(f"Summary: {summary}")
    if body:
        parts.append(f"Text: {body}")

    return "\n\n".join(parts).strip()


@lru_cache(maxsize=4)
def load_embedding_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover - import/runtime environment
        raise ImportError(
            "sentence-transformers is required to load embedding models. "
            "Install with: pip install sentence-transformers"
        ) from e

    return SentenceTransformer(model_name)


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = load_embedding_model(model_name)

    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def generate_embedding(self, text: str) -> np.ndarray:
        if not clean_text(text):
            raise ValueError("Cannot generate embedding for empty text.")

        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=np.float32)

    def generate_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = [clean_text(text) for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("All texts must be non-empty.")

        vectors = self.model.encode(
            cleaned,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def generate_article_embedding(self, article: ArticleInput) -> np.ndarray:
        return self.generate_embedding(build_embedding_text(article))

    def generate_article_embeddings(
        self, articles: Iterable[ArticleInput]
    ) -> np.ndarray:
        texts = [build_embedding_text(article) for article in articles]
        return self.generate_embeddings(texts)
