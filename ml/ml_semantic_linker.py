from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .ml_embeddings import ArticleInput, EmbeddingService
from .ml_similarity_thresholds import SemanticSimilarityConfig, classify_score
from .vector_store import add_vector, load_index, save_index, search_vectors


@dataclass(frozen=True, slots=True)
class SemanticMatchResult:
    article_id: str
    matched_article_id: str
    similarity_score: float
    relationship_type: str
    matched_title: str | None
    matched_source: str | None
    matched_url: str | None
    matched_published_at: str | None
    faiss_index_id: int


class SemanticLinker:
    def __init__(self, config: SemanticSimilarityConfig) -> None:
        config.validate()
        self.config = config
        self.embedding_service = EmbeddingService(config.model_name)
        self.index = load_index(config.index_path, self.embedding_service.dimension())
        self.db_path = Path(config.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def process_article(self, article: ArticleInput) -> list[SemanticMatchResult]:
        self._validate_article(article)
        existing = self.get_article_metadata(article.article_id)
        if existing is not None:
            raise ValueError(f"article_id already exists: {article.article_id}")

        query_vector = self.embedding_service.generate_article_embedding(article)
        matches = self.find_similar_articles(query_vector, article.article_id, self.config.top_k)
        faiss_index_id = add_vector(self.index, query_vector)
        self.insert_article_embedding(article, faiss_index_id)
        self.store_semantic_matches(article.article_id, matches)
        save_index(self.index, self.config.index_path)
        return matches

    def process_articles(self, articles: list[ArticleInput]) -> dict[str, list[SemanticMatchResult]]:
        results: dict[str, list[SemanticMatchResult]] = {}
        for article in articles:
            results[article.article_id] = self.process_article(article)
        return results

    def find_similar_articles(
        self,
        query_vector: np.ndarray,
        current_article_id: str,
        top_k: int,
    ) -> list[SemanticMatchResult]:
        distances, ids = search_vectors(self.index, query_vector, top_k)
        matches: list[SemanticMatchResult] = []

        if distances.size == 0 or ids.size == 0:
            return matches

        for score, idx in zip(distances[0].tolist(), ids[0].tolist(), strict=False):
            if idx < 0:
                continue
            metadata = self.get_article_metadata_by_faiss_id(idx)
            if metadata is None:
                continue
            if metadata["article_id"] == current_article_id:
                continue

            similarity = float(max(min(score, 1.0), -1.0))
            matches.append(
                SemanticMatchResult(
                    article_id=current_article_id,
                    matched_article_id=metadata["article_id"],
                    similarity_score=similarity,
                    relationship_type=classify_score(similarity, self.config),
                    matched_title=metadata["title"],
                    matched_source=metadata["source"],
                    matched_url=metadata["url"],
                    matched_published_at=metadata["published_at"],
                    faiss_index_id=idx,
                )
            )
        return matches

    def insert_article_embedding(self, article: ArticleInput, faiss_index_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO article_embeddings (
                    article_id,
                    source,
                    title,
                    url,
                    published_at,
                    embedding_model,
                    faiss_index_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.article_id,
                    article.source,
                    article.title,
                    article.url,
                    article.published_at,
                    self.config.model_name,
                    faiss_index_id,
                    _utc_now_iso(),
                ),
            )
            conn.commit()

    def store_semantic_matches(self, article_id: str, matches: list[SemanticMatchResult]) -> None:
        if not matches:
            return

        rows = [
            (
                article_id,
                match.matched_article_id,
                match.similarity_score,
                match.relationship_type,
                _utc_now_iso(),
            )
            for match in matches
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO semantic_matches (
                    article_id,
                    matched_article_id,
                    similarity_score,
                    relationship_type,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def export_review_csv(self, output_path: str | None = None) -> Path:
        destination = Path(output_path or self.config.review_output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        query = """
        SELECT
            article_id,
            matched_article_id,
            similarity_score,
            relationship_type
        FROM semantic_matches
        ORDER BY created_at DESC, similarity_score DESC
        """

        with self._connect() as conn, destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "article_id",
                    "matched_article_id",
                    "similarity_score",
                    "relationship_type",
                    "human_label",
                ]
            )
            for row in conn.execute(query):
                writer.writerow([row[0], row[1], row[2], row[3], ""])

        return destination

    def get_article_metadata(self, article_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT article_id, source, title, url, published_at, embedding_model, faiss_index_id
                FROM article_embeddings
                WHERE article_id = ?
                """,
                (article_id,),
            ).fetchone()
        return _row_to_dict(row)

    def get_article_metadata_by_faiss_id(self, faiss_index_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT article_id, source, title, url, published_at, embedding_model, faiss_index_id
                FROM article_embeddings
                WHERE faiss_index_id = ?
                """,
                (faiss_index_id,),
            ).fetchone()
        return _row_to_dict(row)

    def rebuild_index_from_database(self, embedding_lookup: dict[str, np.ndarray]) -> faiss.Index:
        dimension = self.embedding_service.dimension()
        rebuilt = faiss.IndexFlatIP(dimension)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT article_id
                FROM article_embeddings
                ORDER BY faiss_index_id ASC
                """
            ).fetchall()

        vectors: list[np.ndarray] = []
        for (article_id,) in rows:
            vector = embedding_lookup.get(article_id)
            if vector is None:
                raise KeyError(f"Missing embedding for article_id={article_id}")
            vectors.append(np.asarray(vector, dtype=np.float32))

        if vectors:
            stacked = np.vstack(vectors).astype(np.float32)
            rebuilt.add(stacked)

        self.index = rebuilt
        save_index(self.index, self.config.index_path)
        return rebuilt

    def _initialize_database(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id TEXT UNIQUE NOT NULL,
                    source TEXT,
                    title TEXT,
                    url TEXT,
                    published_at TEXT,
                    embedding_model TEXT NOT NULL,
                    faiss_index_id INTEGER UNIQUE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id TEXT NOT NULL,
                    matched_article_id TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    relationship_type TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(article_id) REFERENCES article_embeddings(article_id),
                    FOREIGN KEY(matched_article_id) REFERENCES article_embeddings(article_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_cluster_id TEXT NOT NULL,
                    article_id TEXT NOT NULL,
                    confidence_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(article_id) REFERENCES article_embeddings(article_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_article_embeddings_article_id
                ON article_embeddings(article_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_article_embeddings_faiss_index_id
                ON article_embeddings(faiss_index_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_matches_article_id
                ON semantic_matches(article_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_matches_matched_article_id
                ON semantic_matches(matched_article_id)
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        return conn

    @staticmethod
    def _validate_article(article: ArticleInput) -> None:
        if not article.article_id.strip():
            raise ValueError("article_id is required.")
        if not ((article.title and article.title.strip()) or (article.text and article.text.strip()) or (article.summary and article.summary.strip())):
            raise ValueError("At least one of title, summary, or text must be provided.")


def _row_to_dict(row: tuple[Any, ...] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "article_id": row[0],
        "source": row[1],
        "title": row[2],
        "url": row[3],
        "published_at": row[4],
        "embedding_model": row[5],
        "faiss_index_id": row[6],
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
