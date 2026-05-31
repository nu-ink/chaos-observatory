#!/usr/bin/env python3
"""
Topic discovery and convergence monitor for Chaos Observatory.

Purpose:
- Read normalized news/event records from SQLite.
- Create semantic embeddings.
- Cluster similar records.
- Detect clusters gaining cross-source membership.
- Write a Markdown report.

Expected source table:
    documents, articles, or normalized_items

Expected columns, ideally:
    id/doc_id, title, body/body_text/summary/content, source/source_label,
    url, published_at/published_at_utc

If your schema uses different names, edit SOURCE_QUERIES below.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


DOCUMENTS_SOURCE_QUERY = """
SELECT
    doc_id AS id,
    COALESCE(title, '') AS title,
    COALESCE(body_text, '') AS body,
    COALESCE(source_label, source_id, 'unknown') AS source,
    COALESCE(url, '') AS url,
    COALESCE(published_at_utc, ingested_at_utc, normalized_at_utc, '') AS published_at
FROM documents
WHERE DATE(COALESCE(published_at_utc, ingested_at_utc, normalized_at_utc)) >= DATE(?)
"""


ARTICLES_SOURCE_QUERY = """
SELECT
    id,
    COALESCE(title, '') AS title,
    COALESCE(summary, content, description, '') AS body,
    COALESCE(source, feed_name, domain, 'unknown') AS source,
    COALESCE(url, link, '') AS url,
    COALESCE(published_at, published, created_at, '') AS published_at
FROM articles
WHERE DATE(COALESCE(published_at, published, created_at)) >= DATE(?)
"""


NORMALIZED_ITEMS_SOURCE_QUERY = """
SELECT
    id,
    COALESCE(title, '') AS title,
    COALESCE(summary, content, description, '') AS body,
    COALESCE(source, feed_name, domain, 'unknown') AS source,
    COALESCE(url, link, '') AS url,
    COALESCE(published_at, published, created_at, '') AS published_at
FROM normalized_items
WHERE DATE(COALESCE(published_at, published, created_at)) >= DATE(?)
"""


SOURCE_QUERIES = (
    DOCUMENTS_SOURCE_QUERY,
    ARTICLES_SOURCE_QUERY,
    NORMALIZED_ITEMS_SOURCE_QUERY,
)


def utc_now() -> str:
    """Return a UTC timestamp string."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def stable_text_id(row: pd.Series) -> str:
    """Create a stable ID when source IDs are inconsistent."""
    raw = f"{row.get('title', '')}|{row.get('url', '')}|{row.get('published_at', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def clean_text(value: Any) -> str:
    """Normalize a text value for embedding and reporting."""
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def build_document_text(row: pd.Series) -> str:
    """Combine title and body into one semantic document."""
    title = clean_text(row.get("title", ""))
    body = clean_text(row.get("body", ""))
    return f"{title}. {body}".strip()


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create discovery tables if they do not already exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS topic_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            since_date TEXT NOT NULL,
            model_name TEXT NOT NULL,
            total_documents INTEGER NOT NULL,
            clustered_documents INTEGER NOT NULL,
            noise_documents INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS topic_document_embeddings (
            document_id TEXT PRIMARY KEY,
            source_id TEXT,
            title TEXT,
            source TEXT,
            url TEXT,
            published_at TEXT,
            model_name TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS topic_cluster_assignments (
            run_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            source TEXT,
            title TEXT,
            url TEXT,
            published_at TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, document_id)
        );

        CREATE TABLE IF NOT EXISTS topic_cluster_summaries (
            run_id TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            document_count INTEGER NOT NULL,
            source_count INTEGER NOT NULL,
            sources_json TEXT NOT NULL,
            top_terms_json TEXT NOT NULL,
            representative_titles_json TEXT NOT NULL,
            convergence_score REAL NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, cluster_id)
        );
        """
    )
    conn.commit()


def load_records(conn: sqlite3.Connection, since_date: str) -> pd.DataFrame:
    """Load source records from the Chaos Observatory database."""
    df = pd.DataFrame()
    for query in SOURCE_QUERIES:
        try:
            df = pd.read_sql_query(query, conn, params=(since_date,))
            break
        except pd.errors.DatabaseError:
            continue

    if df.empty:
        return df

    df["document_id"] = df["id"].fillna("").astype(str)
    missing_ids = df["document_id"].str.strip() == ""
    df.loc[missing_ids, "document_id"] = df[missing_ids].apply(stable_text_id, axis=1)
    df["text"] = df.apply(build_document_text, axis=1)
    df = df[df["text"].str.len() >= 40].copy()
    df["source"] = df["source"].fillna("unknown").replace("", "unknown")

    return df.drop_duplicates(subset=["document_id"])


def text_hash(text: str) -> str:
    """Hash text so changed articles can be re-embedded later."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_cached_embedding(
    conn: sqlite3.Connection,
    document_id: str,
    model_name: str,
    current_hash: str,
) -> np.ndarray | None:
    """Return cached embedding if text and model match."""
    row = conn.execute(
        """
        SELECT embedding_json
        FROM topic_document_embeddings
        WHERE document_id = ?
          AND model_name = ?
          AND text_hash = ?
        """,
        (document_id, model_name, current_hash),
    ).fetchone()

    if not row:
        return None

    return np.array(json.loads(row[0]), dtype=np.float32)


def cache_embedding(
    conn: sqlite3.Connection,
    row: pd.Series,
    model_name: str,
    current_hash: str,
    embedding: np.ndarray,
) -> None:
    """Store an embedding in SQLite."""
    conn.execute(
        """
        INSERT OR REPLACE INTO topic_document_embeddings (
            document_id,
            source_id,
            title,
            source,
            url,
            published_at,
            model_name,
            text_hash,
            embedding_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["document_id"],
            str(row.get("id", "")),
            clean_text(row.get("title", "")),
            clean_text(row.get("source", "unknown")),
            clean_text(row.get("url", "")),
            clean_text(row.get("published_at", "")),
            model_name,
            current_hash,
            json.dumps(embedding.astype(float).tolist()),
            utc_now(),
        ),
    )


def embed_documents(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """Create or load embeddings for each document."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for ML topic convergence. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings: list[np.ndarray | None] = []
    missing_indexes: list[int] = []
    missing_texts: list[str] = []

    for idx, row in df.iterrows():
        current_hash = text_hash(row["text"])
        cached = get_cached_embedding(conn, row["document_id"], model_name, current_hash)

        if cached is not None:
            embeddings.append(cached)
        else:
            embeddings.append(None)
            missing_indexes.append(len(embeddings) - 1)
            missing_texts.append(row["text"])

    if missing_texts:
        new_vectors = model.encode(
            missing_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        for local_idx, vector in zip(missing_indexes, new_vectors):
            embeddings[local_idx] = np.array(vector, dtype=np.float32)

    for row_position, (_, row) in enumerate(df.iterrows()):
        current_hash = text_hash(row["text"])
        cache_embedding(conn, row, model_name, current_hash, embeddings[row_position])

    conn.commit()

    return np.vstack(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
) -> np.ndarray:
    """Cluster embeddings with HDBSCAN."""
    if len(embeddings) < min_cluster_size:
        return np.full(shape=(len(embeddings),), fill_value=-1, dtype=int)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )

    return clusterer.fit_predict(embeddings)


def top_terms_for_cluster(texts: list[str], max_terms: int = 8) -> list[str]:
    """Extract simple TF-IDF terms to label a cluster."""
    if not texts:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=(1, 2),
    )

    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    top_indexes = scores.argsort()[::-1][:max_terms]

    return [str(terms[i]) for i in top_indexes if scores[i] > 0]


def convergence_score(document_count: int, source_count: int) -> float:
    """
    Calculate a simple convergence score.

    Higher score means:
    - more documents
    - more source diversity
    """
    if document_count <= 0:
        return 0.0

    return round(float(np.log1p(document_count) * source_count), 3)


def summarize_clusters(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Create summary rows for each non-noise cluster."""
    summaries: list[dict[str, Any]] = []

    clustered = df[df["cluster_id"] >= 0].copy()

    for cluster_id, group in clustered.groupby("cluster_id"):
        texts = group["text"].tolist()
        terms = top_terms_for_cluster(texts)
        sources = sorted(group["source"].dropna().unique().tolist())
        titles = group["title"].dropna().head(5).tolist()

        label = ", ".join(terms[:4]) if terms else f"cluster-{cluster_id}"

        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "label": label,
                "document_count": int(len(group)),
                "source_count": int(len(sources)),
                "sources": sources,
                "top_terms": terms,
                "representative_titles": titles,
                "convergence_score": convergence_score(len(group), len(sources)),
            }
        )

    return sorted(
        summaries,
        key=lambda item: (
            item["convergence_score"],
            item["source_count"],
            item["document_count"],
        ),
        reverse=True,
    )


def save_run(
    conn: sqlite3.Connection,
    run_id: str,
    since_date: str,
    model_name: str,
    df: pd.DataFrame,
    summaries: list[dict[str, Any]],
) -> None:
    """Persist run metadata, assignments, and cluster summaries."""
    total_documents = len(df)
    noise_documents = int((df["cluster_id"] == -1).sum())
    clustered_documents = total_documents - noise_documents

    conn.execute(
        """
        INSERT OR REPLACE INTO topic_runs (
            run_id,
            created_at,
            since_date,
            model_name,
            total_documents,
            clustered_documents,
            noise_documents
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            utc_now(),
            since_date,
            model_name,
            total_documents,
            clustered_documents,
            noise_documents,
        ),
    )

    for _, row in df.iterrows():
        conn.execute(
            """
            INSERT OR REPLACE INTO topic_cluster_assignments (
                run_id,
                document_id,
                cluster_id,
                source,
                title,
                url,
                published_at,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                row["document_id"],
                int(row["cluster_id"]),
                clean_text(row.get("source", "")),
                clean_text(row.get("title", "")),
                clean_text(row.get("url", "")),
                clean_text(row.get("published_at", "")),
                utc_now(),
            ),
        )

    for item in summaries:
        conn.execute(
            """
            INSERT OR REPLACE INTO topic_cluster_summaries (
                run_id,
                cluster_id,
                label,
                document_count,
                source_count,
                sources_json,
                top_terms_json,
                representative_titles_json,
                convergence_score,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                item["cluster_id"],
                item["label"],
                item["document_count"],
                item["source_count"],
                json.dumps(item["sources"]),
                json.dumps(item["top_terms"]),
                json.dumps(item["representative_titles"]),
                item["convergence_score"],
                utc_now(),
            ),
        )

    conn.commit()


def write_markdown_report(
    output_path: Path,
    run_id: str,
    since_date: str,
    df: pd.DataFrame,
    summaries: list[dict[str, Any]],
) -> None:
    """Write a human-readable convergence report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    noise_count = int((df["cluster_id"] == -1).sum())
    clustered_count = len(df) - noise_count

    lines: list[str] = [
        "# Topic Discovery & Convergence Report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Since date: `{since_date}`",
        f"- Total documents: **{len(df)}**",
        f"- Clustered documents: **{clustered_count}**",
        f"- Noise/unclustered documents: **{noise_count}**",
        "",
        "## Highest Convergence Clusters",
        "",
    ]

    if not summaries:
        lines.append("No stable clusters found for this run.")
    else:
        for item in summaries[:20]:
            lines.extend(
                [
                    f"### Cluster {item['cluster_id']}: {item['label']}",
                    "",
                    f"- Documents: **{item['document_count']}**",
                    f"- Source count: **{item['source_count']}**",
                    f"- Convergence score: **{item['convergence_score']}**",
                    f"- Sources: {', '.join(item['sources'])}",
                    f"- Top terms: {', '.join(item['top_terms'])}",
                    "",
                    "**Representative titles:**",
                    "",
                ]
            )

            for title in item["representative_titles"]:
                lines.append(f"- {title}")

            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    """Main execution flow."""
    db_path = Path(args.db)
    report_path = Path(args.output)

    run_id = dt.datetime.now(dt.UTC).strftime("topic-convergence-%Y%m%dT%H%M%SZ")

    with sqlite3.connect(db_path) as conn:
        df = load_records(conn, args.since)

        if df.empty:
            print(f"No records found since {args.since}.")
            return

        ensure_tables(conn)

        embeddings = embed_documents(
            conn=conn,
            df=df,
            model_name=args.model,
            batch_size=args.batch_size,
        )

        labels = cluster_embeddings(
            embeddings=embeddings,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )

        df["cluster_id"] = labels

        summaries = summarize_clusters(df)

        save_run(
            conn=conn,
            run_id=run_id,
            since_date=args.since,
            model_name=args.model,
            df=df,
            summaries=summaries,
        )

        write_markdown_report(
            output_path=report_path,
            run_id=run_id,
            since_date=args.since,
            df=df,
            summaries=summaries,
        )

    print(f"Wrote report: {report_path}")
    print(f"Run ID: {run_id}")
    print(f"Documents: {len(df)}")
    print(f"Clusters: {len(summaries)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run topic discovery and convergence monitoring."
    )

    parser.add_argument(
        "--db",
        default="storage/chaos.db",
        help="Path to Chaos Observatory SQLite database.",
    )

    parser.add_argument(
        "--since",
        default=(dt.date.today() - dt.timedelta(days=1)).isoformat(),
        help="Only process records since this date. Format: YYYY-MM-DD.",
    )

    parser.add_argument(
        "--output",
        default=f"reports/topic_convergence_{dt.date.today().isoformat()}.md",
        help="Markdown report output path.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence Transformer model name.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=4,
        help="Minimum HDBSCAN cluster size.",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples. Defaults to same behavior as min_cluster_size.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
