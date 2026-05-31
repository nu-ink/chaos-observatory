from __future__ import annotations

from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


def create_index(dimension: int) -> faiss.IndexFlatIP:
    if dimension <= 0:
        raise ValueError("dimension must be > 0")
    return faiss.IndexFlatIP(dimension)


def save_index(index: faiss.Index, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))


def load_index(path: str, dimension: int) -> faiss.Index:
    index_path = Path(path)
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return create_index(dimension)


def add_vector(index: faiss.Index, vector: np.ndarray) -> int:
    vectors = _ensure_2d_float32(vector)
    if vectors.shape[0] != 1:
        raise ValueError("add_vector expects exactly one vector.")
    next_id = index.ntotal
    index.add(vectors)
    return int(next_id)


def add_vectors(index: faiss.Index, vectors: np.ndarray) -> list[int]:
    array = _ensure_2d_float32(vectors)
    start = index.ntotal
    index.add(array)
    return list(range(start, start + array.shape[0]))


def search_vectors(index: faiss.Index, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if index.ntotal == 0:
        return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
    query = _ensure_2d_float32(query_vector)
    k = min(top_k, index.ntotal)
    return index.search(query, k)


def reconstruct_vector(index: faiss.Index, idx: int) -> np.ndarray:
    if idx < 0 or idx >= index.ntotal:
        raise IndexError("FAISS index id out of range.")
    return np.asarray(index.reconstruct(idx), dtype=np.float32)


def _ensure_2d_float32(array: np.ndarray | Iterable[float]) -> np.ndarray:
    np_array = np.asarray(array, dtype=np.float32)
    if np_array.ndim == 1:
        np_array = np_array.reshape(1, -1)
    if np_array.ndim != 2:
        raise ValueError("Expected a 1D or 2D array.")
    return np_array
