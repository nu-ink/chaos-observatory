from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

# Prefer native FAISS if available; otherwise provide a light-weight numpy fallback
try:  # pragma: no cover - environment specific
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:
    faiss = None  # type: ignore
    _HAVE_FAISS = False


class _NumpyIndex:
    """A minimal FAISS-like index using numpy (inner-product search).

    This provides only the subset of API used by the project: `add`, `ntotal`,
    `search`, and `reconstruct`.
    """

    def __init__(self, dim: int) -> None:
        self._dim = int(dim)
        self._vectors = np.zeros((0, self._dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError("Vectors must be 2D with shape (n, dim)")
        self._vectors = np.vstack([self._vectors, arr])

    def search(self, query: np.ndarray, k: int):
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self._dim:
            raise ValueError("Query vector has wrong dimension")

        if self.ntotal == 0 or k == 0:
            return np.empty((q.shape[0], 0), dtype=np.float32), np.empty((q.shape[0], 0), dtype=np.int64)

        # inner-product distances
        scores = np.dot(q, self._vectors.T)
        # get top-k indices
        k = min(k, self.ntotal)
        idx = np.argpartition(-scores, range(k), axis=1)[:, :k]
        # sort selected indices by score
        sorted_idx = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
        final_idx = np.take_along_axis(idx, sorted_idx, axis=1)
        distances = np.take_along_axis(scores, final_idx, axis=1)
        return distances.astype(np.float32), final_idx.astype(np.int64)

    def reconstruct(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.ntotal:
            raise IndexError("Index id out of range")
        return np.asarray(self._vectors[idx], dtype=np.float32)




def create_index(dimension: int):
    if dimension <= 0:
        raise ValueError("dimension must be > 0")
    if _HAVE_FAISS:
        return faiss.IndexFlatIP(dimension)
    return _NumpyIndex(dimension)


def save_index(index, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _HAVE_FAISS:
        faiss.write_index(index, str(output_path))
        return
    # fallback: save numpy vectors if possible
    if hasattr(index, "_vectors"):
        np.savez_compressed(str(output_path), vectors=index._vectors)
        return
    raise NotImplementedError("save_index not implemented for this index type")


def load_index(path: str, dimension: int):
    index_path = Path(path)
    if index_path.exists():
        if _HAVE_FAISS:
            return faiss.read_index(str(index_path))
        # attempt to load numpy fallback
        data = np.load(str(index_path))
        vectors = data.get("vectors")
        idx = create_index(dimension)
        if vectors is not None and vectors.size:
            idx.add(vectors.astype(np.float32))
        return idx
    return create_index(dimension)


def add_vector(index, vector: np.ndarray) -> int:
    vectors = _ensure_2d_float32(vector)
    if vectors.shape[0] != 1:
        raise ValueError("add_vector expects exactly one vector.")
    next_id = index.ntotal
    index.add(vectors)
    return int(next_id)


def add_vectors(index, vectors: np.ndarray) -> list[int]:
    array = _ensure_2d_float32(vectors)
    start = index.ntotal
    index.add(array)
    return list(range(start, start + array.shape[0]))


def search_vectors(index, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if index.ntotal == 0:
        return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
    query = _ensure_2d_float32(query_vector)
    k = min(top_k, index.ntotal)
    return index.search(query, k)


def reconstruct_vector(index, idx: int) -> np.ndarray:
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
