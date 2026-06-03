from __future__ import annotations

import numpy as np

from ml.vector_store import add_vector, create_index, search_vectors


def test_vector_store_add_and_search() -> None:
    index = create_index(3)
    added_id = add_vector(index, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    distances, ids = search_vectors(
        index, np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=5
    )

    assert added_id == 0
    assert ids.tolist() == [[0]]
    assert distances.tolist() == [[1.0]]
