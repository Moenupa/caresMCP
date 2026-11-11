"""Utilities for comparing embeddings produced by the same encoder."""

import numpy as np


def _normalize(embedding: list[float]) -> np.ndarray:
    """Convert a nonzero, finite embedding to a unit vector."""
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Embeddings must be nonempty one-dimensional vectors")
    if not np.isfinite(vector).all():
        raise ValueError("Embeddings must contain only finite values")
    scale = np.max(np.abs(vector))
    if scale == 0:
        raise ValueError("Cannot compare zero-length embedding vectors")
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Compute cosine similarity between two compatible embeddings."""
    first_vector = _normalize(first)
    second_vector = _normalize(second)
    if first_vector.shape != second_vector.shape:
        raise ValueError("Embedding dimensions did not match")
    return float(np.clip(first_vector @ second_vector, -1.0, 1.0))


def rank_embeddings(
    reference: list[float], candidates: list[list[float]]
) -> list[tuple[int, float]]:
    """Rank candidates by cosine similarity, returning indices and scores.

    Candidates with equal scores retain their original order. All embeddings
    must be finite, nonzero vectors from the same embedding space.
    """
    reference_vector = _normalize(reference)
    scores = []
    for index, candidate in enumerate(candidates):
        candidate_vector = _normalize(candidate)
        if candidate_vector.shape != reference_vector.shape:
            raise ValueError("Embedding dimensions must match")
        score = float(np.clip(reference_vector @ candidate_vector, -1.0, 1.0))
        scores.append((index, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [(index, score) for index, score in scores]
