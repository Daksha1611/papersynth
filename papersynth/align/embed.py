"""Embeddings for candidate generation.

Vectors are used only to propose merges, never to decide them (DD-05). The
default embedder is a dependency-free hashing model so the suite stays fast,
offline, and deterministic; sentence-transformers is used when installed.

Determinism matters here beyond convenience: NFR-02 requires identical inputs
to produce identical specs, and an embedder whose output drifts between runs
would silently reshuffle clusters.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic bag-of-character-ngrams projection.

    Not competitive with a trained model on semantics, but it is exactly
    reproducible, needs no download, and is sufficient for the job it actually
    does here - proposing candidates that a stricter check then confirms.
    """

    def __init__(self, dim: int = 256, ngram: int = 3) -> None:
        self.dim = dim
        self.ngram = ngram

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            self._add(vector, f"w:{token}", 1.0)
            padded = f" {token} "
            for i in range(len(padded) - self.ngram + 1):
                self._add(vector, padded[i : i + self.ngram], 0.5)

        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def _add(self, vector: list[float], key: str, weight: float) -> None:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * weight


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers when the optional extra is installed."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


def build_embedder(model_name: str | None = None) -> Embedder:
    """Prefer the configured model; fall back to hashing if unavailable.

    Falling back rather than raising keeps the pipeline runnable on a machine
    without the optional extra installed - alignment degrades in quality but
    the run completes and says nothing false.
    """
    if not model_name:
        return HashEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except (ImportError, OSError, ValueError):
        return HashEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)
