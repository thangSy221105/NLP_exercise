"""Minimal TF-IDF implementation for LAB 01.

Complete the TODO sections without using sklearn's TfidfVectorizer for the
core implementation. The reference implementation may be used afterwards
for comparison.
"""

from __future__ import annotations

import math
import unittest
from collections import Counter
from typing import Iterable, Mapping, Sequence


Document = Sequence[str]


def build_vocabulary(documents: Sequence[Document]) -> list[str]:
    """Return a deterministic vocabulary shared by all documents."""
    raise NotImplementedError


def compute_counts(document: Document, vocabulary: Sequence[str]) -> list[int]:
    """Return term counts in vocabulary order."""
    raise NotImplementedError


def compute_tf(counts: Sequence[int]) -> list[float]:
    """Return normalized term frequencies."""
    raise NotImplementedError


def compute_idf(documents: Sequence[Document], vocabulary: Sequence[str]) -> list[float]:
    """Return unsmoothed IDF: log(N / df)."""
    raise NotImplementedError


def compute_tfidf(tf: Sequence[float], idf: Sequence[float]) -> list[float]:
    """Return element-wise TF-IDF."""
    raise NotImplementedError


def cosine_similarity(x: Sequence[float], y: Sequence[float]) -> float:
    """Return cosine similarity, handling zero vectors explicitly."""
    raise NotImplementedError


class TestTFIDF(unittest.TestCase):
    """Add at least one test for each function as implementation progresses."""

    documents = [
        ["cat", "eats", "fish"],
        ["dog", "eats", "fish"],
        ["cat", "likes", "fish"],
    ]

    def test_build_vocabulary(self) -> None:
        self.skipTest("TODO: implement and enable")

    def test_compute_counts(self) -> None:
        self.skipTest("TODO: implement and enable")

    def test_compute_tf(self) -> None:
        self.skipTest("TODO: implement and enable")

    def test_compute_idf(self) -> None:
        self.skipTest("TODO: implement and enable")

    def test_compute_tfidf(self) -> None:
        self.skipTest("TODO: implement and enable")

    def test_cosine_similarity(self) -> None:
        self.skipTest("TODO: implement and enable")


if __name__ == "__main__":
    unittest.main()

