"""Minimal TF-IDF implementation for LAB 01.

Complete the TODO sections without using sklearn's TfidfVectorizer for the
core implementation. The reference implementation may be used afterwards
for comparison.
"""

from __future__ import annotations

import math
import unittest
from collections import Counter
from typing import Sequence


Document = Sequence[str]


def build_vocabulary(documents: Sequence[Document]) -> list[str]:
    """Return a deterministic vocabulary shared by all documents."""
    terms = {term for document in documents for term in document}
    return sorted(terms)


def compute_counts(document: Document, vocabulary: Sequence[str]) -> list[int]:
    """Return term counts in vocabulary order."""
    counts = Counter(document)
    return [counts[term] for term in vocabulary]


def compute_tf(counts: Sequence[int]) -> list[float]:
    """Return normalized term frequencies."""
    if any(count < 0 for count in counts):
        raise ValueError("counts must be non-negative")

    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]
    return [count / total for count in counts]


def compute_idf(documents: Sequence[Document], vocabulary: Sequence[str]) -> list[float]:
    """Return unsmoothed IDF: log(N / df)."""
    if not documents:
        raise ValueError("documents must not be empty")

    number_of_documents = len(documents)
    document_sets = [set(document) for document in documents]
    idf: list[float] = []

    for term in vocabulary:
        document_frequency = sum(term in document_terms for document_terms in document_sets)
        if document_frequency == 0:
            raise ValueError(f"term {term!r} does not occur in documents")
        idf.append(math.log(number_of_documents / document_frequency))

    return idf


def compute_tfidf(tf: Sequence[float], idf: Sequence[float]) -> list[float]:
    """Return element-wise TF-IDF."""
    if len(tf) != len(idf):
        raise ValueError("tf and idf must have the same dimension")
    return [tf_value * idf_value for tf_value, idf_value in zip(tf, idf)]


def cosine_similarity(x: Sequence[float], y: Sequence[float]) -> float:
    """Return cosine similarity, handling zero vectors explicitly."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same dimension")

    dot_product = sum(x_value * y_value for x_value, y_value in zip(x, y))
    norm_x = math.sqrt(sum(value * value for value in x))
    norm_y = math.sqrt(sum(value * value for value in y))
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0
    return dot_product / (norm_x * norm_y)


class TestTFIDF(unittest.TestCase):
    """Unit tests for the minimal TF-IDF implementation."""

    documents = [
        ["cat", "eats", "fish"],
        ["dog", "eats", "fish"],
        ["cat", "likes", "fish"],
    ]

    def test_build_vocabulary(self) -> None:
        expected = ["cat", "dog", "eats", "fish", "likes"]
        self.assertEqual(build_vocabulary(self.documents), expected)
        self.assertEqual(build_vocabulary(list(reversed(self.documents))), expected)

    def test_compute_counts(self) -> None:
        vocabulary = ["cat", "dog", "eats", "fish", "likes"]
        self.assertEqual(compute_counts(self.documents[0], vocabulary), [1, 0, 1, 1, 0])
        self.assertEqual(compute_counts(self.documents[1], vocabulary), [0, 1, 1, 1, 0])

    def test_compute_tf(self) -> None:
        tf = compute_tf([1, 0, 1, 1, 0])
        self.assertEqual(tf, [1 / 3, 0.0, 1 / 3, 1 / 3, 0.0])
        self.assertAlmostEqual(sum(tf), 1.0)

        self.assertEqual(compute_tf([0, 0]), [0.0, 0.0])

    def test_compute_idf(self) -> None:
        vocabulary = build_vocabulary(self.documents)
        idf = compute_idf(self.documents, vocabulary)
        expected = [math.log(3 / 2), math.log(3), math.log(3 / 2), 0.0, math.log(3)]
        for actual, expected_value in zip(idf, expected):
            self.assertAlmostEqual(actual, expected_value)
        self.assertEqual(idf[3], 0.0)  # fish occurs in all three documents.

    def test_compute_tfidf(self) -> None:
        self.assertEqual(compute_tfidf([0.5, 0.25], [2.0, 4.0]), [1.0, 1.0])
        with self.assertRaises(ValueError):
            compute_tfidf([1.0], [1.0, 2.0])

    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 1, 1], [1, 1, 1]), 1.0)
        self.assertAlmostEqual(
            cosine_similarity([1, 1, 1], [1, 1, 0]),
            2 / math.sqrt(6),
        )
        self.assertEqual(cosine_similarity([0, 0], [1, 2]), 0.0)
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
