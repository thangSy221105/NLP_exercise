"""Starter scaffold for LAB 02 n-gram language models.

Input convention for this scaffold: a corpus is a sequence of tokenized
sentences. Adapt the data loader/preprocessing to the course corpus and record
your choices in README.md. Implement the TODOs yourself.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

Token = str
Sentence = Sequence[Token]
Corpus = Sequence[Sentence]
Ngram = tuple[Token, ...]


def build_vocabulary(corpus: Corpus) -> list[Token]:
    """Build and return the model vocabulary from a tokenized corpus."""
    raise NotImplementedError("TODO: implement vocabulary construction")


def count_ngrams(tokens: Sequence[Token], n: int) -> Counter[Ngram]:
    """Count contiguous n-grams in a token sequence."""
    raise NotImplementedError("TODO: implement n-gram counting")


def train_unigram(corpus: Corpus) -> Counter[Ngram]:
    """Return unigram counts for the corpus."""
    raise NotImplementedError("TODO: implement unigram counting")


def train_bigram(corpus: Corpus) -> Counter[Ngram]:
    """Return bigram counts for the corpus."""
    raise NotImplementedError("TODO: implement bigram counting")


def train_trigram(corpus: Corpus) -> Counter[Ngram]:
    """Return trigram counts for the corpus."""
    raise NotImplementedError("TODO: implement trigram counting")


class NGramLanguageModel:
    """Starter API for unigram, bigram, and trigram language models."""

    def __init__(self, n: int, smoothing: str = "mle") -> None:
        """Store model order and smoothing choice; validate supported values."""
        raise NotImplementedError("TODO: implement model initialization")

    def fit(self, corpus: Corpus) -> NGramLanguageModel:
        """Build vocabulary and counts from the training corpus."""
        raise NotImplementedError("TODO: implement model fitting")

    def probability(self, context: Sequence[Token], word: Token) -> float:
        """Return P(word | context) under the selected model/smoothing."""
        raise NotImplementedError("TODO: implement conditional probability")

    def sentence_probability(self, sentence: Sentence) -> float:
        """Return the probability assigned to a sentence."""
        raise NotImplementedError("TODO: implement sentence probability")

    def sentence_log_probability(self, sentence: Sentence) -> float:
        """Return log probability to avoid multiplying tiny values directly."""
        raise NotImplementedError("TODO: implement sentence log probability")

    def next_word_distribution(self, context: Sequence[Token]) -> dict[Token, float]:
        """Return candidate next words and their probabilities."""
        raise NotImplementedError("TODO: implement next-word distribution")
