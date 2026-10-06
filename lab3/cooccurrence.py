"""From-scratch word-context counts and cosine similarity for LAB 03."""
import re
from collections import Counter
import numpy as np
from scipy import sparse

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)*")


def tokenize_sentence(text):
    return WORD_RE.findall(text.lower())


def document_to_sentences(text):
    return [tokens for part in SENTENCE_RE.split(text.replace("\r", "\n"))
            if (tokens := tokenize_sentence(part))]


def _tokens(sentence):
    return tokenize_sentence(sentence) if isinstance(sentence, str) else [
        str(word).lower() for word in sentence]


def build_vocabulary(corpus, min_count=1, max_size=None):
    """Select by frequency (alphabetical ties), then return sorted words."""
    if min_count < 1 or (max_size is not None and max_size < 1):
        raise ValueError("min_count and max_size must be positive")
    counts = Counter(word for sentence in corpus for word in _tokens(sentence))
    words = sorted((word for word in counts if counts[word] >= min_count),
                   key=lambda word: (-counts[word], word))
    if max_size is not None:
        words = words[:max_size]
    return sorted(words)


def build_cooccurrence_matrix(corpus, vocabulary, window=1):
    """Fixed-radius symmetric window; preserve OOV positions and sentence bounds."""
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("vocabulary contains duplicate words")
    word_id = {word: i for i, word in enumerate(vocabulary)}
    size = len(vocabulary)
    counts = Counter()
    for sentence in corpus:
        ids = [word_id.get(word, -1) for word in _tokens(sentence)]
        for i, target in enumerate(ids):
            if target < 0:
                continue
            for j in range(max(0, i-window), min(len(ids), i+window+1)):
                context = ids[j]
                if i != j and context >= 0:
                    counts[target*size+context] += 1
    if not counts:
        return sparse.csr_matrix((size, size), dtype=np.int64)
    keys = np.fromiter(counts, dtype=np.int64)
    values = np.fromiter(counts.values(), dtype=np.int64)
    return sparse.csr_matrix((values, (keys//size, keys % size)),
                             shape=(size, size))


def cosine_similarity(x, y):
    """Return zero when either vector has zero norm."""
    if sparse.issparse(x) or sparse.issparse(y):
        x = sparse.csr_matrix(x, dtype=float).reshape((1, -1))
        y = sparse.csr_matrix(y, dtype=float).reshape((1, -1))
        if x.shape != y.shape:
            raise ValueError("vectors must have the same length")
        denominator = np.sqrt(x.multiply(x).sum()*y.multiply(y).sum())
        value = (x @ y.T)[0, 0]
    else:
        x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError("vectors must have the same length")
        denominator = np.linalg.norm(x)*np.linalg.norm(y)
        value = np.dot(x, y)
    return float(np.clip(value/denominator, -1, 1)) if denominator else 0.0


def most_similar(word, matrix, vocabulary, top_k=5):
    """Exclude query and zero vectors; break ties alphabetically."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    word_id = {token: i for i, token in enumerate(vocabulary)}
    if word not in word_id:
        raise KeyError(f"Word outside vocabulary: {word}")
    if matrix.shape[0] != len(vocabulary):
        raise ValueError("matrix rows must match vocabulary")
    i = word_id[word]
    if sparse.issparse(matrix):
        matrix = matrix.astype(float)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        products = (matrix @ matrix.getrow(i).T).toarray().ravel()
    else:
        matrix = np.asarray(matrix, dtype=float)
        norms = np.linalg.norm(matrix, axis=1)
        products = matrix @ matrix[i]
    denominator = norms*norms[i]
    scores = np.divide(products, denominator, out=np.zeros_like(products),
                       where=denominator > 0)
    candidates = [(token, float(np.clip(scores[j], -1, 1)))
                  for j, token in enumerate(vocabulary)
                  if j != i and norms[j] > 0 and scores[j] > 0]
    return sorted(candidates, key=lambda item: (-item[1], item[0]))[:top_k]
