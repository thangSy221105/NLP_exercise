"""Small utilities shared by the Lab 3 notebook experiments."""
import gzip
import hashlib
import json
from collections import Counter
from time import perf_counter
import numpy as np
from scipy import sparse
from gensim.models import Word2Vec
from cooccurrence import document_to_sentences, tokenize_sentence

STOPWORDS = set("the a an and or is are was were be been to of in on at for from with by as it its this that these those i you he she they we not but have has had can will would also their his her s".split())


def load_corpus(path, limit=10000):
    documents = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if len(documents) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "") if isinstance(record, dict) else ""
            documents.append(document_to_sentences(text if isinstance(text, str) else ""))
    return documents


def stable_hash(word):
    return int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:4], "little")


def train_model(sentences, vector_size=100, window=5, sg=0, epochs=10, min_count=2, seed=42):
    start = perf_counter()
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window,
                     min_count=min_count, epochs=epochs, sg=sg, hs=0, negative=5,
                     sample=1e-3, workers=1, seed=seed, hashfxn=stable_hash,
                     shrink_windows=False)
    return model, perf_counter()-start


def parameter_bytes(model):
    """Float weight arrays only, excluding metadata and norm caches."""
    return model.wv.vectors.nbytes + model.syn1neg.nbytes


def safe_similarity(model, first, second):
    return float(model.wv.similarity(first, second)) if (
        first in model.wv and second in model.wv) else float("nan")


def analogy(model, positive, negative, expected, topn=5):
    missing = [word for word in positive+negative+[expected] if word not in model.wv]
    if missing:
        return {"expected": expected, "status": "OOV: "+", ".join(missing),
                "top1": "", "rank": None, "hit1": None, "hit5": None, "candidates": []}
    candidates = model.wv.most_similar(positive=positive, negative=negative, topn=topn)
    words = [word for word, score in candidates]
    rank = words.index(expected)+1 if expected in words else None
    return {"expected": expected, "status": "OK", "top1": words[0], "rank": rank,
            "hit1": int(words[0] == expected), "hit5": int(expected in words),
            "candidates": [(word, float(score)) for word, score in candidates]}


def context_evidence(sentences, words, window=5):
    profiles = {word: Counter() for word in words}
    samples = {word: [] for word in words}
    for sentence in sentences:
        for i, word in enumerate(sentence):
            if word not in profiles:
                continue
            contexts = sentence[max(0, i-window):i]+sentence[i+1:i+window+1]
            profiles[word].update(w for w in contexts if w not in STOPWORDS)
            snippet = " ".join(sentence[max(0, i-8):i+9])
            if len(samples[word]) < 2 and snippet not in samples[word]:
                samples[word].append(snippet)
    return profiles, samples


def shared_context(first, second, profiles, topn=5):
    x, y = profiles[first], profiles[second]
    shared = sorted(set(x) & set(y), key=lambda w: (-min(x[w], y[w]), w))[:topn]
    return ", ".join(f"{w}({x[w]}/{y[w]})" for w in shared) or "Không có trong window khảo sát"


def document_vectors(documents, model):
    vectors = np.zeros((len(documents), model.vector_size), dtype=np.float32)
    coverage = np.zeros(len(documents))
    for i, document in enumerate(documents):
        words = [w for sentence in document for w in sentence if w not in STOPWORDS]
        known = [w for w in words if w in model.wv]
        coverage[i] = len(known)/len(words) if words else 0
        if known:
            vectors[i] = np.mean(model.wv[known], axis=0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0), coverage


def semantic_scores(query, vectors, model):
    words = [w for w in tokenize_sentence(query) if w not in STOPWORDS]
    known = [w for w in words if w in model.wv]
    if not known:
        return np.zeros(len(vectors)), words
    vector = np.mean(model.wv[known], axis=0)
    norm = np.linalg.norm(vector)
    return (vectors @ (vector/norm) if norm else np.zeros(len(vectors))), [
        word for word in words if word not in model.wv]


def lexical_index(documents, vocabulary):
    """Sublinear TF, smoothed IDF and L2-normalized document vectors."""
    word_id = {word: i for i, word in enumerate(vocabulary)}
    rows, columns, values = [], [], []
    for i, document in enumerate(documents):
        counts = Counter(word for sentence in document for word in sentence
                         if word in word_id and word not in STOPWORDS)
        for word, count in counts.items():
            rows.append(i)
            columns.append(word_id[word])
            values.append(1+np.log(count))
    matrix = sparse.csr_matrix((values, (rows, columns)),
                               shape=(len(documents), len(vocabulary)))
    df = np.asarray((matrix > 0).sum(axis=0)).ravel()
    idf = np.log((1+len(documents))/(1+df))+1
    matrix = matrix.multiply(idf).tocsr()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    matrix = sparse.diags(np.divide(1, norms, out=np.zeros_like(norms), where=norms > 0)) @ matrix
    return matrix.tocsr(), word_id, idf


def lexical_scores(query, matrix, word_id, idf):
    counts = Counter(w for w in tokenize_sentence(query) if w in word_id and w not in STOPWORDS)
    indices = [word_id[w] for w in counts]
    values = [(1+np.log(counts[w]))*idf[word_id[w]] for w in counts]
    vector = sparse.csr_matrix((values, ([0]*len(indices), indices)),
                               shape=(1, matrix.shape[1]))
    norm = np.sqrt(vector.multiply(vector).sum())
    return (matrix @ (vector.T/norm)).toarray().ravel() if norm else np.zeros(matrix.shape[0])


# Artificial retrieval check: topic labels are not human annotations of C4.
DEMO_TEXTS = [
    ("medical", "The physician provides therapy for patients suffering from disease."),
    ("medical", "A hospital offers clinical care and medicine for sick people."),
    ("medical", "Doctors diagnose illness and help patients recover."),
    ("sports", "The soccer team scored a goal and won the game at the stadium."),
    ("sports", "Players compete in a football league tournament."),
    ("sports", "A coach trains athletes for the next match."),
    ("technology", "The laptop runs applications and an operating system."),
    ("technology", "Programmers develop programs and fix hardware problems."),
    ("technology", "Computers use software to process digital information."),
    ("river", "The stream flows beside the shore and through the valley."),
    ("river", "Water moves along the river near trees and rocks."),
    ("river", "People sit on the bank beside the river."),
]
DEMO_QUERIES = [("medical", "medical treatment"), ("sports", "football match"),
                ("technology", "computer software"), ("river", "river bank")]


def retrieval_metrics(model):
    docs = [document_to_sentences(text) for label, text in DEMO_TEXTS]
    vectors, coverage = document_vectors(docs, model)
    labels = [label for label, text in DEMO_TEXTS]
    rows = []
    for label, query in DEMO_QUERIES:
        scores, missing = semantic_scores(query, vectors, model)
        if missing:
            rows.append({"Query": query, "MRR": np.nan, "nDCG@3": np.nan,
                         "P@3": np.nan, "OOV": ", ".join(missing)})
            continue
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        relevant_ranks = [rank for rank, i in enumerate(order, 1) if labels[i] == label]
        gains = np.array([labels[i] == label for i in order[:3]], dtype=float)
        discounts = 1/np.log2(np.arange(2, 5))
        rows.append({"Query": query, "MRR": 1/relevant_ranks[0],
                     "nDCG@3": float(np.dot(gains, discounts)/discounts.sum()),
                     "P@3": float(gains.mean()), "OOV": ""})
    return rows
