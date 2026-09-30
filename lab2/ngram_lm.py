"""From-scratch unigram, bigram and trigram language models."""

import math
import re
from collections import Counter
from heapq import nlargest

UNK = "<unk>"
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)*")


def tokenize_sentence(text):
    return WORD_RE.findall(text.lower())


def document_to_sentences(text):
    return [tokens for part in SENTENCE_RE.split(text.lower().replace("\r", "\n"))
            if (tokens := tokenize_sentence(part))]


def tokens_of(sentence):
    if isinstance(sentence, str):
        return tokenize_sentence(sentence)
    return [str(word).lower() for word in sentence]


def build_vocabulary(corpus, min_count=1, include_special_tokens=True):
    counts = Counter(word for sentence in corpus for word in tokens_of(sentence))
    words = {word for word, count in counts.items() if count >= min_count and word != UNK}
    if include_special_tokens:
        words.add(UNK)
    return sorted(words)


def count_ngrams(tokens, n):
    words = tokens_of(tokens)
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def train_ngrams(corpus, n):
    counts = Counter()
    for sentence in corpus:
        words = tokens_of(sentence)
        for i in range(len(words) - n + 1):
            counts[tuple(words[i:i + n])] += 1
    return counts


def train_unigram(corpus):
    return train_ngrams(corpus, 1)


def train_bigram(corpus):
    return train_ngrams(corpus, 2)


def train_trigram(corpus):
    return train_ngrams(corpus, 3)

class NGramLanguageModel:
    def __init__(self, n, smoothing="mle", alpha=1.0, min_count=1):
        if n not in (1, 2, 3):
            raise ValueError("n must be 1, 2 or 3")
        self.n = n
        self.smoothing = smoothing.lower()
        self.alpha = alpha
        self.min_count = min_count

    def _encode(self, ids):
        key = 0
        for token_id in ids:
            key = key * self.base + token_id
        return key

    def fit(self, corpus):
        sentences = [tokens_of(sentence) for sentence in corpus]
        self.vocabulary = build_vocabulary(sentences, self.min_count)
        self.word_id = {word: i for i, word in enumerate(self.vocabulary)}
        self.unk_id = self.word_id[UNK]
        self.base = len(self.vocabulary)
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.bigram_context_counts = Counter()
        self.trigram_context_counts = Counter()
        self.total_tokens = 0

        for sentence in sentences:
            ids = [self.word_id.get(word, self.unk_id) for word in sentence]
            self.unigram_counts.update(ids)
            self.total_tokens += len(ids)
            if self.n >= 2:
                for i in range(len(ids) - 1):
                    self.bigram_counts[self._encode(ids[i:i + 2])] += 1
                    self.bigram_context_counts[ids[i]] += 1
            if self.n >= 3:
                for i in range(len(ids) - 2):
                    self.trigram_counts[self._encode(ids[i:i + 3])] += 1
                    self.trigram_context_counts[self._encode(ids[i:i + 2])] += 1

        self.ngram_counts = {
            1: self.unigram_counts,
            2: self.bigram_counts,
            3: self.trigram_counts,
        }[self.n]
        return self

    def _context_ids(self, context):
        words = tokens_of(context)
        ids = [self.word_id.get(word, self.unk_id) for word in words]
        return ids[-(self.n - 1):] if self.n > 1 else []

    def probability(self, context, word, smoothing=None):
        mode = (smoothing or self.smoothing).lower()
        context = self._context_ids(context)
        word_id = self.word_id.get(str(word).lower(), self.unk_id)
        vocabulary_size = len(self.vocabulary)

        if self.n == 1 or not context:
            count = self.unigram_counts[word_id]
            denominator = self.total_tokens
        elif self.n == 2 or len(context) == 1:
            previous = context[-1]
            count = self.bigram_counts[self._encode([previous, word_id])]
            denominator = (self.unigram_counts[previous] if mode == "laplace"
                           else self.bigram_context_counts[previous])
        else:
            history = context[-2:]
            count = self.trigram_counts[self._encode(history + [word_id])]
            denominator = self.trigram_context_counts[self._encode(history)]

        if mode == "mle":
            return count / denominator if denominator else 0.0
        if mode == "laplace":
            return (count + self.alpha) / (denominator + self.alpha * vocabulary_size)
        raise ValueError("smoothing must be 'mle' or 'laplace'")

    def sentence_log_probability(self, sentence, smoothing=None):
        words = tokens_of(sentence)
        score = 0.0
        for i, word in enumerate(words):
            probability = self.probability(words[max(0, i - self.n + 1):i], word, smoothing)
            if probability == 0:
                return -math.inf
            score += math.log(probability)
        return score

    def sentence_probability(self, sentence, smoothing=None):
        score = self.sentence_log_probability(sentence, smoothing)
        return 0.0 if score == -math.inf else math.exp(score)

    def perplexity(self, corpus, smoothing=None):
        total_log, total_tokens = 0.0, 0
        for sentence in corpus:
            words = tokens_of(sentence)
            score = self.sentence_log_probability(words, smoothing)
            if score == -math.inf:
                return math.inf
            total_log += score
            total_tokens += len(words)
        return math.exp(-total_log / total_tokens) if total_tokens else math.nan

    def next_word_distribution(self, context, smoothing=None):
        words = ((word, self.probability(context, word, smoothing))
                 for word in self.vocabulary)
        return dict(sorted(((word, p) for word, p in words if p > 0),
                           key=lambda item: (-item[1], item[0])))

    def count_summary(self, top_k=5000):
        counts = self.ngram_counts.values()
        return {
            "unique_ngrams": len(self.ngram_counts),
            "singletons": sum(count == 1 for count in counts),
            "top_frequencies": nlargest(top_k, self.ngram_counts.values()),
        }

    def ngram_coverage(self, corpus):
        total = unseen = 0
        for sentence in corpus:
            ids = [self.word_id.get(word, self.unk_id) for word in tokens_of(sentence)]
            for i in range(len(ids) - self.n + 1):
                total += 1
                if self._encode(ids[i:i + self.n]) not in self.ngram_counts:
                    unseen += 1
        return {"total_ngrams": total, "unseen_ngrams": unseen,
                "unseen_rate": unseen / total if total else 0.0}