"""Small reusable helpers for Lab 4 classification experiments."""
from time import perf_counter
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from preprocessing import tokenize, sentences, stable_hash, SEED

LABELS = {0: "negative", 1: "positive"}

def scores(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {"accuracy": float(accuracy_score(y_true,y_pred)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "macro_f1": float(macro),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def make_lr():
    return LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=SEED)

def fit_tfidf(name, train, validation, max_features=10000, bigrams=False,
              remove_stopwords=False, svm=False):
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
        min_df=2, max_features=max_features, ngram_range=(1,2) if bigrams else (1,1),
        stop_words=sorted(ENGLISH_STOP_WORDS) if remove_stopwords else None,
        sublinear_tf=True, norm="l2", dtype=np.float64)
    start = perf_counter()
    x_train = vectorizer.fit_transform(train.text)
    feature_seconds = perf_counter()-start
    clf = LinearSVC(C=1, max_iter=5000, random_state=SEED, dual="auto") if svm else make_lr()
    start = perf_counter()
    clf.fit(x_train, train.label)
    classifier_seconds = perf_counter()-start
    result = {"name": name, "representation": "TF-IDF", "classifier": "Linear SVM" if svm else "LR",
        "vectorizer": vectorizer, "clf": clf, "x_train": x_train,
        "feature_count": x_train.shape[1], "max_features": max_features,
        "ngrams": "1+2" if bigrams else "1", "stopwords": "removed" if remove_stopwords else "kept",
        "feature_seconds": feature_seconds, "classifier_seconds": classifier_seconds,
        "training_seconds": feature_seconds+classifier_seconds}
    x_val = vectorizer.transform(validation.text)
    result["x_validation"] = x_val
    result["validation"] = scores(validation.label, clf.predict(x_val))
    return result

def average_vectors(texts, model):
    vectors = np.zeros((len(texts), model.vector_size), dtype=np.float64)
    coverage = np.zeros(len(texts))
    for i,text in enumerate(texts):
        words = tokenize(text)
        known = [w for w in words if w in model.wv]
        coverage[i] = len(known)/len(words) if words else 0
        if known:
            vectors[i] = model.wv[known].mean(axis=0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms>0)
    return vectors, coverage

def fit_embeddings(train, validation):
    start = perf_counter()
    corpus = [sentence for text in train.text for sentence in sentences(text)]
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=2,
        sg=0, hs=0, negative=5, sample=1e-3, epochs=10, workers=1,
        seed=SEED, hashfxn=stable_hash, shrink_windows=False)
    x_train, train_coverage = average_vectors(train.text.tolist(), model)
    feature_seconds = perf_counter()-start
    clf = make_lr()
    start = perf_counter()
    clf.fit(x_train,train.label)
    classifier_seconds = perf_counter()-start
    x_val, val_coverage = average_vectors(validation.text.tolist(),model)
    return {"name": "Avg Word2Vec + LR", "representation": "Avg Word2Vec", "classifier": "LR",
        "wv": model, "clf": clf, "x_train": x_train, "x_validation": x_val,
        "train_coverage": train_coverage, "validation_coverage": val_coverage,
        "feature_count": 100, "max_features": None, "ngrams": "N/A",
        "stopwords": "kept", "feature_seconds": feature_seconds,
        "classifier_seconds": classifier_seconds,
        "training_seconds": feature_seconds+classifier_seconds,
        "validation": scores(validation.label,clf.predict(x_val))}

def transform(model, texts):
    if model["representation"] == "TF-IDF":
        return model["vectorizer"].transform(texts)
    return average_vectors(list(texts),model["wv"])[0]

def evaluate_test(model, test):
    x = transform(model,test.text.tolist())
    pred = model["clf"].predict(x)
    model["x_test"], model["test_pred"] = x,pred
    model["test"] = scores(test.label,pred)
    if hasattr(model["clf"],"predict_proba"):
        model["test_probability"] = model["clf"].predict_proba(x)[:,1]
    return model["test"]

def result_row(model, split):
    return {"seed": SEED, "model": model["name"], "split": split, "representation": model["representation"],
        "classifier": model["classifier"], "feature_count": model["feature_count"],
        "vocabulary_size": len(model["wv"].wv) if "wv" in model else model["feature_count"],
        "max_features": model["max_features"], "ngrams": model["ngrams"],
        "stopwords": model["stopwords"], "C": 1.0, "train_samples": 6000,
        "validation_samples": 2000, "test_samples": 2000,
        "feature_fit_seconds": model["feature_seconds"],
        "classifier_fit_seconds": model["classifier_seconds"],
        "training_seconds": model["training_seconds"], **model[split]}

def feature_contributions(model, text, top_k=5):
    vector = model["vectorizer"].transform([text]).tocsr()
    names = model["vectorizer"].get_feature_names_out()
    weights = model["clf"].coef_[0]
    contributions = weights[vector.indices]*vector.data
    order = np.argsort(-np.abs(contributions),kind="stable")[:top_k]
    return pd.DataFrame([{"feature": names[vector.indices[i]],
                          "tfidf": float(vector.data[i]),
                          "weight": float(weights[vector.indices[i]]),
                          "contribution": float(contributions[i])} for i in order])

def snippet(text, tokens, limit=450):
    """Keep nearby evidence; full review is printed separately in error-analysis section."""
    flat = " ".join(text.split())
    lower = flat.lower()
    positions = [lower.find(word) for word in tokens if word and lower.find(word)>=0]
    start = max(0,min(positions)-90) if positions else 0
    return ("..." if start else "")+flat[start:start+limit]+("..." if start+limit<len(flat) else "")

def select_examples(model, test):
    actual = test.label.to_numpy()
    predicted = model["test_pred"]
    selected = []
    # Include both labels and both FP/FN where available.
    for correct in [True,False]:
        candidates = np.flatnonzero((predicted==actual)==correct).tolist()
        chosen = []
        for label in [0,1]:
            pool = [i for i in candidates if actual[i]==label]
            if pool: chosen.append(min(pool,key=lambda i:(test.iloc[i].token_count,i)))
        others = [i for i in candidates if i not in chosen]
        # Prefer an example with negation for the third case, without claiming it caused the error.
        negated = [i for i in others if "not" in tokenize(test.iloc[i].text)]
        pool = negated or others
        if pool: chosen.append(min(pool,key=lambda i:(test.iloc[i].token_count,i)))
        selected += chosen[:3]
    return selected
