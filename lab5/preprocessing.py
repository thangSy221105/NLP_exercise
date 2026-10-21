"""Stateless tokenizer and train-only vocabulary for LAB 05."""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

SEED=2026
PAD=0
UNK=1
WORD_RE=re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
def clean_text(text):
    text=html.unescape(str(text)).replace("’","'")
    text=re.sub(r"<br\s*/?>"," ",text,flags=re.I)
    text=re.sub(r"<[^>]+>"," ",text).lower()
    text=text.replace("can't","can not").replace("won't","will not")
    return " ".join(re.sub(r"\b([a-z]+)n't\b",r"\1 not",text).split())
def tokenize(text):
    return WORD_RE.findall(clean_text(text))
def load_data(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    path=root/"data"/"imdb_10k.jsonl.gz"
    audit=json.loads((root/"data"/"dataset_audit.json").read_text(encoding="utf-8"))
    if audit["seed"]!=SEED:
        raise ValueError("Dataset seed differs from SEED. Copy the intended Lab 4 subset first.")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=audit["subset_sha256"]:
        raise ValueError("Dataset fingerprint differs from audit.")
    data=pd.read_json(path,lines=True)
    data["tokens"]=data.text.map(tokenize)
    data["length"]=data.tokens.map(len)
    return data,audit
def build_vocabulary(train_tokens,max_size=10000,min_count=2):
    counts=Counter(word for words in train_tokens for word in words)
    words=sorted((w for w in counts if counts[w]>=min_count),key=lambda w:(-counts[w],w))
    return {"<PAD>":PAD,"<UNK>":UNK,**{w:i+2 for i,w in enumerate(words[:max_size-2])}}
def encode_batch(token_lists,vocabulary,max_length):
    ids=np.zeros((len(token_lists),max_length),dtype=np.int64)
    lengths=np.ones(len(token_lists),dtype=np.int64)
    for i,words in enumerate(token_lists):
        values=[vocabulary.get(w,UNK) for w in words[:max_length]] or [UNK]
        lengths[i]=len(values)
        ids[i,:len(values)]=values
    return ids,lengths
def retention(token_lists,vocabulary,max_length):
    lengths=np.array([len(w) for w in token_lists])
    retained=np.minimum(lengths,max_length)
    original_total=int(lengths.sum())
    kept=[w for words in token_lists for w in words[:max_length]]
    return {"max_length":max_length,"tokens_retained_percent":100*retained.sum()/max(original_total,1),
        "mean_review_retained_percent":100*np.mean(retained/np.maximum(lengths,1)),
        "reviews_truncated_percent":100*np.mean(lengths>max_length),
        "padding_percent":100*(1-retained.sum()/(len(lengths)*max_length)),
        "retained_oov_percent":100*sum(w not in vocabulary for w in kept)/max(len(kept),1)}
