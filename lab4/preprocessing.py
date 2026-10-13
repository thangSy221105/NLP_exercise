"""Load IMDb, normalize text, remove duplicates, and create reproducible splits."""
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import tarfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 2026
DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")

def clean_text(text):
    """Stateless normalization; keep negation, punctuation boundaries and word order."""
    text = html.unescape(str(text)).replace("’", "'")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = text.replace("can't", "can not").replace("won't", "will not")
    text = re.sub(r"\b([a-z]+)n't\b", r"\1 not", text)
    return " ".join(text.split())

def tokenize(text):
    return WORD_RE.findall(clean_text(text))

def sentences(text):
    return [tokens for part in re.split(r"[.!?]+", clean_text(text))
            if (tokens := WORD_RE.findall(part))]

def text_hash(text):
    return hashlib.sha256(clean_text(text).encode("utf-8")).hexdigest()

def stable_hash(word):
    return int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8], "little")

def prepare_dataset(folder=None):
    folder = Path(folder or Path(__file__).resolve().parent) / "data"
    folder.mkdir(parents=True, exist_ok=True)
    cached, audit_file = folder / "imdb_10k.jsonl.gz", folder / "dataset_audit.json"
    if cached.exists() and audit_file.exists():
        cached_audit = json.loads(audit_file.read_text(encoding="utf-8"))
        if cached_audit.get("seed") == SEED:
            data = pd.read_json(cached, lines=True)
            return data, cached_audit
        print(f"Rebuilding subset: cached seed={cached_audit.get('seed')}, requested seed={SEED}", flush=True)
    archive = folder / "aclImdb_v1.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".tmp")
        subprocess.run(["curl.exe", "--fail", "--location", "--retry", "2",
                        "--output", str(temporary), DATA_URL], check=True)
        temporary.replace(archive)
    rows = []
    # Read only official labeled review members. Never extract archive paths.
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")
            if (member.isfile() and len(parts) == 4 and parts[0] == "aclImdb"
                    and parts[1] in ["train", "test"] and parts[2] in ["pos", "neg"]
                    and parts[3].endswith(".txt")):
                text = tar.extractfile(member).read().decode("utf-8")
                rows.append({"id": member.name, "official_split": parts[1],
                             "text": text, "label": int(parts[2] == "pos")})
    raw = pd.DataFrame(rows).sort_values(["official_split", "id"],
                                         ascending=[False, True]).reset_index(drop=True)
    raw["clean_text"] = raw.text.map(clean_text)
    raw["text_hash"] = raw.text.map(text_hash)
    missing_text = int(raw.clean_text.eq("").sum())
    missing_label = int(raw.label.isna().sum())
    duplicates = int(raw.text.duplicated().sum())
    normalized_duplicates = int(raw.text_hash.duplicated().sum())
    conflict_hashes = raw.groupby("text_hash").label.nunique()
    conflict_hashes = set(conflict_hashes[conflict_hashes > 1].index)
    eligible = raw[raw.clean_text.ne("") & raw.label.notna()
                   & ~raw.text_hash.isin(conflict_hashes)].copy()
    eligible = eligible.drop_duplicates("text_hash", keep="first")
    train_pool = eligible[eligible.official_split.eq("train")]
    test_pool = eligible[eligible.official_split.eq("test")]
    sampled_train = pd.concat([train_pool[train_pool.label.eq(label)].sample(
        n=4000, random_state=SEED) for label in [0, 1]])
    train, val = train_test_split(sampled_train, test_size=2000,
                                  stratify=sampled_train.label, random_state=SEED)
    test = pd.concat([test_pool[test_pool.label.eq(label)].sample(
        n=1000, random_state=SEED) for label in [0, 1]])
    frames = []
    for name, frame in [("train", train), ("validation", val), ("test", test)]:
        frame = frame.sample(frac=1, random_state=SEED).copy()
        frame["split"] = name
        frame["token_count"] = frame.text.map(lambda s: len(tokenize(s)))
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    audit = {
        "source_url": DATA_URL, "seed": SEED, "source_labeled_samples": len(raw),
        "source_classes": 2, "source_missing_text": missing_text,
        "source_missing_labels": missing_label, "source_exact_duplicate_excess": duplicates,
        "source_normalized_duplicate_excess": normalized_duplicates,
        "source_conflicting_hash_groups": len(conflict_hashes),
        "eligible_after_cleaning": len(eligible), "selected_samples": len(data),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "split_policy": "6000 train + 2000 validation from official train; 2000 official test",
        "duplicate_policy": "exclude conflicting normalized hashes, keep one per hash before sampling",
        "selected_normalized_duplicate_excess": int(data.text_hash.duplicated().sum()),
        "counts": data.groupby(["split", "label"]).size().to_dict(),
    }
    audit["counts"] = {f"{split}/{label}": int(n) for (split,label),n in audit["counts"].items()}
    data.to_json(cached, orient="records", lines=True, force_ascii=False, compression="gzip")
    audit["subset_sha256"] = hashlib.sha256(cached.read_bytes()).hexdigest()
    audit_file.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return data, audit
