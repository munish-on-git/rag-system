"""
src/chunking/semantic_chunker.py

Three chunking strategies over data/processed/passages.jsonl, each written
to its own file in data/chunks/ so you can compare/evaluate them separately
in retrieval_metrics.py and show real thought behind the choice, not one
naive fixed-size splitter.

Strategies:
  1. passage_baseline   - 1 passage = 1 chunk, no splitting. The naive
                           control group -- included on purpose, so the
                           other two strategies have something to be
                           measured against.
  2. sentence_window     - splits on sentence boundaries (never mid-sentence),
                           groups sentences into ~max_tokens windows with
                           overlap_sentences carried into the next window.
  3. metadata_aware      - passage-level chunks (like #1) but enriched with
                           extracted keyword metadata and language tag, so
                           retrieval can filter/boost before the vector
                           search even runs.

Usage:
    python src/chunking/semantic_chunker.py
"""

import json
import re
from collections import Counter
from pathlib import Path

IN_PATH = Path("data/processed/passages.jsonl")
OUT_DIR = Path("data/chunks")

# Sentence-ending punctuation across Latin + common Indic scripts
# (। is the Devanagari/Bengali/etc. purna viram, ॥ its doubled form)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")

# Cheap stopword-ish filter so keyword extraction isn't just "the/is/a" or
# their Hindi/Indic function-word equivalents -- not exhaustive, good enough
# for retrieval-boosting metadata, not for linguistic correctness.
MIN_TOKEN_LEN = 4


def load_passages():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def split_sentences(text: str):
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences if sentences else [text]


def approx_token_count(text: str) -> int:
    return len(text.split())


# --- Strategy 1: passage baseline -------------------------------------------

def chunk_passage_baseline(passage: dict):
    yield {
        "chunk_id": f"{passage['passage_id']}_p0",
        "text": passage["text"],
        "strategy": "passage_baseline",
        "lang": passage["lang"],
        "source_passage_id": passage["passage_id"],
        "chunk_index": 0,
    }


# --- Strategy 2: sentence window with overlap -------------------------------

def chunk_sentence_window(passage: dict, max_tokens: int = 60, overlap_sentences: int = 1):
    sentences = split_sentences(passage["text"])
    windows, current, current_tokens = [], [], 0

    for sent in sentences:
        sent_tokens = approx_token_count(sent)
        if current and current_tokens + sent_tokens > max_tokens:
            windows.append(current)
            # carry the last N sentences into the next window as overlap
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = sum(approx_token_count(s) for s in current)
        current.append(sent)
        current_tokens += sent_tokens
    if current:
        windows.append(current)

    for i, window_sentences in enumerate(windows):
        yield {
            "chunk_id": f"{passage['passage_id']}_w{i}",
            "text": " ".join(window_sentences),
            "strategy": "sentence_window",
            "lang": passage["lang"],
            "source_passage_id": passage["passage_id"],
            "chunk_index": i,
            "num_windows_in_passage": len(windows),
        }


# --- Strategy 3: metadata-aware ---------------------------------------------

def extract_keywords(text: str, top_k: int = 5):
    tokens = re.findall(r"\w+", text)
    candidates = [t for t in tokens if len(t) >= MIN_TOKEN_LEN]
    counts = Counter(candidates)
    return [word for word, _ in counts.most_common(top_k)]


def chunk_metadata_aware(passage: dict):
    keywords = extract_keywords(passage["text"])
    yield {
        "chunk_id": f"{passage['passage_id']}_m0",
        "text": passage["text"],
        "strategy": "metadata_aware",
        "lang": passage["lang"],
        "source_passage_id": passage["passage_id"],
        "chunk_index": 0,
        "keywords": keywords,
        "token_count": approx_token_count(passage["text"]),
    }


STRATEGIES = {
    "passage_baseline": chunk_passage_baseline,
    "sentence_window": chunk_sentence_window,
    "metadata_aware": chunk_metadata_aware,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    passages = list(load_passages())
    print(f"Loaded {len(passages)} passages from {IN_PATH}")

    for strategy_name, chunk_fn in STRATEGIES.items():
        out_path = OUT_DIR / f"{strategy_name}.jsonl"
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for passage in passages:
                for chunk in chunk_fn(passage):
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    count += 1
        print(f"[{strategy_name}] wrote {count} chunks -> {out_path}")

    print("\nNext: src/embedding/encode.py, run once per strategy file, "
          "then compare retrieval quality in src/eval/retrieval_metrics.py "
          "to justify which strategy you ship with.")


if __name__ == "__main__":
    main()