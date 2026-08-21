"""
src/indexing/vector_store.py

Builds one FAISS index per chunking strategy from the embeddings encode.py
produced, and provides a VectorStore class the retriever agent uses at
query time. In-process FAISS on purpose -- no network hop to a hosted
vector DB, which matters directly for the <200ms retrieval target.

Build usage:
    python src/indexing/vector_store.py

Query-time usage (imported elsewhere):
    from src.indexing.vector_store import VectorStore
    store = VectorStore("sentence_window")
    results = store.search(query_embedding, top_k=5)
"""

import os
# Must be set before faiss is imported -- FAISS and PyTorch both grab OpenMP
# threads on macOS; uncapped, this segfaults with no Python traceback.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import faiss
import numpy as np

EMBEDDINGS_DIR = Path("data/chunks/embeddings")
INDEX_DIR = Path("data/index")


def build_index(strategy: str):
    emb_path = EMBEDDINGS_DIR / f"{strategy}.npy"
    meta_path = EMBEDDINGS_DIR / f"{strategy}_meta.jsonl"
    if not emb_path.exists():
        print(f"skip: {emb_path} not found (run encode.py first)")
        return

    embeddings = np.load(emb_path)
    dim = embeddings.shape[1]

    # IndexFlatIP on normalized vectors == cosine similarity.
    # Exact search, no approximation -- fine at this corpus size and keeps
    # retrieval quality comparisons across strategies apples-to-apples.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / f"{strategy}.faiss"))

    # metadata stays row-aligned with the index (row i == FAISS id i)
    with open(meta_path) as src, open(INDEX_DIR / f"{strategy}_meta.jsonl", "w") as dst:
        dst.write(src.read())

    print(f"[{strategy}] indexed {index.ntotal} vectors, dim={dim} -> {INDEX_DIR / f'{strategy}.faiss'}")


class VectorStore:
    """Query-time wrapper: load once, search many times."""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.index = faiss.read_index(str(INDEX_DIR / f"{strategy}.faiss"))
        self.meta = []
        with open(INDEX_DIR / f"{strategy}_meta.jsonl") as f:
            for line in f:
                self.meta.append(json.loads(line))

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        scores, ids = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            meta = self.meta[idx]
            results.append({**meta, "score": float(score)})
        return results


def main():
    for strategy in ["passage_baseline", "sentence_window", "metadata_aware"]:
        build_index(strategy)
    print("\nNext: python src/eval/retrieval_metrics.py  (compares strategies before you pick one in configs/agents.yaml)")


if __name__ == "__main__":
    main()