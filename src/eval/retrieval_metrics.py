"""
src/eval/retrieval_metrics.py

Builds a small eval set from a slice of MSMARCO-XI NOT used in loader.py
(rows 20000:21000, vs loader.py's 0:20000 -- no contamination), then measures
Recall@k and MRR@10 for each chunking strategy's FAISS index. This is what
justifies which strategy you actually ship, instead of guessing.

Usage:
    python -m src.eval.retrieval_metrics
"""

import os
# Must be set before faiss/torch are imported -- FAISS and PyTorch both try
# to grab OpenMP threads on macOS, which segfaults if left uncapped.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.indexing.vector_store import VectorStore

LANG_PREFIX = {"hi": "hin"}  # extend if you prep other languages
EVAL_OUT = Path("data/eval/retrieval_comparison.json")


def build_eval_set(lang="hi", start=20000, end=21000):
    prefix = LANG_PREFIX[lang]
    ds = load_dataset(
        "ai4bharat/MSMARCO-XI",
        data_files={"validation": f"validation/{prefix}val.parquet"},
        split=f"validation[{start}:{end}]",
    )
    import hashlib
    eval_set = []
    for row in ds:
        passages = row["passages"]["Translated_passages"]
        is_selected = row["passages"]["is_selected"]
        positives = [p for p, s in zip(passages, is_selected) if s == 1]
        if not positives or not row["query"].strip():
            continue
        positive_hash = hashlib.md5(positives[0].strip().encode("utf-8")).hexdigest()
        eval_set.append({"query": row["query"].strip(), "positive_passage_id": positive_hash})
    print(f"Built eval set: {len(eval_set)} queries")
    return eval_set


def load_embedder():
    with open("configs/embedding.yaml") as f:
        cfg = yaml.safe_load(f)
    model_path = cfg["model"]["finetuned_path"] if cfg["model"]["use_finetuned"] else cfg["model"]["base_model"]
    return SentenceTransformer(model_path, device=cfg["inference"]["device"])


def evaluate_strategy(strategy: str, eval_set: list, model, k_values=(5, 10, 20)):
    store = VectorStore(strategy)
    recalls = {k: 0 for k in k_values}
    reciprocal_ranks = []

    for item in eval_set:
        query_emb = model.encode(item["query"], normalize_embeddings=True)
        results = store.search(query_emb, top_k=max(k_values))
        # sentence_window chunks are sub-windows of a passage -- match on
        # source_passage_id, not chunk_id, so multi-window passages still count
        result_passage_ids = [r["source_passage_id"] for r in results]

        rank = None
        for i, pid in enumerate(result_passage_ids):
            if pid == item["positive_passage_id"]:
                rank = i + 1
                break
        reciprocal_ranks.append(1 / rank if rank else 0)

        for k in k_values:
            if item["positive_passage_id"] in result_passage_ids[:k]:
                recalls[k] += 1

    n = len(eval_set)
    return {
        **{f"recall@{k}": round(recalls[k] / n, 4) for k in k_values},
        "mrr@10": round(float(np.mean(reciprocal_ranks)), 4),
        "n_queries": n,
    }


def main():
    eval_set = build_eval_set()
    model = load_embedder()

    results = {}
    for strategy in ["passage_baseline", "sentence_window", "metadata_aware"]:
        print(f"\nEvaluating {strategy}...")
        results[strategy] = evaluate_strategy(strategy, eval_set, model)
        print(results[strategy])

    Path("data/eval").mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved comparison to {EVAL_OUT}")
    best = max(results, key=lambda s: results[s]["recall@5"])
    print(f"Best recall@5: {best} -- set retrieval.strategy: \"{best}\" in configs/agents.yaml")


if __name__ == "__main__":
    main()