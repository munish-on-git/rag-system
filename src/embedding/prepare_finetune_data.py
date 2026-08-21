"""
src/embedding/prepare_finetune_data.py

Loads ai4bharat/MSMARCO-XI for one or more languages and writes
(query, positive_passage, negative_passage) triplets to data/finetune/
as JSONL, ready for finetune.py (sentence-transformers MultipleNegativesRankingLoss
or a triplet loss).

Usage:
    python src/embedding/prepare_finetune_data.py --langs hi --limit 200000
    python src/embedding/prepare_finetune_data.py --langs hi bn mr --limit 100000
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("data/finetune")


def build_triplets(lang: str, limit: int | None, max_negatives: int):
    """
    Yields dicts: {"query": ..., "positive": ..., "negative": ..., "lang": ...}
    One row per (positive, negative) pair so a query with 1 positive and
    3 negatives yields 3 training triplets.
    """
    split = f"train[:{limit}]" if limit else "train"
    ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split=split)

    kept, skipped = 0, 0
    for row in ds:
        passages = row["passages"]["Translated_passages"]
        is_selected = row["passages"]["is_selected"]

        positives = [p for p, s in zip(passages, is_selected) if s == 1]
        negatives = [p for p, s in zip(passages, is_selected) if s == 0]

        # skip rows with no gold passage or no negatives — useless for contrastive loss
        if not positives or not negatives:
            skipped += 1
            continue

        query = row["query"].strip()
        positive = positives[0].strip()
        if not query or not positive:
            skipped += 1
            continue

        for neg in negatives[:max_negatives]:
            neg = neg.strip()
            if not neg:
                continue
            yield {"query": query, "positive": positive, "negative": neg, "lang": lang}
            kept += 1

    print(f"[{lang}] wrote {kept} triplets, skipped {skipped} rows with no usable pos/neg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["hi"],
                         help="Language codes, e.g. hi bn mr ta te")
    parser.add_argument("--limit", type=int, default=200_000,
                         help="Max rows to pull per language (streamed/sliced, not the full 10M)")
    parser.add_argument("--max-negatives", type=int, default=2,
                         help="Max negative passages to pair with each query's positive")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "msmarco_xi_triplets.jsonl"

    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for lang in args.langs:
            for triplet in build_triplets(lang, args.limit, args.max_negatives):
                f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                total += 1

    print(f"\nWrote {total} total triplets to {out_path}")
    print("Next: point src/embedding/finetune.py at this file to train the embedder.")


if __name__ == "__main__":
    main()