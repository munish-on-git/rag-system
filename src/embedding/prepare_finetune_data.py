"""
src/embedding/prepare_finetune_data.py

Streams ai4bharat/MSMARCO-XI directly from the Hub (no full-file download)
and writes (query, positive_passage, negative_passage) triplets to
data/finetune/ as JSONL, ready for finetune.py.

Laptop-friendly by default: reads the smaller validation split and stops
after --limit rows via streaming, so it never pulls the multi-GB train
parquet file to disk.

Usage:
    python src/embedding/prepare_finetune_data.py --langs hi --limit 20000
    python src/embedding/prepare_finetune_data.py --langs hi bn --limit 10000 --split train
"""

import argparse
import itertools
import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("data/finetune")

# language code -> file prefix used in the repo's train/ and validation/ folders
# (confirmed against actual files: train/hintrain.parquet, validation/telval.parquet, etc.)
LANG_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "gu", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "or", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


def stream_split(lang: str, split: str, limit: int):
    """
    Streams one language's parquet file directly from the Hub — no full
    download. `split` is "train" or "validation" (validation is much smaller).
    """
    prefix = LANG_PREFIX[lang]
    folder = "train" if split == "train" else "validation"
    suffix = "train" if split == "train" else "val"
    remote_path = f"{folder}/{prefix}{suffix}.parquet"

    ds = load_dataset(
        "ai4bharat/MSMARCO-XI",
        data_files={split: remote_path},
        split=split,
        streaming=True,   # <- key: reads over HTTP in chunks, never saves the full file
    )
    return itertools.islice(ds, limit)


def build_triplets(lang: str, split: str, limit: int, max_negatives: int):
    kept, skipped = 0, 0
    for row in stream_split(lang, split, limit):
        passages = row["passages"]["Translated_passages"]
        is_selected = row["passages"]["is_selected"]

        positives = [p for p, s in zip(passages, is_selected) if s == 1]
        negatives = [p for p, s in zip(passages, is_selected) if s == 0]

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

    print(f"[{lang}/{split}] wrote {kept} triplets, skipped {skipped} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["hi"])
    parser.add_argument("--split", choices=["train", "validation"], default="validation",
                         help="validation is far smaller and streams fast; use train only once this works")
    parser.add_argument("--limit", type=int, default=20_000,
                         help="rows to stream per language (kept small on purpose for a laptop)")
    parser.add_argument("--max-negatives", type=int, default=2)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "msmarco_xi_triplets.jsonl"

    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for lang in args.langs:
            for triplet in build_triplets(lang, args.split, args.limit, args.max_negatives):
                f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                total += 1

    print(f"\nWrote {total} total triplets to {out_path}")


if __name__ == "__main__":
    main()