import argparse
import hashlib
import itertools
import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("data/processed")

LANG_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "gu", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "or", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


def stream_split(lang: str, split: str, start: int, end: int):
    prefix = LANG_PREFIX[lang]
    folder = "train" if split == "train" else "validation"
    suffix = "train" if split == "train" else "val"
    remote_path = f"{folder}/{prefix}{suffix}.parquet"
    ds = load_dataset(
        "ai4bharat/MSMARCO-XI",
        data_files={split: remote_path},
        split=split,
        streaming=True,
    )
    return itertools.islice(ds, start, end)


def extract_passages(lang: str, split: str, start: int, end: int):
    seen_hashes = set()
    kept, duplicates = 0, 0

    for row in stream_split(lang, split, start, end):
        passages = row["passages"]["Translated_passages"]
        query_id = row.get("query_id")

        for idx, text in enumerate(passages):
            text = (text or "").strip()
            if not text:
                continue
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(h)
            kept += 1
            yield {
                "passage_id": h,
                "text": text,
                "lang": lang,
                "source_query_id": query_id,
                "source_passage_index": idx,
            }

    print(f"[{lang}/{split}] kept {kept} unique passages, skipped {duplicates} duplicates")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["hi"])
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=21_000,
                         help="must cover retrieval_metrics.py's eval row range (20000:21000)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "passages.jsonl"

    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for lang in args.langs:
            for passage in extract_passages(lang, args.split, args.start, args.end):
                f.write(json.dumps(passage, ensure_ascii=False) + "\n")
                total += 1

    print(f"\nWrote {total} total unique passages to {out_path}")
    print("Next: python -m src.chunking.semantic_chunker")


if __name__ == "__main__":
    main()