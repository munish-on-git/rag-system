# You already have the full 953k-passage embeddings encoded locally --
# re-embedding a smaller slice would waste the ~12 hours you already spent.
# This instead takes the FIRST N rows of the existing .npy + meta.jsonl
# (already in the order they were encoded, no reshuffling needed) and builds
# a separate, smaller FAISS index sized to fit Hugging Face's free 1GB
# Space storage quota alongside your model checkpoint.

# Your full 953k index stays local for retrieval_metrics.py evaluation
# (report those numbers honestly in your writeup) -- this smaller one is
# specifically what gets deployed.

# Usage:
#     python -m src.indexing.build_deploy_index --max-passages 450000

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np

EMBEDDINGS_DIR = Path("data/chunks/embeddings")
DEPLOY_INDEX_DIR = Path("data/index_deploy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="passage_baseline")
    parser.add_argument("--max-passages", type=int, default=450_000,
                         help="sized to fit ~1GB free HF Space quota alongside the model checkpoint")
    args = parser.parse_args()

    emb_path = EMBEDDINGS_DIR / f"{args.strategy}.npy"
    meta_path = EMBEDDINGS_DIR / f"{args.strategy}_meta.jsonl"

    embeddings = np.load(emb_path)
    print(f"Full local embeddings: {embeddings.shape}")

    sliced = embeddings[: args.max_passages]
    print(f"Deploying slice: {sliced.shape}")

    dim = sliced.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(sliced)

    DEPLOY_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(DEPLOY_INDEX_DIR / f"{args.strategy}.faiss"))

    with open(meta_path) as src, open(DEPLOY_INDEX_DIR / f"{args.strategy}_meta.jsonl", "w") as dst:
        for i, line in enumerate(src):
            if i >= args.max_passages:
                break
            dst.write(line)

    out_faiss = DEPLOY_INDEX_DIR / f"{args.strategy}.faiss"
    size_mb = out_faiss.stat().st_size / (1024 * 1024)
    print(f"\nWrote {index.ntotal} vectors -> {out_faiss} ({size_mb:.0f} MB)")
    print("Next: point VectorStore at data/index_deploy/ for the deployed app,")
    print("      keep data/index/ (the full 953k set) for local retrieval_metrics.py runs.")


if __name__ == "__main__":
    main()