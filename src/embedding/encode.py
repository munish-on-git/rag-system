"""
src/embedding/encode.py

Batch-embeds each data/chunks/{strategy}.jsonl file with the fine-tuned
embedder from configs/embedding.yaml, saves aligned .npy embeddings +
.jsonl metadata to data/chunks/embeddings/, ready for vector_store.py.

Usage:
    python -m src.embedding.encode
    python -m src.embedding.encode --strategies sentence_window --limit 100000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = Path("data/chunks")
OUT_DIR = CHUNKS_DIR / "embeddings"


def load_config():
    with open("configs/embedding.yaml") as f:
        return yaml.safe_load(f)


def load_model(cfg):
    model_path = cfg["model"]["finetuned_path"] if cfg["model"]["use_finetuned"] else cfg["model"]["base_model"]
    print(f"Loading embedder: {model_path}")
    model = SentenceTransformer(model_path, device=cfg["inference"]["device"])
    model.max_seq_length = cfg["model"]["max_seq_length"]
    return model


def encode_strategy(model, cfg, strategy: str, limit: int):
    in_path = CHUNKS_DIR / f"{strategy}.jsonl"
    if not in_path.exists():
        print(f"skip: {in_path} not found")
        return

    chunks = []
    with open(in_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            chunks.append(json.loads(line))

    texts = [c["text"] for c in chunks]
    print(f"[{strategy}] encoding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=cfg["inference"]["batch_size"],
        normalize_embeddings=cfg["model"]["normalize_embeddings"],
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{strategy}.npy", embeddings.astype("float32"))
    with open(OUT_DIR / f"{strategy}_meta.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[{strategy}] saved {embeddings.shape} -> {OUT_DIR / f'{strategy}.npy'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+",
                         default=["passage_baseline", "sentence_window", "metadata_aware"])
    parser.add_argument("--limit", type=int, default=0,
                         help="0 = encode all chunks (default now that loader.py bounds the row range); "
                              "set a positive number to cap for a quick speed test")
    args = parser.parse_args()

    cfg = load_config()
    model = load_model(cfg)

    for strategy in args.strategies:
        encode_strategy(model, cfg, strategy, args.limit)

    print("\nNext: python src/indexing/vector_store.py")


if __name__ == "__main__":
    main()