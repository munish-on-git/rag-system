import json
from pathlib import Path

from src.eval.retrieval_metrics import build_eval_set

META_PATH = Path("data/chunks/embeddings/sentence_window_meta.jsonl")


def main():
    eval_set = build_eval_set()
    eval_positive_ids = {item["positive_passage_id"] for item in eval_set}

    indexed_passage_ids = set()
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            indexed_passage_ids.add(row["source_passage_id"])

    covered = eval_positive_ids & indexed_passage_ids
    print(f"Eval queries: {len(eval_set)}")
    print(f"Unique positive passage ids needed: {len(eval_positive_ids)}")
    print(f"Indexed passages available: {len(indexed_passage_ids)}")
    print(f"Positives actually present in the index: {len(covered)} "
          f"({100 * len(covered) / len(eval_positive_ids):.1f}%)")

    if len(covered) / len(eval_positive_ids) < 0.5:
        print("\n>> CONFIRMED: most eval positives were never indexed. "
              "This is an index-coverage gap, not a retrieval-quality failure. "
              "Fix: re-run loader.py + encode.py over the FULL passage set "
              "(or at minimum, a set that includes the eval query range), "
              "then re-run retrieval_metrics.py.")
    else:
        print("\n>> Index coverage looks fine -- recall failure is likely "
              "a genuine retrieval-quality issue, worth checking embedding "
              "quality or query/passage phrasing mismatch next.")


if __name__ == "__main__":
    main()