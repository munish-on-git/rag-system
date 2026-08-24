import csv
import time
from pathlib import Path

import numpy as np

from src.graph.build_graph import (
    build_app, EMBEDDER, STORE, AGENTS_CFG, client,
)

OUT_CSV = Path("data/eval/latency_log.csv")

# Swap in real questions from your domain/eval set once you have them;
# this is a placeholder set covering a spread of topic types.
TEST_QUERIES = [
    "होम लोन क्या है?",
    "क्रेडिट स्कोर कैसे सुधारें?",
    "बीमा पॉलिसी क्या होती है?",
    "म्यूचुअल फंड में निवेश कैसे करें?",
    "पर्सनल लोन के लिए ब्याज दर क्या है?",
    # ... extend this to 30-50 for a real benchmark run before submitting
]


def timed_retrieve(query: str):
    start = time.perf_counter()
    query_emb = EMBEDDER.encode(query, normalize_embeddings=True)
    results = STORE.search(query_emb, top_k=AGENTS_CFG["retrieval"]["top_k"])
    return results, (time.perf_counter() - start) * 1000


def timed_generate(query: str, results: list):
    context = "\n\n".join(f"[{i+1}] {r['text']}" for i, r in enumerate(results))
    prompt = f"""Answer using ONLY this context, cite as [1],[2]. If insufficient, say so.

Context:
{context}

Question: {query}"""
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=AGENTS_CFG["model"]["generation_model"],
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content, (time.perf_counter() - start) * 1000


def run_benchmark(queries: list):
    rows = []
    for q in queries:
        results, retrieve_ms = timed_retrieve(q)
        answer, generate_ms = timed_generate(q, results)
        rows.append({
            "query": q,
            "retrieve_ms": round(retrieve_ms, 2),
            "generate_ms": round(generate_ms, 2),
            "retrieval_only_total_ms": round(retrieve_ms, 2),
            "retrieval_plus_generation_ms": round(retrieve_ms + generate_ms, 2),
        })
        print(f"retrieve={retrieve_ms:.1f}ms  generate={generate_ms:.1f}ms  | {q}")
    return rows


def percentile_report(rows: list, field: str):
    values = [r[field] for r in rows]
    return {
        "p50": round(float(np.percentile(values, 50)), 2),
        "p70": round(float(np.percentile(values, 70)), 2),
        "p100": round(float(np.max(values)), 2),
    }


def main():
    rows = run_benchmark(TEST_QUERIES)

    Path("data/eval").mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nLogged {len(rows)} runs -> {OUT_CSV}")
    print("\n--- Retrieval only (the <200ms target) ---")
    print(percentile_report(rows, "retrieve_ms"))
    print("\n--- Retrieval + generation (full text pipeline, STT excluded) ---")
    print(percentile_report(rows, "retrieval_plus_generation_ms"))
    print("\nNote: report STT latency separately from Sarvam's own "
          "TranscriptionResponse.latency_ms -- don't fold it into these numbers.")


if __name__ == "__main__":
    main()