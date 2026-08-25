import csv
import time
from pathlib import Path

import numpy as np

from src.graph.build_graph import (
    EMBEDDER,
    STORE,
    RERANKER,
    AGENTS_CFG,
    client,
    generate,
    output_guardrail,
)

OUT_CSV = Path("data/eval/latency_log.csv")


TEST_QUERIES = [
    "होम लोन क्या है?",
    "क्रेडिट स्कोर कैसे सुधारें?",
    "बीमा पॉलिसी क्या होती है?",
    "म्यूचुअल फंड में निवेश कैसे करें?",
    "पर्सनल लोन के लिए ब्याज दर क्या है?",
    "बैंक खाता कैसे खोलें?",
    "डेबिट कार्ड और क्रेडिट कार्ड में क्या अंतर है?",
    "फिक्स्ड डिपॉजिट क्या होता है?",
    "लोन लेने के लिए कौन से दस्तावेज चाहिए?",
    "ब्याज दर क्या होती है?",
    "शेयर मार्केट क्या है?",
    "म्यूचुअल फंड कैसे काम करता है?",
    "ईएमआई क्या होती है?",
    "क्रेडिट स्कोर क्यों महत्वपूर्ण है?",
    "होम लोन कितने समय के लिए मिलता है?",
    "बीमा क्लेम कैसे किया जाता है?",
    "सेविंग अकाउंट क्या होता है?",
    "करंट अकाउंट किसके लिए होता है?",
    "क्रेडिट कार्ड की लिमिट क्या होती है?",
    "फाइनेंस में जोखिम क्या है?",
    "निवेश और बचत में क्या अंतर है?",
    "एफडी और आरडी में क्या अंतर है?",
    "पर्सनल लोन क्या होता है?",
    "लोन की ईएमआई कैसे तय होती है?",
    "बैंक ब्याज कैसे कमाता है?",
    "इंश्योरेंस प्रीमियम क्या होता है?",
    "म्यूचुअल फंड में जोखिम कितना होता है?",
    "होम लोन के लिए डाउन पेमेंट क्या है?",
    "क्रेडिट रिपोर्ट क्या होती है?",
    "बैंक लोन को कैसे मंजूर करता है?",
]


def timed_retrieve_and_rerank(query: str):
    start = time.perf_counter()

    # Embed query
    query_emb = EMBEDDER.encode(
        query,
        normalize_embeddings=True,
    )

    # Retrieve top candidate_k
    candidate_k = AGENTS_CFG["retrieval"].get("candidate_k", 20)
    candidates = STORE.search(
        query_emb,
        top_k=candidate_k,
    )

    retrieve_ms = (time.perf_counter() - start) * 1000

    # Rerank candidates
    rerank_start = time.perf_counter()

    if candidates:
        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        scores = RERANKER.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_k = AGENTS_CFG["retrieval"]["top_k"]

        reranked = [
            candidate
            for candidate, _ in ranked[:top_k]
        ]
    else:
        reranked = []

    rerank_ms = (
        time.perf_counter() - rerank_start
    ) * 1000

    return (
        reranked,
        retrieve_ms,
        rerank_ms,
    )


def timed_generation(query: str, reranked: list):
    state = {
        "query": query,
        "reranked": reranked,
    }

    start = time.perf_counter()

    state = generate(state)

    generate_ms = (
        time.perf_counter() - start
    ) * 1000

    return (
        state["answer"],
        generate_ms,
    )


def timed_grounding_check(
    query: str,
    reranked: list,
    answer: str,
):
    state = {
        "query": query,
        "reranked": reranked,
        "answer": answer,
    }

    start = time.perf_counter()

    state = output_guardrail(state)

    grounding_ms = (
        time.perf_counter() - start
    ) * 1000

    return (
        state["answer"],
        state.get("grounded"),
        grounding_ms,
    )


def run_benchmark(queries: list):
    rows = []

    for i, query in enumerate(queries, 1):

        total_start = time.perf_counter()

        (
            reranked,
            retrieve_ms,
            rerank_ms,
        ) = timed_retrieve_and_rerank(query)

        answer, generate_ms = timed_generation(
            query,
            reranked,
        )

        (
            final_answer,
            grounded,
            grounding_ms,
        ) = timed_grounding_check(
            query,
            reranked,
            answer,
        )

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        row = {
            "query": query,
            "retrieve_ms": round(retrieve_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "retrieval_rerank_ms": round(
                retrieve_ms + rerank_ms,
                2,
            ),
            "generate_ms": round(generate_ms, 2),
            "grounding_check_ms": round(
                grounding_ms,
                2,
            ),
            "text_pipeline_total_ms": round(
                total_ms,
                2,
            ),
            "grounded": grounded,
            "answer": final_answer,
        }

        rows.append(row)

        print(
            f"[{i}/{len(queries)}] "
            f"retrieve={retrieve_ms:.1f}ms | "
            f"rerank={rerank_ms:.1f}ms | "
            f"generate={generate_ms:.1f}ms | "
            f"grounding={grounding_ms:.1f}ms | "
            f"total={total_ms:.1f}ms"
        )

    return rows


def percentile_report(rows: list, field: str):
    values = [r[field] for r in rows]

    return {
        "P50": round(
            float(np.percentile(values, 50)),
            2,
        ),
        "P70": round(
            float(np.percentile(values, 70)),
            2,
        ),
        "P100": round(
            float(np.max(values)),
            2,
        ),
    }


def main():
    rows = run_benchmark(TEST_QUERIES)

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nLogged {len(rows)} runs -> {OUT_CSV}"
    )

    print("\n========== FINAL LATENCY RESULTS ==========")

    print("\n--- Retrieval ---")
    print(
        percentile_report(
            rows,
            "retrieve_ms",
        )
    )

    print("\n--- Reranking ---")
    print(
        percentile_report(
            rows,
            "rerank_ms",
        )
    )

    print("\n--- Retrieval + Reranking ---")
    print(
        percentile_report(
            rows,
            "retrieval_rerank_ms",
        )
    )

    print("\n--- Generation ---")
    print(
        percentile_report(
            rows,
            "generate_ms",
        )
    )

    print("\n--- Grounding Check ---")
    print(
        percentile_report(
            rows,
            "grounding_check_ms",
        )
    )

    print("\n--- Full Text RAG Pipeline ---")
    print(
        percentile_report(
            rows,
            "text_pipeline_total_ms",
        )
    )

    print(
        "\nNote: STT is measured separately because this benchmark "
        "uses text queries. Voice end-to-end latency should be "
        "reported as STT + text pipeline."
    )


if __name__ == "__main__":
    main()