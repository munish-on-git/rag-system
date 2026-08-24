# Voice-Enabled Hindi RAG System

## Deploy 200K indexed passages - https://huggingface.co/spaces/MunishHF/Rag_System
An end-to-end **voice-enabled Retrieval-Augmented Generation system** built on the Hindi portion of the **MSMARCO-XI dataset**.

The system accepts a spoken or text query, converts voice to text, rewrites the query when necessary, retrieves relevant information from a large Hindi corpus, reranks the retrieved passages, compresses the context, and generates a grounded answer with guardrails against unsupported or off-topic responses.

This project was built for the **HH Goa 2026 – Build a Voice-Enabled RAG Model** task.

---

## Overview

The system is designed as a complete RAG pipeline rather than a simple:

> Prompt → LLM → Answer

Instead, the pipeline uses multiple specialized components for retrieval quality, orchestration, answer grounding, evaluation, and observability.

### Pipeline

```text
Voice Input
    │
    ▼
Speech-to-Text
    │
    ▼
Query Rewriting
    │
    ▼
Hybrid Retrieval
    │
    ├── Vector Search
    └── Keyword / Lexical Retrieval
    │
    ▼
Candidate Retrieval
    │
    ▼
Cross-Encoder Reranking
    │
    ▼
Context Compression
    │
    ▼
Grounding / Guardrail Checks
    │
    ▼
Answer Generation
    │
    ▼
Final Response + Query Logging

# Runbook

Reproduces every metric in this repo's submission, from a fresh clone. All commands here
run **entirely locally** against the full 953k-passage corpus — they never call the
deployed Hugging Face Space, which runs a separate, storage-constrained 200k-passage/fp16
copy of this same system (see "Local vs. deployed" at the bottom).

Run these in order from the repo root.

## 0. Setup

```bash
git clone https://github.com/munish-on-git/rag-system.git
cd rag-system
pip install -r requirements.txt --break-system-packages
export GROQ_API_KEY=your_key
export SARVAM_API_KEY=your_key
```

## 1. Rebuild the full local corpus (953k passages)

Check first: `ls -lh data/chunks/embeddings/` — if `passage_baseline.npy` is already there,
skip to step 2. This step alone took ~12 hours on a laptop; don't re-run it casually.

```bash
python -m src.ingestion.loader --langs hi --end 953398
python -m src.chunking.semantic_chunker
python -m src.embedding.encode --strategies passage_baseline
```

## 2. Build the local FAISS index from existing embeddings (seconds, not hours)

```bash
python -m src.indexing.vector_store
```

Expected output — note the three strategies intentionally have different vector counts
(see the strategy-comparison caveat below, this is not a bug):
```
[passage_baseline] indexed 953398 vectors, dim=384 -> data/index/passage_baseline.faiss
[sentence_window] indexed 307149 vectors, dim=384 -> data/index/sentence_window.faiss
[metadata_aware] indexed 206351 vectors, dim=384 -> data/index/metadata_aware.faiss
```

## 3. Retrieval quality — Recall@k and MRR

```bash
python -m src.eval.retrieval_metrics
```

**Read the strategy-comparison caveat below before interpreting this output** — the three
strategies are evaluated against indexes of different sizes, so the printed "best recall@5"
recommendation is not a fair comparison. `passage_baseline` (953k) is the configured,
reported strategy.

Sanity-check index coverage if numbers look off:
```bash
python -m src.eval.diagnose_recall
```

## 4. End-to-end harness sanity check (text path, one query)

```bash
python -m src.graph.build_graph
```

Confirms the full graph — guardrail, retrieval, rerank, generation, grounding check,
logging — runs without error and returns a grounded, cited answer.

## 5. Latency benchmark — P50/P70/P100

```bash
python -m src.eval.latency_bench
```

Reports retrieval-only latency (the <200ms target) and retrieval+generation latency
separately. Logs every run to `data/eval/latency_log.csv`.

## 6. Full query log — every request, every stage

```bash
cat data/eval/query_log.csv
```

Columns: timestamp, query, blocked/block_reason, retrieved chunk IDs + scores, reranked
chunk IDs, answer, grounded flag, overlap score, per-stage latency.

## 7. Run the voice app locally

```bash
python app.py
```

Opens a Gradio UI with microphone input. Reads `data/index/` (the storage-constrained
slice).Rebuild it if missing:

```bash
python -m src.indexing.build_deploy_index --max-passages 953398
gzip -kf data/index/passage_baseline_meta.jsonl
rm data/index/passage_baseline_meta.jsonl
python app.py
```

---

## Local vs. deployed — two separate systems, same architecture

| | Local (this runbook) | Deployed (Hugging Face Space) |
|---|---|---|
| Corpus size | 953,398 passages | 200,000 passages |
| Embedder | fp32 | fp16 |
| Index location | `data/index/` | 
| Reason for the difference | None — this is the full system | Hugging Face free-tier Space storage caps at 1GB; the full corpus + fp32 model exceed that |

All retrieval/latency numbers in the submission were measured **locally**, against the full
953k corpus, using the commands above — not against the deployed Space.

## Strategy-comparison caveat — read before trusting "best recall@5"

`retrieval_metrics.py` evaluates all three chunking strategies, but **only `passage_baseline`
was rebuilt at full 953k scale**; `sentence_window` (307k) and `metadata_aware` (206k) still
reflect an earlier, smaller corpus size. A larger index is a strictly harder retrieval
task — more near-duplicate and plausible-but-wrong candidates compete for the top-k slots —
so `passage_baseline`'s recall numbers are **not directly comparable** to the other two in
the current output, and the script's own "best recall@5, switch to X" suggestion should be
**ignored** until all three are rebuilt at the same scale. `configs/agents.yaml` is
intentionally left on `passage_baseline`, the only strategy actually verified at full scale.

## Known, documented limitations (stated plainly, not hidden)

- Deployed corpus (200k) and embedder precision (fp16) are both smaller than the local,
  fully-evaluated system (953k, fp32), due to Hugging Face's free-tier 1GB storage limit.
- `metadata_aware`'s extra fields (keywords, language tag) exist but aren't yet used as
  retrieval filters — the capability is built, the filtering hook isn't wired in, due to
  time constraints.
- Chunking-strategy comparison numbers are not apples-to-apples until all three are
  rebuilt at matching corpus scale (see caveat above).