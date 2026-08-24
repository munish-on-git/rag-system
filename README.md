# Voice-Enabled Hindi RAG System

**Live demo (Hugging Face Space):** https://huggingface.co/spaces/MunishHF/Rag_System

An end-to-end voice-enabled Retrieval-Augmented Generation system built on the Hindi
portion of the MSMARCO-XI dataset. Built for the **HH Goa 2026 – Build a Voice-Enabled
RAG Model** task.

The system accepts a spoken query, transcribes it, retrieves relevant passages from a
Hindi corpus using a fine-tuned embedder and FAISS, reranks the results with a
cross-encoder, generates a citation-grounded answer, and checks that answer against the
retrieved context before returning it — with logging at every stage.

---

## Pipeline (what's actually running)

```text
Voice input
    │
    ▼
Speech-to-text (Sarvam Saaras v3)
    │
    ▼
Input guardrail (Llama Prompt Guard 2 — jailbreak/injection classifier)
    │
    ▼
Vector retrieval (FAISS, fine-tuned embedder, top ~20 candidates)
    │
    ▼
Cross-encoder reranking (top 5)
    │
    ▼
Answer generation (citation-forcing prompt, temperature=0)
    │
    ▼
Output guardrail (context-overlap grounding check)
    │
    ▼
Final answer + full query log (data/eval/query_log.csv)
```

**Not currently implemented** (present as empty scaffolding / documented as future work,
not wired into the running pipeline): hybrid keyword search, query rewriting, context
compression. See "Known limitations" below for why, and what's there instead.

---

## Two versions of this system: local (full) vs. deployed (constrained)

| | Local (`data/index/`) | Deployed Space (`data/index_deploy/`) |
|---|---|---|
| Corpus size | 953,398 passages | 200,000 passages |
| Embedder precision | fp32 | fp16 |
| Why smaller for deploy | — | Hugging Face free-tier Space storage caps at 1GB total; the full fp32 model + full index exceed that |

All retrieval and latency numbers in this repo's submission were measured **locally**,
against the full 953k-passage corpus. The deployed Space is a legitimate, smaller
version of the same architecture, not a different system.

`app.py` decides which index to load via the `RAG_INDEX_DIR` environment variable
(defaults to `data/index_deploy/` for Space deployment). To run `app.py` locally against
the full 953k corpus instead, set it explicitly before launching:

```bash
export RAG_INDEX_DIR=data/index
python app.py
```

---

## Setup

```bash
git clone https://github.com/munish-on-git/rag-system.git
cd rag-system
pip install -r requirements.txt --break-system-packages
export GROQ_API_KEY=your_key       # console.groq.com — generation + guardrail models
export SARVAM_API_KEY=your_key     # sarvam.ai — speech-to-text
```

This repo contains **code only** — no data, embeddings, index, or model weights are
committed (they'd be several GB). Rebuild them locally with the steps below.

## 1. Build the full local corpus (953,398 passages)

Check first — if `data/chunks/embeddings/passage_baseline.npy` already exists locally,
skip to step 2. This step took roughly 12 hours on a laptop CPU; don't re-run casually.

`--end` in the loader is a **row index into the source MSMARCO-XI dataset**, not a
passage count (each row expands to several passages) — `--end 210000` is what actually
produces the 953k-passage corpus used throughout this repo:

```bash
python -m src.ingestion.loader --langs hi --end 210000
python -m src.chunking.semantic_chunker
python -m src.embedding.encode --strategies passage_baseline
```

## 2. Build the local FAISS index (seconds, not hours — reuses the embeddings above)

```bash
python -m src.indexing.vector_store
```

Expected:
```
[passage_baseline] indexed 953398 vectors, dim=384 -> data/index/passage_baseline.faiss
```

(`sentence_window` and `metadata_aware` also get built here, at smaller vector counts —
see the strategy-comparison caveat below before comparing them to `passage_baseline`.)

## 3. Retrieval quality — Recall@k and MRR

```bash
python -m src.eval.retrieval_metrics
```

Sanity-check index/eval-set alignment if numbers look off:
```bash
python -m src.eval.diagnose_recall
```

## 4. End-to-end harness check (text query, no audio needed)

```bash
python -m src.graph.build_graph
```

## 5. Latency benchmark — P50/P70/P100

```bash
python -m src.eval.latency_bench
```

Reports retrieval-only latency and retrieval+generation latency separately, logged to
`data/eval/latency_log.csv`.

## 6. Inspect the full query log

```bash
cat data/eval/query_log.csv
```

Every request: query, block status, retrieved/reranked chunk IDs and scores, answer,
grounded flag, overlap score, per-stage latency.

## 7. Run the voice app locally, against the full 953k corpus

```bash
export RAG_INDEX_DIR=data/index
python app.py
```

Opens a Gradio UI with microphone input at `http://127.0.0.1:7860`.

To instead reproduce exactly what the deployed Space runs (200k-passage, fp16 slice):

```bash
python -m src.indexing.build_deploy_index --max-passages 200000
gzip -kf data/index_deploy/passage_baseline_meta.jsonl
rm data/index_deploy/passage_baseline_meta.jsonl
python app.py   # RAG_INDEX_DIR unset -> defaults to data/index_deploy/
```

---

## Strategy-comparison caveat

`retrieval_metrics.py` evaluates three chunking strategies, but only `passage_baseline`
was rebuilt at the full 953k scale — `sentence_window` (307k) and `metadata_aware` (206k)
reflect an earlier, smaller corpus. A larger index is a strictly harder retrieval task, so
`passage_baseline`'s numbers are **not directly comparable** to the other two as currently
built, and the script's own "best recall@5" suggestion should be ignored until all three
are rebuilt at matching scale. `configs/agents.yaml` intentionally stays on
`passage_baseline` — the only strategy verified at full scale.

## Known limitations (stated plainly)

- **Deployed corpus/precision are reduced** (200k passages, fp16) vs. the full local
  system (953k, fp32), due to Hugging Face's free-tier 1GB Space storage limit.
- **Grounding check is a word-overlap heuristic**, not a hard semantic constraint. Testing
  found it can be fooled by out-of-corpus questions that share common vocabulary with
  unrelated retrieved passages (e.g., a general-knowledge question retrieving passages
  that happen to share a few words), letting the LLM answer from its own pretrained
  knowledge rather than declining. A stricter or LLM-judged grounding check would close
  this gap; not implemented due to time and added-latency tradeoffs.
- **Hybrid (keyword) search, query rewriting, and context compression** are not wired
  into the pipeline — `src/indexing/keyword_store.py` exists as scaffolding but is unused.
  Cut for time; pure vector retrieval + reranking was sufficient to hit the latency target.
- **`metadata_aware`'s extra fields** (keywords, language tag) exist but aren't used as
  retrieval filters yet — the capability is built, the filtering hook isn't wired in.