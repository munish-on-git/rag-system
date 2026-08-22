# Voice-Enabled Hindi RAG System

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

Deploy 200K indexed passages - [text](https://huggingface.co/spaces/MunishHF/Rag_System)