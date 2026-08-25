"""
main.py

Adapter exposing this project's embedder and generator to
rag-local-eval-loop, per its TARGET_INTERFACE.md contract. This is a flat
main.py (no app/ package), so the eval loop is pointed at it via:

    export EVAL_EMBEDDER_MODULE=main
    export EVAL_GENERATOR_MODULE=main

This file does NOT reimplement retrieval, reranking, or the LangGraph
harness -- it exposes exactly the two required surfaces (embed, generate)
so the eval loop can build its own throwaway index and call the real
embedding model and real generation/grounding logic in-process. Your
actual production FAISS index and reranker are not exercised by this
suite (documented in its own README under "Scope and limitations") --
only embedding quality and generation/grounding behavior are.

The grounding logic here is copied from src/graph/build_graph.py's
output_guardrail(), not reimplemented differently, so the eval loop's
"lying factor" check is testing the SAME guardrail that's actually
deployed, not a stand-in.
"""

import time

import yaml
from groq import Groq
from sentence_transformers import SentenceTransformer

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# --- shared config/clients, loaded once at import time ---------------------

with open("configs/agents.yaml") as f:
    _AGENTS_CFG = yaml.safe_load(f)
with open("configs/embedding.yaml") as f:
    _EMBED_CFG = yaml.safe_load(f)

_client = Groq(api_key=os.environ["GROQ_API_KEY"])

_model_path = (
    _EMBED_CFG["model"]["finetuned_path"]
    if _EMBED_CFG["model"]["use_finetuned"]
    else _EMBED_CFG["model"]["base_model"]
)
_model = None  # lazy-loaded by get_model()


# --- embedder module contract -----------------------------------------------

def get_model():
    """Called once by the eval loop; only the side effect (loading the
    model) matters. Returns the loaded SentenceTransformer for convenience,
    though the eval loop doesn't require a particular return value."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_model_path, device=_EMBED_CFG["inference"]["device"])
    return _model


def embed(texts: list[str]):
    """texts -> array-like, shape (len(texts), dim)"""
    model = get_model()
    return model.encode(texts, normalize_embeddings=_EMBED_CFG["model"]["normalize_embeddings"])


def embed_one(text: str):
    """text -> array-like, shape (dim,)"""
    model = get_model()
    return model.encode(text, normalize_embeddings=_EMBED_CFG["model"]["normalize_embeddings"])


# --- generator module contract ----------------------------------------------

class AnswerResult:
    """Plain answer object matching the eval loop's required attributes."""
    def __init__(self, text: str, grounded: bool, generation_ms: float, model: str):
        self.text = text
        self.grounded = grounded
        self.generation_ms = generation_ms
        self.model = model


def _is_grounded(answer_text: str, context_text: str, min_overlap: float) -> tuple[bool, float]:
    """
    Same word-overlap grounding heuristic as src/graph/build_graph.py's
    output_guardrail() -- kept in sync deliberately, not reimplemented, so
    this suite's reliability check reflects the real deployed guardrail.
    """
    if "I don't have enough information" in answer_text:
        return True, 0.0  # a correct decline counts as grounded

    context_lower = context_text.lower()
    answer_words = [w for w in answer_text.lower().split() if len(w) >= 4]
    if not answer_words:
        return True, 0.0

    overlap = sum(1 for w in answer_words if w in context_lower) / len(answer_words)
    return overlap >= min_overlap, overlap


def generate_answer(query: str, results: list) -> AnswerResult:
    """
    results: list of objects with .text and .source (str) attributes,
    supplied by the eval loop's own throwaway index -- not this project's
    real VectorStore/reranker output.
    """
    context = "\n\n".join(f"[{i+1}] {r.text}" for i, r in enumerate(results))

    prompt = f"""You are answering strictly from the context below. Do not use any
outside knowledge, even if you recognize the topic. If the context does not
contain the answer, you MUST say exactly: "I don't have enough information to answer that."
Cite sources inline like [1], [2] for every claim.

Context:
{context}

Question: {query}"""

    start = time.perf_counter()
    resp = _client.chat.completions.create(
        model=_AGENTS_CFG["model"]["generation_model"],
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    generation_ms = (time.perf_counter() - start) * 1000

    answer_text = resp.choices[0].message.content.strip()

    min_overlap = _AGENTS_CFG["guardrails"].get("min_context_overlap", 0.35)
    grounded, _overlap = _is_grounded(answer_text, context, min_overlap)

    if not grounded:
        answer_text = ("I don't have enough information to answer that confidently "
                        "based on the retrieved context.")

    return AnswerResult(
        text=answer_text,
        grounded=grounded,
        generation_ms=generation_ms,
        model=_AGENTS_CFG["model"]["generation_model"],
    )