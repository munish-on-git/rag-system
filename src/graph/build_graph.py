"""
The harness: audio -> STT (Sarvam) -> input guardrail (Prompt Guard) ->
retrieve -> rerank -> generate -> output guardrail (grounding check) -> log.

Usage (text, for local testing):
    from src.graph.build_graph import build_app
    app = build_app()
    result = app.invoke({"query": "मॉर्गेज क्या है?"})
    print(result["answer"])

Usage (voice, the real pipeline):
    result = app.invoke({"audio_bytes": raw_bytes, "audio_filename": "recording.webm"})
"""

import csv
import os
import time
from pathlib import Path
from typing import TypedDict, Optional

import yaml
from groq import Groq
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.indexing.vector_store import VectorStore
from src.stt.sarvam_client import SarvamSTTClient, SarvamSTTError

client = Groq(api_key=os.environ["GROQ_API_KEY"])
stt_client = SarvamSTTClient(api_key=os.environ["SARVAM_API_KEY"])

LOG_PATH = Path("data/eval/query_log.csv")
LOG_FIELDS = [
    "timestamp", "query", "blocked", "block_reason",
    "retrieved_ids", "retrieved_scores", "reranked_ids",
    "answer", "grounded", "overlap_score",
    "stt_latency_ms", "retrieve_latency_ms", "rerank_latency_ms", "generate_latency_ms",
]


class RAGState(TypedDict):
    audio_bytes: Optional[bytes]
    audio_filename: Optional[str]

    query: str
    stt_latency_ms: Optional[float]

    blocked: bool
    block_reason: Optional[str]

    retrieved: list
    retrieve_latency_ms: Optional[float]
    reranked: list
    rerank_latency_ms: Optional[float]

    answer: str
    grounded: bool
    overlap_score: Optional[float]
    generate_latency_ms: Optional[float]


def load_configs():
    with open("configs/agents.yaml") as f:
        agents_cfg = yaml.safe_load(f)
    with open("configs/embedding.yaml") as f:
        embed_cfg = yaml.safe_load(f)
    return agents_cfg, embed_cfg


AGENTS_CFG, EMBED_CFG = load_configs()

_model_path = EMBED_CFG["model"]["finetuned_path"] if EMBED_CFG["model"]["use_finetuned"] else EMBED_CFG["model"]["base_model"]
EMBEDDER = SentenceTransformer(_model_path, device=EMBED_CFG["inference"]["device"])
STORE = VectorStore(AGENTS_CFG["retrieval"]["strategy"])

# Small multilingual cross-encoder reranker -- scores (query, chunk) pairs
# directly rather than relying only on embedding cosine similarity.
RERANKER = CrossEncoder(
    AGENTS_CFG["retrieval"].get("reranker_model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
    device=EMBED_CFG["inference"]["device"],
)


def speech_to_text(state: RAGState) -> RAGState:
    if state.get("query"):
        state["stt_latency_ms"] = None
        return state

    start = time.perf_counter()
    try:
        result = stt_client.transcribe(
            audio_bytes=state["audio_bytes"],
            filename=state.get("audio_filename", "recording.webm"),
            language_code="hi-IN",
            mode="transcribe",
        )
    except SarvamSTTError as exc:
        state["query"] = ""
        state["blocked"] = True
        state["block_reason"] = f"Could not transcribe audio: {exc}"
        state["stt_latency_ms"] = (time.perf_counter() - start) * 1000
        return state

    state["query"] = result.transcript
    state["stt_latency_ms"] = result.latency_ms
    return state


def input_guardrail(state: RAGState) -> RAGState:
    if state.get("blocked"):
        return state

    resp = client.chat.completions.create(
        model=AGENTS_CFG["model"]["guardrail_model"],
        max_tokens=10,
        messages=[{"role": "user", "content": state["query"]}],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        jailbreak_score = float(raw)
    except ValueError:
        state["blocked"] = True
        state["block_reason"] = f"Safety classifier returned an unexpected response: {raw!r}"
        return state

    threshold = AGENTS_CFG["guardrails"].get("jailbreak_threshold", 0.5)
    state["blocked"] = jailbreak_score >= threshold
    state["block_reason"] = (
        f"Query flagged as unsafe (jailbreak score {jailbreak_score:.4f} >= threshold {threshold})."
        if state["blocked"] else None
    )
    return state


def retrieve(state: RAGState) -> RAGState:
    start = time.perf_counter()
    query_emb = EMBEDDER.encode(state["query"], normalize_embeddings=True)
    # pull more than we need (candidate_k) so the reranker has real signal to work with
    candidate_k = AGENTS_CFG["retrieval"].get("candidate_k", 20)
    state["retrieved"] = STORE.search(query_emb, top_k=candidate_k)
    state["retrieve_latency_ms"] = (time.perf_counter() - start) * 1000
    return state


def rerank(state: RAGState) -> RAGState:
    start = time.perf_counter()
    candidates = state["retrieved"]
    top_k = AGENTS_CFG["retrieval"]["top_k"]

    if not candidates:
        state["reranked"] = []
        state["rerank_latency_ms"] = (time.perf_counter() - start) * 1000
        return state

    pairs = [[state["query"], c["text"]] for c in candidates]
    scores = RERANKER.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    state["reranked"] = [c for c, _ in ranked[:top_k]]
    state["rerank_latency_ms"] = (time.perf_counter() - start) * 1000
    return state


def generate(state: RAGState) -> RAGState:
    start = time.perf_counter()
    context = "\n\n".join(f"[{i+1}] {r['text']}" for i, r in enumerate(state["reranked"]))

    prompt = f"""You are answering strictly from the context below. Do not use any
outside knowledge, even if you recognize the topic. If the context does not
contain the answer, you MUST say exactly: "I don't have enough information to answer that."
Cite sources inline like [1], [2] for every claim.

Context:
{context}

Question: {state['query']}"""

    resp = client.chat.completions.create(
        model=AGENTS_CFG["model"]["generation_model"],
        max_tokens=500,
        temperature=0,  # deterministic, reduces "confident outside knowledge" drift
        messages=[{"role": "user", "content": prompt}],
    )
    state["answer"] = resp.choices[0].message.content.strip()
    state["generate_latency_ms"] = (time.perf_counter() - start) * 1000
    return state


def output_guardrail(state: RAGState) -> RAGState:
    if "I don't have enough information" in state["answer"]:
        state["grounded"] = True
        state["overlap_score"] = None
        return state

    context_text = " ".join(r["text"] for r in state["reranked"]).lower()
    answer_words = [w for w in state["answer"].lower().split() if len(w) >= 4]
    if not answer_words:
        state["grounded"] = True
        state["overlap_score"] = None
        return state

    overlap = sum(1 for w in answer_words if w in context_text) / len(answer_words)
    state["overlap_score"] = round(overlap, 4)

    # Raised threshold (was 0.15) after observing confident out-of-corpus
    # answers slip through on topics the LLM already knows from pretraining --
    # a stricter bar catches more of those, at the cost of occasionally
    # declining a genuinely-grounded but loosely-worded answer.
    threshold = AGENTS_CFG["guardrails"].get("min_context_overlap", 0.35)
    state["grounded"] = overlap >= threshold

    if not state["grounded"]:
        state["answer"] = ("I don't have enough information to answer that confidently "
                            "based on the retrieved context.")
    return state


def log_query(state: RAGState) -> RAGState:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": state.get("query", ""),
        "blocked": state.get("blocked", False),
        "block_reason": state.get("block_reason", ""),
        "retrieved_ids": ";".join(r["chunk_id"] for r in state.get("retrieved", [])),
        "retrieved_scores": ";".join(f"{r['score']:.4f}" for r in state.get("retrieved", [])),
        "reranked_ids": ";".join(r["chunk_id"] for r in state.get("reranked", [])),
        "answer": state.get("answer", ""),
        "grounded": state.get("grounded", ""),
        "overlap_score": state.get("overlap_score", ""),
        "stt_latency_ms": state.get("stt_latency_ms", ""),
        "retrieve_latency_ms": state.get("retrieve_latency_ms", ""),
        "rerank_latency_ms": state.get("rerank_latency_ms", ""),
        "generate_latency_ms": state.get("generate_latency_ms", ""),
    }

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    return state


def route_after_stt(state: RAGState) -> str:
    return "blocked" if state.get("blocked") else "proceed"


def route_after_input_guardrail(state: RAGState) -> str:
    return "blocked" if state["blocked"] else "proceed"


def build_app():
    graph = StateGraph(RAGState)
    graph.add_node("speech_to_text", speech_to_text)
    graph.add_node("input_guardrail", input_guardrail)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("output_guardrail", output_guardrail)
    graph.add_node("log_query", log_query)

    graph.set_entry_point("speech_to_text")
    graph.add_conditional_edges(
        "speech_to_text",
        route_after_stt,
        {"blocked": "log_query", "proceed": "input_guardrail"},
    )
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"blocked": "log_query", "proceed": "retrieve"},
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "output_guardrail")
    graph.add_edge("output_guardrail", "log_query")
    graph.add_edge("log_query", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_app()
    result = app.invoke({"query": "होम लोन क्या है?"})
    if result.get("blocked"):
        print("BLOCKED:", result["block_reason"])
    else:
        print("ANSWER:", result["answer"])
        print("GROUNDED:", result["grounded"], "| overlap:", result["overlap_score"])