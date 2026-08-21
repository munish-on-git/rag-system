"""
src/graph/build_graph.py

The harness: audio -> STT (Sarvam) -> input guardrail (Prompt Guard) ->
retrieve -> generate -> output guardrail (grounding check).

Deliberately linear, no retry loop -- that's a scope cut for the deadline,
not a missing piece; a single guardrail-blocks-and-explains path satisfies
requirement #6 without the added latency/complexity of a retry cycle.

Usage (text, for local testing):
    from src.graph.build_graph import build_app
    app = build_app()
    result = app.invoke({"query": "मॉर्गेज क्या है?"})
    print(result["answer"])

Usage (voice, the real pipeline):
    result = app.invoke({"audio_bytes": raw_bytes, "audio_filename": "recording.webm"})
"""

import os
import time
from typing import TypedDict, Optional

import yaml
from groq import Groq
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

from src.indexing.vector_store import VectorStore
from src.stt.sarvam_client import SarvamSTTClient, SarvamSTTError

# Groq: free tier, no credit card, OpenAI-compatible, and fast (LPU hardware) --
# the speed matters directly for the generation-latency numbers.
client = Groq(api_key=os.environ["GROQ_API_KEY"])
stt_client = SarvamSTTClient(api_key=os.environ["SARVAM_API_KEY"])


class RAGState(TypedDict):
    # voice path inputs (either this or `query` is provided at invoke time)
    audio_bytes: Optional[bytes]
    audio_filename: Optional[str]

    query: str
    stt_latency_ms: Optional[float]

    blocked: bool
    block_reason: Optional[str]
    retrieved: list
    answer: str
    grounded: bool


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


def speech_to_text(state: RAGState) -> RAGState:
    """
    Only runs when audio_bytes was provided instead of a text query --
    if the caller already passed `query` directly (e.g. local dev/testing),
    this is a no-op and skips straight through.
    """
    if state.get("query"):
        state["stt_latency_ms"] = None
        return state

    start = time.perf_counter()
    try:
        result = stt_client.transcribe(
            audio_bytes=state["audio_bytes"],
            filename=state.get("audio_filename", "recording.webm"),
            language_code="unknown",
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
    # STT already failed and set blocked=True -- don't overwrite that.
    if state.get("blocked"):
        return state

    # Llama Prompt Guard 2 is a purpose-built classifier, not a chat model:
    # give it the raw query as-is (no instruction wrapper). It returns a
    # single float in message.content -- the probability the input is a
    # jailbreak/injection attempt (0.0 = safe, 1.0 = attack). 0.5 is the
    # threshold recommended in Meta's own model card.
    resp = client.chat.completions.create(
        model=AGENTS_CFG["model"]["guardrail_model"],
        max_tokens=10,
        messages=[{"role": "user", "content": state["query"]}],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        jailbreak_score = float(raw)
    except ValueError:
        # Fallback in case Groq ever changes this to a text label instead
        # of a score -- fail safe (block) rather than crash the harness.
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
    query_emb = EMBEDDER.encode(state["query"], normalize_embeddings=True)
    state["retrieved"] = STORE.search(query_emb, top_k=AGENTS_CFG["retrieval"]["top_k"])
    return state


def generate(state: RAGState) -> RAGState:
    context = "\n\n".join(f"[{i+1}] {r['text']}" for i, r in enumerate(state["retrieved"]))

    prompt = f"""Answer the question using ONLY the numbered context below.
Cite sources inline like [1], [2]. If the context does not contain enough
information to answer, say exactly: "I don't have enough information to answer that."

Context:
{context}

Question: {state['query']}"""

    resp = client.chat.completions.create(
        model=AGENTS_CFG["model"]["generation_model"],
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    state["answer"] = resp.choices[0].message.content.strip()
    return state


def output_guardrail(state: RAGState) -> RAGState:
    if "I don't have enough information" in state["answer"]:
        state["grounded"] = True  # correctly declining counts as grounded
        return state

    # Fast heuristic grounding check: what fraction of the answer's
    # significant words actually appear in the retrieved context?
    # Cheap (no extra LLM call -> no extra latency), good enough as a
    # first-pass hallucination flag under the time budget.
    context_text = " ".join(r["text"] for r in state["retrieved"]).lower()
    answer_words = [w for w in state["answer"].lower().split() if len(w) >= 4]
    if not answer_words:
        state["grounded"] = True
        return state

    overlap = sum(1 for w in answer_words if w in context_text) / len(answer_words)
    state["grounded"] = overlap >= AGENTS_CFG["guardrails"]["min_context_overlap"]

    if not state["grounded"]:
        state["answer"] = ("I don't have enough information to answer that confidently "
                            "based on the retrieved context.")
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
    graph.add_node("generate", generate)
    graph.add_node("output_guardrail", output_guardrail)

    graph.set_entry_point("speech_to_text")
    graph.add_conditional_edges(
        "speech_to_text",
        route_after_stt,
        {"blocked": END, "proceed": "input_guardrail"},
    )
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"blocked": END, "proceed": "retrieve"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "output_guardrail")
    graph.add_edge("output_guardrail", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_app()
    # text path (no audio) -- speech_to_text no-ops through since query is set
    result = app.invoke({"query": "होम लोन क्या है?"})
    if result.get("blocked"):
        print("BLOCKED:", result["block_reason"])
    else:
        print("ANSWER:", result["answer"])
        print("GROUNDED:", result["grounded"])