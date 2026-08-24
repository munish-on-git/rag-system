import os

# Must be set before build_graph.py (and therefore vector_store.py) is
# imported, so VectorStore resolves to the smaller deployed index.
os.environ.setdefault("RAG_INDEX_DIR", "data/index")

import time

import gradio as gr
import spaces

from src.graph.build_graph import build_app

app = build_app()


@spaces.GPU
def answer_question(audio_filepath):
    if audio_filepath is None:
        return "Please record a question first.", "", ""

    start = time.perf_counter()

    with open(audio_filepath, "rb") as f:
        audio_bytes = f.read()

    result = app.invoke({
        "audio_bytes": audio_bytes,
        "audio_filename": audio_filepath,
    })

    total_ms = (time.perf_counter() - start) * 1000

    if result.get("blocked"):
        transcript = result.get("query", "(transcription failed)")
        answer = f"⚠️ {result['block_reason']}"
        meta = f"Total: {total_ms:.0f}ms | STT: {result.get('stt_latency_ms', 'n/a')}ms"
        return transcript, answer, meta

    transcript = result["query"]
    answer = result["answer"]
    grounded = result.get("grounded")
    meta = (
        f"Total: {total_ms:.0f}ms | STT: {result.get('stt_latency_ms', 0):.0f}ms | "
        f"Grounded: {'✅' if grounded else '⚠️ not confidently grounded'}"
    )
    return transcript, answer, meta


with gr.Blocks(title="Voice RAG — HH Goa 2026") as demo:
    gr.Markdown("# 🎙️ Voice-Enabled RAG\nAsk a question by voice (Hindi supported). "
                "Pipeline: Sarvam STT → guardrail → FAISS retrieval → rerank → generation → grounding check.")

    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Ask your question")
    submit_btn = gr.Button("Submit", variant="primary")

    transcript_output = gr.Textbox(label="Transcript (from Sarvam STT)")
    answer_output = gr.Textbox(label="Answer", lines=4)
    meta_output = gr.Textbox(label="Latency & guardrail status")

    submit_btn.click(
        fn=answer_question,
        inputs=[audio_input],
        outputs=[transcript_output, answer_output, meta_output],
    )

if __name__ == "__main__":
    demo.launch()

# --- Deploy notes ---
# Secrets needed in Space Settings -> Repository secrets:
#   GROQ_API_KEY
#   SARVAM_API_KEY