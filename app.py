"""
app.py

Gradio app for Hugging Face Spaces -- the "live working link" deliverable.
Wraps the existing harness (src/graph/build_graph.py) with a mic-input UI.

Local run:
    python app.py

Deploy: push this file + requirements.txt + models/ + data/index/ + configs/
to a Hugging Face Space (see deploy notes at the bottom of this file).
"""

import time

import gradio as gr

from src.graph.build_graph import build_app

app = build_app()


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
                "Pipeline: Sarvam STT → guardrail → FAISS retrieval → generation → grounding check.")

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

# --- Deploy to Hugging Face Spaces ---
#
# 1. huggingface.co/new-space -> SDK: Gradio -> create
# 2. In the Space's Files tab (or via git), upload:
#      - this file, renamed to app.py (already named correctly)
#      - requirements.txt (see below)
#      - the full src/ folder
#      - configs/
#      - models/embedder-indic-finetuned/  (your fine-tuned checkpoint)
#      - data/index/  (the FAISS indexes + *_meta.jsonl -- NOT the raw
#        data/chunks or data/processed, those aren't needed at runtime)
# 3. In Space Settings -> Repository secrets, add:
#      GROQ_API_KEY
#      SARVAM_API_KEY
# 4. Space auto-builds and gives you a public URL -- that's your live link.
#
# Size note: models/ + data/index/ together are a few hundred MB. Spaces
# free tier handles this fine, but use git-lfs if pushing via git rather
# than the web upload UI, since GitHub-style small-file uploads choke on
# .faiss/.npy files that size.