import streamlit as st
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rag_engine import load_pdf, chunk_text

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.title("📄 AI PDF Assistant")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Process PDF only once
if uploaded_file is not None:
    if (
        "chunks" not in st.session_state
        or st.session_state.get("uploaded_filename") != uploaded_file.name
    ):

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        text = load_pdf("temp.pdf")
        chunks = chunk_text(text)
        chunk_embeddings = embedder.encode(chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings = chunk_embeddings
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.history = []

        st.success("PDF processed successfully!")

# Ask question
with st.form("question_form", clear_on_submit=True):
    question = st.text_input("Ask a question")
    ask_button = st.form_submit_button("Enter")

# Handle question
if ask_button:
    if not question.strip():
        st.warning("Please enter a question.")
    elif "chunks" not in st.session_state:
        st.warning("Please upload a PDF first!")
    else:
        chunks = st.session_state.chunks
        chunk_embeddings = st.session_state.embeddings

        with st.spinner("Generating answer..."):
            # Embed question
            question_embedding = embedder.encode([question])

            # Similarity
            similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]

            # Top-k retrieval
            top_k = 3
            top_indices = similarities.argsort()[-top_k:][::-1]

            # Build context
            context = ""
            for i in top_indices:
                context += chunks[i] + "\n\n"

            # Prompt
            prompt = f"""
You are a strict AI assistant.

Answer ONLY using the context below.

Rules:
- Do NOT use prior knowledge
- Do NOT assume anything not in context
- If unsure, say "Not found in document"
- Be concise

Context:
{context}

Question:
{question}
"""

            # Call local LLM (Ollama)
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60,
                )
                response.raise_for_status()
                answer = response.json().get("response", "No response received.")
            except requests.RequestException as exc:
                answer = f"Error talking to Ollama: {exc}"
                st.error(answer)

        # Save chat
        st.session_state.history.append((question, answer))

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.history = []

# Chat display (ChatGPT style)
for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        st.markdown(a)
