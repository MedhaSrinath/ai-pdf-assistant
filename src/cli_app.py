import requests
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rag_engine import (
    chunk_text,
    create_embeddings,
    load_embeddings,
    load_pdf,
    save_embeddings,
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
PDF_FILE = "sample.pdf"

pdf_stem = Path(PDF_FILE).stem
chunks_file = f"{pdf_stem}_chunks.npy"
embeddings_file = f"{pdf_stem}_embeddings.npy"

# ---- STEP 1: Load or Create embeddings ----

try:
    chunks, chunk_embeddings = load_embeddings(chunks_file, embeddings_file)
    print("Loaded saved embeddings ⚡")
except:
    print("Creating embeddings (first time)...")

    text = load_pdf(PDF_FILE)
    chunks = chunk_text(text)
    chunk_embeddings = create_embeddings(chunks)

    save_embeddings(chunks, chunk_embeddings, chunks_file, embeddings_file)
    print("Embeddings saved ✅")

# ---- STEP 2: Chat loop ----

chat_history = ""

while True:
    question = input("\nAsk (or type 'exit'): ").strip()

    if not question:
        continue

    if question.lower() == "exit":
        break

    # Embed question
    question_embedding = embedder.encode([question])

    # Find top 3 chunks
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    top_k = 3
    top_indices = similarities.argsort()[-top_k:][::-1]

    context = ""
    for i in top_indices:
        context += chunks[i] + "\n\n"

    # Add memory
    prompt = f"""
You are a strict AI assistant.

Answer ONLY using the context below.

Rules:
- Do NOT use prior knowledge
- Do NOT assume anything not in context
- If unsure, say "Not clearly mentioned in document"
- Be concise

Context:
{context}

Question:
{question}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
        },
    )

    data = response.json()
    answer = data["response"]

    print("\nAI:\n", answer)

    # Update memory
    chat_history += f"\nUser: {question}\nAI: {answer}\n"
