import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text


def chunk_text(text, chunk_size=800):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    return get_embedder().encode(chunks)


def save_embeddings(chunks, embeddings, chunks_file="chunks.npy", embeddings_file="embeddings.npy"):
    np.save(chunks_file, chunks)
    np.save(embeddings_file, embeddings)


def load_embeddings(chunks_file="chunks.npy", embeddings_file="embeddings.npy"):
    chunks = np.load(chunks_file)
    embeddings = np.load(embeddings_file)
    return chunks, embeddings

