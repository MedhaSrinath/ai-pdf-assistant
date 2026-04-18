import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity 

#load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

reader = PdfReader("Sample.pdf")

text = ""
for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted

#split into chunks
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0,len(text), chunk_size)]

chunks = chunk_text(text)

#create embeddings for each chunk
chunk_embeddings = embedder.encode(chunks)

question = input("Ask something about the PDF: ")

question_embedding = embedder.encode([question])

#step6: find best chunk using similarity
similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
best_index = similarities.argmax()
best_chunk = chunks[best_index]

prompt = f"""
You are a helpful assistant.

Answer ONLU using the context below.

Context:
{best_chunk}

Question:
{question}
"""

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
)

data = response.json()

# Step 9: Output
print("\nAI says:\n")
print(data["response"])