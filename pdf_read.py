import requests
from pypdf import PdfReader

reader = PdfReader("Sample.pdf")

text = ""

for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted
#split into chunks
def chunk_text(text,chunk_size=500):
    chunks=[]
    for i in range(0,len(text),chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = chunk_text(text)

question = input("Ask something about the PDF: ")

prompt = f"""
You are a helpful assistant.

Answer clearly and in simple terms using ONLY the context elow.

Context:
{text[:2000]}

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

if "response" in data:
    print("\nAI says:\n")
    print(data["response"])
else:
    print("Something went wrong:", data.get("error"))

