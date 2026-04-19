# 📄 ChatPDF – AI Document Assistant

ChatPDF is an AI-powered application that allows users to upload PDF documents and interact with them using natural language. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-based answers from the uploaded document.

---

## 🚀 Features

- 📂 Upload any PDF document  
- 💬 Chat with your PDF like ChatGPT  
- 🔍 Semantic search using embeddings  
- 🤖 Context-aware responses using LLM  
- ⚡ Fast retrieval with top-k similarity  
- 🎯 Strict answering (no hallucination outside document)  
- 🧹 Clear chat functionality  
- 🎨 Clean ChatGPT-style UI  

---

## 🧠 How It Works

1. PDF is uploaded and text is extracted  
2. Text is split into smaller chunks  
3. Each chunk is converted into embeddings  
4. User query is embedded  
5. Cosine similarity is used to find relevant chunks  
6. Relevant context is sent to the LLM  
7. LLM generates an answer based only on that context  

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)  
- **LLM:** Ollama (LLaMA 3)  
- **Similarity Search:** Scikit-learn (Cosine Similarity)  

---

## 📦 Installation

```bash
git clone https://github.com/your-username/chatpdf-ai.git
cd chatpdf-ai
pip install -r requirements.txt
