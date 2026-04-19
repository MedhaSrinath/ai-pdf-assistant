import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.rag_engine import load_pdf, chunk_text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PDF Assistant", layout="centered")

# ---------------- CSS ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* GLOBAL */
.stApp {
    background-color: #f5f5f4;
    font-family: 'Inter', sans-serif;
}

/* MAIN CONTAINER — the boundary card */
.block-container {
    max-width: 900px;
    margin: 2rem auto;
    padding: 2rem 2rem 1.5rem 2rem !important;
    background: #ffffff;
    border-radius: 16px;
    border: 1px solid #e7e5e4;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    min-height: calc(100vh - 4rem);
}

/* TITLE */
h1 {
    font-size: 18px;
    font-weight: 600;
    color: #1c1917;
    text-align: center;
    margin-bottom: 2px;
    letter-spacing: -0.3px;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    font-size: 13px;
    color: #78716c;
    margin-bottom: 1.5rem;
}

/* DIVIDER */
hr {
    border: none;
    border-top: 1px solid #f0eeec;
    margin: 1.25rem 0;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #d6d3d1;
    border-radius: 12px;
    padding: 1.75rem 1.25rem;
    background: #fafaf9;
    transition: all 0.15s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #a8a29e;
    background: #f5f5f4;
}

/* SUCCESS / INFO / WARNING */
[data-testid="stAlert"] {
    border-radius: 10px;
    font-size: 13px;
    padding: 10px 14px;
}

/* CHAT MESSAGES */
.stChatMessage {
    padding: 10px 14px !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
    font-size: 15.5px !important;
    line-height: 1.6 !important;
    max-width: 88% !important;
}

/* USER bubble */
[data-testid="stChatMessage-user"] {
    background: #eff6ff !important;
    border: 0.5px solid #bfdbfe !important;
    border-top-right-radius: 4px !important;
    color: #1e40af !important;
    margin-left: auto !important;
}

/* ASSISTANT bubble */
[data-testid="stChatMessage-assistant"] {
    background: #fafaf9 !important;
    border: 0.5px solid #e7e5e4 !important;
    border-top-left-radius: 4px !important;
    color: #1c1917 !important;
    margin-right: auto !important;
}

/* HIDE avatars */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: none !important;
}

/* CHAT INPUT */
[data-testid="stChatInput"] {
    position: sticky;
    bottom: 10px;
    border-radius: 14px !important;
    border: 1.5px solid #e7e5e4 !important;
    background: #fafaf9 !important;
    padding: 6px !important;
    margin-top: 1rem;
}
[data-testid="stChatInput"] textarea {
    font-size: 15px !important;
    color: #1c1917 !important;
    background: transparent !important;
    padding: 12px 14px !important;
    line-height: 1.5 !important;
}
[data-testid="stChatInput"] textarea:focus {
    box-shadow: none !important;
}

/* CLEAR BUTTON */
.stButton > button {
    font-size: 12px !important;
    font-weight: 400 !important;
    border-radius: 8px !important;
    border: 1px solid #e7e5e4 !important;
    background: transparent !important;
    color: #a8a29e !important;
    padding: 5px 12px !important;
    margin-top: 0.5rem;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #f5f5f4 !important;
    color: #57534e !important;
    border-color: #d6d3d1 !important;
}

/* SPINNER */
[data-testid="stSpinner"] {
    font-size: 13px;
    color: #78716c;
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #e7e5e4; border-radius: 99px; }

/* HIDE streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- HEADER ----------------
st.markdown("<h1>PDF assistant</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a document and ask anything about it</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a PDF to begin", type="pdf", label_visibility="collapsed")

if uploaded_file is not None:
    if "chunks" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Reading and indexing your PDF…"):
            text = load_pdf("temp.pdf")
            chunks = chunk_text(text)
            embeddings = embedder.encode(chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.filename = uploaded_file.name
        st.success(f"✓  {uploaded_file.name}  ·  {len(chunks)} chunks indexed")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- CHAT STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- EMPTY STATE ----------------
if len(st.session_state.history) == 0 and "chunks" not in st.session_state:
    st.markdown(
        "<div style='text-align:center; color:#78716c; padding:20px;'>"
        "Upload a PDF above to start chatting with your document"
        "</div>",
        unsafe_allow_html=True
    )

# ---------------- CHAT DISPLAY ----------------
for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# ---------------- INPUT ----------------
question = st.chat_input("Ask something about your document…")

if question:
    if "chunks" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        chunks = st.session_state.chunks
        embeddings = st.session_state.embeddings

        q_embed = embedder.encode([question])
        similarities = cosine_similarity(q_embed, embeddings)[0]

        top_k = 3
        top_indices = similarities.argsort()[-top_k:][::-1]

        context = ""
        for i in top_indices:
            context += chunks[i] + "\n\n"

        prompt = f"""You are a strict AI assistant.

Answer ONLY using the context below.

Rules:
- Do NOT use prior knowledge
- Do NOT assume anything not in the context
- If the answer is not in the context, say "Not found in document"
- Be concise and clear

Context:
{context}

Question:
{question}
"""

        with st.spinner("Thinking…"):
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False
                }
            )
            answer = response.json()["response"]

        st.session_state.history.append((question, answer))
        st.rerun()

# ---------------- CLEAR CHAT ----------------
if st.session_state.history:
    col1, col2, col3 = st.columns([4, 2, 4])
    with col2:
        if st.button("Clear chat"):
            st.session_state.history = []
            st.rerun()