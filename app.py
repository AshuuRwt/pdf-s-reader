import streamlit as st

# ---------- MUST BE FIRST ----------
st.set_page_config(
    page_title="Chat with PDFs",
    page_icon="📄",
    layout="wide"
)

import os
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import faiss
import requests
from sentence_transformers import SentenceTransformer

from htmlTemplates import css, user_template, bot_template

# ---------------- ENV ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not GROQ_API_KEY or not HF_TOKEN:
    st.error("Missing API keys in .env")
    st.stop()

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------- PDF ----------------
def read_pdfs(files):
    text = ""
    for pdf in files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, size=1200, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# ---------------- VECTOR STORE ----------------
def build_faiss(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

def retrieve(query, index, chunks, k=6):
    q_emb = embedder.encode([query]).astype("float32")
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]

# ---------------- LLM ----------------
def ask_groq(context, question):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "user",
                "content": f"""
Answer strictly using the document context below.

Rules:
- Give a long, detailed explanation
- Use paragraphs and bullet points if needed
- Do NOT add labels like Bot or Assistant
- If information is missing, say: Not found in document.

Context:
{context}

Question:
{question}
"""
            }
        ],
        "temperature": 0.2,
        "max_tokens": 900
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(r.text)

    return r.json()["choices"][0]["message"]["content"]

# ---------------- UI ----------------
def render_chat():
    # Group messages into (user, bot) pairs
    pairs = []
    chat = st.session_state.chat
    for i in range(0, len(chat), 2):
        if i + 1 < len(chat):
            pairs.append((chat[i], chat[i + 1]))

    # Reverse pairs → latest Q&A at top
    for (user_role, user_msg), (bot_role, bot_msg) in reversed(pairs):
        st.markdown(user_template.replace("{{MSG}}", user_msg), unsafe_allow_html=True)
        st.markdown(bot_template.replace("{{MSG}}", bot_msg), unsafe_allow_html=True)

def main():
    st.markdown(css, unsafe_allow_html=True)

    # Session-only state (clears on refresh)
    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.chat = []

    with st.sidebar:
        st.subheader("Upload PDFs")
        files = st.file_uploader(
            "Drag and drop PDFs",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not files:
                st.warning("Upload at least one PDF")
            else:
                with st.spinner("Processing PDFs..."):
                    text = read_pdfs(files)
                    chunks = chunk_text(text)
                    index, chunks = build_faiss(chunks)

                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.chat = []

                st.success("PDFs processed")

    st.title("📄 Chat with your PDFs")

    if not st.session_state.index:
        st.info("Upload PDFs and click **Process PDFs**")
        return

    # -------- FORM (NO DUPLICATES) --------
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input("Ask a question about your PDFs")
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        docs = retrieve(question, st.session_state.index, st.session_state.chunks)
        context = "\n\n".join(docs)
        answer = ask_groq(context, question)

        st.session_state.chat.append(("user", question))
        st.session_state.chat.append(("bot", answer))

    render_chat()

if __name__ == "__main__":
    main()
