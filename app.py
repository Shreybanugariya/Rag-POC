"""
=============================================================
 RAG PDF Assistant — Streamlit UI v3
 ─────────────────────────────────────
 
 Features:
   - Upload any PDF, remove it, swap it
   - Controls disabled while generating
   - Auto-rebuilds index on PDF change
   - Hybrid retrieval + re-ranking
   - Relevance scores on sources
 
 Run:
   streamlit run app.py
=============================================================
"""

import os
import time
import shutil
import tempfile
import hashlib
import streamlit as st
from rag_assistant import (
    load_pdf,
    chunk_documents,
    create_embeddings,
    create_hybrid_store,
    save_hybrid_store,
    load_hybrid_store,
    create_reranker,
    setup_llm,
    hybrid_retrieve,
    clear_cache,
    RAG_PROMPT_TEMPLATE,
)


# ─── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="centered",
)

st.markdown("""
<style>
    .source-box {
        background-color: #1e1e2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .source-header {
        color: #fbbf24;
        font-weight: 600;
        font-size: 12px;
        margin-bottom: 4px;
    }
    .score-badge {
        background-color: #065f46;
        color: #6ee7b7;
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 11px;
        margin-left: 8px;
    }
    .pdf-status {
        background-color: #1a1a2e;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .pdf-name {
        color: #60a5fa;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE INIT ──────────────────────────────────
# We track everything in session_state so Streamlit reruns
# don't lose our state.

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if "current_pdf_hash" not in st.session_state:
    st.session_state.current_pdf_hash = None

if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None

if "index_data" not in st.session_state:
    st.session_state.index_data = None


# ─── CACHED MODEL LOADING ────────────────────────────────
# These are heavy and only need to load once per session.

@st.cache_resource
def init_embeddings():
    return create_embeddings()

@st.cache_resource
def init_reranker():
    return create_reranker()

@st.cache_resource
def init_llm(model_name):
    return setup_llm(model=model_name)


# ─── PDF MANAGEMENT FUNCTIONS ────────────────────────────

def get_pdf_hash(file_bytes: bytes) -> str:
    """Hash PDF to detect if it's the same file."""
    return hashlib.md5(file_bytes).hexdigest()[:12]


def get_index_path(pdf_hash: str) -> str:
    """Each PDF gets its own index directory."""
    return f"faiss_index_{pdf_hash}"


def save_uploaded_pdf(uploaded_file) -> str:
    """Save uploaded PDF to temp dir, return path."""
    temp_dir = os.path.join(tempfile.gettempdir(), "rag_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    pdf_hash = get_pdf_hash(uploaded_file.getvalue())
    pdf_path = os.path.join(temp_dir, f"{pdf_hash}.pdf")
    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    return pdf_path, pdf_hash


def build_index_for_pdf(pdf_path: str, pdf_hash: str, embeddings):
    """Build or load hybrid index for a given PDF."""
    index_path = get_index_path(pdf_hash)

    if os.path.exists(index_path):
        try:
            vs, bm25, chunks, pmap = load_hybrid_store(index_path, embeddings)
            return {"vector_store": vs, "bm25_index": bm25,
                    "child_chunks": chunks, "parent_map": pmap}
        except Exception:
            shutil.rmtree(index_path, ignore_errors=True)

    pages = load_pdf(pdf_path)
    child_chunks, parent_map = chunk_documents(pages)
    vector_store, bm25_index = create_hybrid_store(child_chunks, embeddings)
    save_hybrid_store(vector_store, bm25_index, child_chunks, parent_map,
                      index_path)

    return {"vector_store": vector_store, "bm25_index": bm25_index,
            "child_chunks": child_chunks, "parent_map": parent_map}


def remove_current_pdf():
    """Remove current PDF and its index, reset chat."""
    if st.session_state.current_pdf_hash:
        idx_path = get_index_path(st.session_state.current_pdf_hash)
        if os.path.exists(idx_path):
            shutil.rmtree(idx_path, ignore_errors=True)

    st.session_state.current_pdf_hash = None
    st.session_state.current_pdf_name = None
    st.session_state.index_data = None
    st.session_state.messages = []
    clear_cache()


# ─── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model_name = st.selectbox(
        "LLM Model",
        ["qwen2.5:3b", "qwen2.5:7b", "mistral", "phi3:mini",
         "gemma2:2b", "llama3.2:3b"],
        index=0,
        help="qwen2.5:3b = fast. 7b = more accurate but slower.",
        disabled=st.session_state.is_generating,
    )

    k_value = st.slider(
        "Final chunks (after re-ranking)", 1, 6, 3,
        disabled=st.session_state.is_generating,
    )

    fetch_k = st.slider(
        "Fetch before re-ranking", 4, 12, 6,
        disabled=st.session_state.is_generating,
    )

    st.markdown("---")

    # ── PDF UPLOAD SECTION ──
    st.markdown("## 📁 Document")

    # Show current PDF status
    if st.session_state.current_pdf_name:
        st.markdown(
            f"<div class='pdf-status'>"
            f"📄 <span class='pdf-name'>{st.session_state.current_pdf_name}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Remove button — disabled while generating
        if st.button(
            "❌ Remove PDF",
            disabled=st.session_state.is_generating,
            help="Remove current PDF and clear chat",
            use_container_width=True,
        ):
            remove_current_pdf()
            st.rerun()

    # Upload widget — disabled while generating
    uploaded_file = st.file_uploader(
        "Upload a PDF" if not st.session_state.current_pdf_name
        else "Replace with another PDF",
        type=["pdf"],
        disabled=st.session_state.is_generating,
        help="Upload any PDF to chat with it"
             + (" (disabled while generating)" if st.session_state.is_generating else ""),
        key="pdf_uploader",
    )

    # Handle new upload
    if uploaded_file is not None:
        new_hash = get_pdf_hash(uploaded_file.getvalue())

        # Only rebuild if it's actually a different PDF
        if new_hash != st.session_state.current_pdf_hash:
            pdf_path, pdf_hash = save_uploaded_pdf(uploaded_file)

            with st.spinner(f"Indexing {uploaded_file.name}..."):
                embeddings = init_embeddings()
                index_data = build_index_for_pdf(pdf_path, pdf_hash, embeddings)

            # Update state
            st.session_state.current_pdf_hash = pdf_hash
            st.session_state.current_pdf_name = uploaded_file.name
            st.session_state.index_data = index_data
            st.session_state.messages = []  # Clear chat for new PDF
            clear_cache()
            st.rerun()

    st.markdown("---")
    st.markdown(
        "**v3** — Speed + Precision\n\n"
        "BGE · BM25+FAISS · Cross-encoder\n\n"
        "100% local — no data leaves your machine."
    )


# ─── LOAD MODELS ─────────────────────────────────────────
# Models load once and stay cached across reruns.
embeddings = init_embeddings()
reranker = init_reranker()
llm = init_llm(model_name)


# ─── HEADER ───────────────────────────────────────────────
st.markdown("# 📄 RAG PDF Assistant")

if not st.session_state.current_pdf_name:
    st.markdown(
        "👈 **Upload a PDF** in the sidebar to get started.\n\n"
        "You can upload any document — employee handbooks, research papers, "
        "contracts, manuals — and ask questions about it."
    )
    st.stop()  # Don't render chat if no PDF loaded

st.markdown(
    f"Chatting with **{st.session_state.current_pdf_name}** · "
    f"Hybrid retrieval + re-ranking for accurate answers."
)


# ─── CHAT HISTORY ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander(
                f"📚 Sources ({msg['source_count']} chunks) · {msg['time']:.1f}s"
            ):
                for src in msg["sources"]:
                    score_html = (
                        f"<span class='score-badge'>"
                        f"relevance: {src['score']:.3f}</span>"
                        if src.get("score") else ""
                    )
                    st.markdown(
                        f"<div class='source-box'>"
                        f"<div class='source-header'>"
                        f"📄 Page {src['page']}"
                        f"{' · ' + src['section'] if src['section'] else ''}"
                        f"{score_html}</div>"
                        f"{src['preview']}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ─── CHAT INPUT ───────────────────────────────────────────
if question := st.chat_input(
    "Ask about your document...",
    disabled=st.session_state.index_data is None,
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # ── Lock UI during generation ──
    st.session_state.is_generating = True

    idx = st.session_state.index_data

    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("🔍 *Retrieving & re-ranking...*")

        start = time.time()

        # Retrieve
        source_docs = hybrid_retrieve(
            question=question,
            vector_store=idx["vector_store"],
            bm25_index=idx["bm25_index"],
            child_chunks=idx["child_chunks"],
            parent_map=idx["parent_map"],
            reranker=reranker,
            fetch_k=fetch_k,
            final_k=k_value,
        )

        status.markdown(
            f"🔍 *Generating answer...*"
        )

        # Build context
        context_parts = []
        for doc in source_docs:
            section = doc.metadata.get("section", "")
            page = doc.metadata.get("page", "?")
            header = (f"[Section: {section}, Page {page}]"
                      if section else f"[Page {page}]")
            context_parts.append(f"{header}\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        # Generate
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        answer = llm.invoke(prompt)
        elapsed = time.time() - start

        # Clear status and show answer
        status.empty()
        st.markdown(answer)

        # Parse sources
        sources = []
        for doc in source_docs:
            sources.append({
                "page": doc.metadata.get("page", "?"),
                "section": doc.metadata.get("section", ""),
                "score": doc.metadata.get("rerank_score", 0),
                "preview": doc.page_content[:250].replace("\n", " "),
            })

        # Display sources
        with st.expander(f"📚 Sources ({len(sources)} chunks) · {elapsed:.1f}s"):
            for src in sources:
                score_html = (
                    f"<span class='score-badge'>"
                    f"relevance: {src['score']:.3f}</span>"
                    if src.get("score") else ""
                )
                st.markdown(
                    f"<div class='source-box'>"
                    f"<div class='source-header'>"
                    f"📄 Page {src['page']}"
                    f"{' · ' + src['section'] if src['section'] else ''}"
                    f"{score_html}</div>"
                    f"{src['preview']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "source_count": len(sources),
            "time": elapsed,
        })

    # ── Unlock UI ──
    st.session_state.is_generating = False
    st.rerun()  # Rerun to re-enable sidebar controls