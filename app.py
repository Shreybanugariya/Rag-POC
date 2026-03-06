"""
=============================================================
 RAG PDF Research Assistant — Streamlit Web UI
 ──────────────────────────────────────────────
 A clean, responsive web interface for your RAG assistant.

 Run:
   streamlit run app.py

 Requirements:
   pip install streamlit
=============================================================
"""

import os
import time
import streamlit as st
from rag_assistant import (
    load_pdf,
    chunk_documents,
    create_embeddings,
    create_vector_store,
    save_vector_store,
    load_vector_store,
    setup_llm,
    create_rag_chain,
)

# ─── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide",
)

# ─── RESPONSIVE STYLING ──────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    /* ── Root tokens ── */
    :root {
        --bg:          #0d0f17;
        --surface:     #13161f;
        --surface-2:   #1c1f2e;
        --border:      #252836;
        --accent:      #f59e0b;
        --accent-dim:  #78350f;
        --text:        #e2e8f0;
        --text-muted:  #64748b;
        --user-bubble: #1a2744;
        --ai-bubble:   #131a2a;
        --radius:      12px;
        --font-sans:   'IBM Plex Sans', sans-serif;
        --font-mono:   'IBM Plex Mono', monospace;
    }

    /* ── Global resets ── */
    html, body, [class*="css"] {
        font-family: var(--font-sans) !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: clamp(1rem, 3vw, 2.5rem) clamp(1rem, 4vw, 3rem) !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        min-width: 240px !important;
        max-width: 280px !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1.2rem !important;
        max-width: 100% !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stTextInput label {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background-color: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: var(--font-mono) !important;
        font-size: 0.82rem !important;
        color: var(--text) !important;
    }

    /* ── Header ── */
    .rag-header {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1.8rem;
        padding-bottom: 1.2rem;
        border-bottom: 1px solid var(--border);
        flex-wrap: wrap;           /* wraps on small screens */
    }
    .rag-header-icon {
        font-size: clamp(1.8rem, 4vw, 2.6rem);
        line-height: 1;
        flex-shrink: 0;
    }
    .rag-header-text h1 {
        font-family: var(--font-mono) !important;
        font-size: clamp(1.2rem, 3vw, 1.75rem) !important;
        font-weight: 600 !important;
        color: var(--accent) !important;
        margin: 0 0 0.25rem !important;
        letter-spacing: -0.02em;
    }
    .rag-header-text p {
        font-size: clamp(0.78rem, 1.8vw, 0.9rem) !important;
        color: var(--text-muted) !important;
        margin: 0 !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
    }
    /* User bubble */
    [data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
        background: var(--user-bubble);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: clamp(0.6rem, 2vw, 0.9rem) clamp(0.8rem, 2.5vw, 1.2rem);
        font-size: clamp(0.85rem, 2vw, 0.95rem);
        line-height: 1.65;
    }
    /* AI bubble */
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
        background: var(--ai-bubble);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: clamp(0.6rem, 2vw, 0.9rem) clamp(0.8rem, 2.5vw, 1.2rem);
        font-size: clamp(0.85rem, 2vw, 0.95rem);
        line-height: 1.75;
    }

    /* ── Source cards ── */
    .source-card {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: clamp(0.6rem, 2vw, 0.85rem) clamp(0.75rem, 2.5vw, 1.1rem);
        margin-bottom: 0.6rem;
        transition: border-color 0.15s ease;
    }
    .source-card:hover { border-color: var(--accent-dim); }
    .source-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: var(--accent-dim);
        color: var(--accent);
        border-radius: 4px;
        padding: 2px 8px;
        font-family: var(--font-mono);
        font-size: clamp(0.68rem, 1.5vw, 0.75rem);
        font-weight: 600;
        margin-bottom: 0.45rem;
    }
    .source-preview {
        font-size: clamp(0.78rem, 1.8vw, 0.85rem);
        color: var(--text-muted);
        line-height: 1.55;
        word-break: break-word;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        margin-top: 0.6rem !important;
    }
    [data-testid="stExpander"] summary {
        font-size: clamp(0.78rem, 1.8vw, 0.85rem) !important;
        color: var(--text-muted) !important;
        font-family: var(--font-mono) !important;
    }

    /* ── Timing badge ── */
    .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
        margin-top: 0.5rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 3px 10px;
        font-family: var(--font-mono);
        font-size: clamp(0.68rem, 1.5vw, 0.75rem);
        color: var(--text-muted);
    }
    .badge-accent { border-color: var(--accent-dim); color: var(--accent); }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--text) !important;
        font-family: var(--font-sans) !important;
        font-size: clamp(0.85rem, 2vw, 0.95rem) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.8rem !important;
        padding: 0.45rem 1rem !important;
        width: 100% !important;
        transition: border-color 0.15s, background 0.15s !important;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        background: var(--accent-dim) !important;
        color: var(--accent) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--accent) !important; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    /* ══════════════════════════════════════
       RESPONSIVE BREAKPOINTS
    ══════════════════════════════════════ */

    /* Tablet (≤ 900px) — collapse sidebar label padding */
    @media (max-width: 900px) {
        .block-container {
            padding: 1.2rem 1rem !important;
        }
    }

    /* Mobile (≤ 640px) — single-column, tighter spacing */
    @media (max-width: 640px) {
        [data-testid="stSidebar"] {
            min-width: 100vw !important;
            max-width: 100vw !important;
        }
        .block-container {
            padding: 0.8rem 0.6rem !important;
        }
        .rag-header {
            flex-direction: column;
            gap: 0.5rem;
        }
        .source-card { padding: 0.55rem 0.75rem; }
        .meta-row { gap: 0.35rem; }
    }
</style>
""", unsafe_allow_html=True)


# ─── CACHED INITIALIZATION ───────────────────────────────
@st.cache_resource
def init_embeddings():
    return create_embeddings()


@st.cache_resource
def init_vector_store(_embeddings, index_path, pdf_path):
    if os.path.exists(index_path):
        return load_vector_store(index_path, _embeddings)
    pages = load_pdf(pdf_path)
    chunks = chunk_documents(pages)
    vs = create_vector_store(chunks, _embeddings)
    save_vector_store(vs, index_path)
    return vs


@st.cache_resource
def init_llm(model_name):
    return setup_llm(model=model_name)


@st.cache_resource
def init_chain(_llm, _vector_store, k):
    return create_rag_chain(_llm, _vector_store, k=k)


# ─── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model_name = st.selectbox(
        "LLM Model",
        ["mistral", "phi3:mini", "qwen2.5:3b", "gemma2:2b", "llama3.2:3b"],
        index=0,
        help="Smaller models are faster on CPU",
    )

    k_value = st.slider(
        "Chunks to retrieve (k)",
        min_value=1, max_value=6, value=3,
        help="More chunks = more context but slower",
    )

    pdf_path = st.text_input(
        "PDF Path",
        value="documents/WORKPLACE_POLICIES.pdf",
    )

    rebuild = st.button("🔄 Rebuild Index", help="Re-process the PDF")
    if rebuild:
        import shutil
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#475569;font-family:var(--font-mono)'>"
        "LangChain · FAISS · Ollama<br>"
        "100 % local — no data leaves your machine."
        "</small>",
        unsafe_allow_html=True,
    )


# ─── INIT PIPELINE ───────────────────────────────────────
with st.spinner("Loading models… (first run takes ~10 s)"):
    embeddings    = init_embeddings()
    vector_store  = init_vector_store(embeddings, "faiss_index", pdf_path)
    llm           = init_llm(model_name)
    rag_chain     = init_chain(llm, vector_store, k_value)


# ─── HEADER ──────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <div class="rag-header-icon">📄</div>
    <div class="rag-header-text">
        <h1>RAG PDF Assistant</h1>
        <p>Ask questions about your document — answers grounded in the PDF, not guesswork.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── CHAT HISTORY ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            label = (
                f"📚 {msg['source_count']} source chunk"
                f"{'s' if msg['source_count'] != 1 else ''} · {msg['time']:.1f} s"
            )
            with st.expander(label):
                for src in msg["sources"]:
                    section_html = (
                        f" &nbsp;·&nbsp; {src['section']}" if src["section"] else ""
                    )
                    st.markdown(
                        f"<div class='source-card'>"
                        f"<div class='source-tag'>📄 Page {src['page']}{section_html}</div>"
                        f"<div class='source-preview'>{src['preview']}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ─── CHAT INPUT ──────────────────────────────────────────
if question := st.chat_input("Ask anything about your document…"):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching & generating…"):
            start  = time.time()
            result = rag_chain.invoke({"query": question})
            elapsed = time.time() - start

        st.markdown(result["result"])

        sources = [
            {
                "page":    doc.metadata.get("page", "?"),
                "section": doc.metadata.get("section", ""),
                "preview": doc.page_content[:220].replace("\n", " "),
            }
            for doc in result["source_documents"]
        ]

        # Meta row: timing + chunk count
        st.markdown(
            f"<div class='meta-row'>"
            f"<span class='badge badge-accent'>📚 {len(sources)} chunk{'s' if len(sources) != 1 else ''}</span>"
            f"<span class='badge'>⏱ {elapsed:.1f} s</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        label = (
            f"📚 {len(sources)} source chunk{'s' if len(sources) != 1 else ''} · {elapsed:.1f} s"
        )
        with st.expander(label):
            for src in sources:
                section_html = f" &nbsp;·&nbsp; {src['section']}" if src["section"] else ""
                st.markdown(
                    f"<div class='source-card'>"
                    f"<div class='source-tag'>📄 Page {src['page']}{section_html}</div>"
                    f"<div class='source-preview'>{src['preview']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append({
            "role":         "assistant",
            "content":      result["result"],
            "sources":      sources,
            "source_count": len(sources),
            "time":         elapsed,
        })