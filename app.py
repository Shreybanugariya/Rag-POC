"""
=============================================================
 RAG PDF Research Assistant — Streamlit Web UI
 ──────────────────────────────────────────────
 A clean web interface for your RAG assistant.
 
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
    layout="centered",
)

# ─── CUSTOM STYLING ──────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
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
    .timing-badge {
        background-color: #1a1a2e;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        color: #94a3b8;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ─── CACHED INITIALIZATION ───────────────────────────────
# @st.cache_resource ensures these only run ONCE per session,
# not on every rerun (which Streamlit does on each interaction)

@st.cache_resource
def init_embeddings():
    """Load embedding model (cached across reruns)."""
    return create_embeddings()


@st.cache_resource
def init_vector_store(_embeddings, index_path, pdf_path):
    """Load or build vector store (cached across reruns).
    
    Note: _embeddings has underscore prefix to tell Streamlit
    not to hash it (unhashable object).
    """
    if os.path.exists(index_path):
        return load_vector_store(index_path, _embeddings)
    else:
        pages = load_pdf(pdf_path)
        chunks = chunk_documents(pages)
        vs = create_vector_store(chunks, _embeddings)
        save_vector_store(vs, index_path)
        return vs


@st.cache_resource
def init_llm(model_name):
    """Load LLM (cached across reruns)."""
    return setup_llm(model=model_name)


@st.cache_resource
def init_chain(_llm, _vector_store, k):
    """Build RAG chain (cached across reruns)."""
    return create_rag_chain(_llm, _vector_store, k=k)


# ─── SIDEBAR: SETTINGS ───────────────────────────────────
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
        # Clear all cached resources to force rebuild
        import shutil
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown(
        "**Built with** LangChain + FAISS + Ollama\n\n"
        "100% local — no data leaves your machine."
    )


# ─── INIT PIPELINE ───────────────────────────────────────
with st.spinner("Loading models... (first run takes ~10s)"):
    embeddings = init_embeddings()
    vector_store = init_vector_store(embeddings, "faiss_index", pdf_path)
    llm = init_llm(model_name)
    rag_chain = init_chain(llm, vector_store, k_value)


# ─── HEADER ───────────────────────────────────────────────
st.markdown("# 📄 RAG PDF Assistant")
st.markdown(
    "Ask questions about your document. "
    "Answers are grounded in the PDF — no hallucination."
)

# ─── CHAT HISTORY ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander(f"📚 Sources ({msg['source_count']} chunks) • {msg['time']:.1f}s"):
                for src in msg["sources"]:
                    st.markdown(
                        f"<div class='source-box'>"
                        f"<div class='source-header'>📄 Page {src['page']}"
                        f"{' • ' + src['section'] if src['section'] else ''}</div>"
                        f"{src['preview']}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ─── CHAT INPUT ───────────────────────────────────────────
if question := st.chat_input("Ask about your document..."):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching & generating..."):
            start = time.time()
            result = rag_chain.invoke({"query": question})
            elapsed = time.time() - start
        
        # Display answer
        st.markdown(result["result"])
        
        # Parse sources
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "page": doc.metadata.get("page", "?"),
                "section": doc.metadata.get("section", ""),
                "preview": doc.page_content[:200].replace("\n", " "),
            })
        
        # Display sources
        with st.expander(f"📚 Sources ({len(sources)} chunks) • {elapsed:.1f}s"):
            for src in sources:
                st.markdown(
                    f"<div class='source-box'>"
                    f"<div class='source-header'>📄 Page {src['page']}"
                    f"{' • ' + src['section'] if src['section'] else ''}</div>"
                    f"{src['preview']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["result"],
            "sources": sources,
            "source_count": len(sources),
            "time": elapsed,
        })