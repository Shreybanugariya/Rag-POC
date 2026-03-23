"""
=============================================================
 RAG PDF Research Assistant v3 — Speed + Precision
 ──────────────────────────────────────────────────
 
 v3 changes (speed focus):
   - Default LLM: qwen2.5:3b (2x faster than 7b on CPU)
   - Reduced fetch_k: 6 instead of 10 (fewer re-rank pairs)
   - Tighter num_ctx=1536 and num_predict=200
   - Lazy cross-encoder loading
   - Everything else from v2 (hybrid search, parent-child, BGE)
 
 Usage:
   python rag_assistant.py
   python rag_assistant.py --rebuild
   python rag_assistant.py --model qwen2.5:3b --fast
=============================================================
"""

import os
import sys
import time
import hashlib
import argparse
import uuid
from dotenv import load_dotenv

load_dotenv()

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np
import pickle


# ==========================================================
#  UTILITY
# ==========================================================

def timed(step_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"   ⏱️  {step_name}: {elapsed:.2f}s")
            return result
        return wrapper
    return decorator


_response_cache = {}

def get_cache_key(question: str) -> str:
    return hashlib.md5(question.strip().lower().encode()).hexdigest()

def get_cached(question: str):
    return _response_cache.get(get_cache_key(question))

def set_cached(question: str, result: dict):
    if len(_response_cache) > 50:
        oldest = next(iter(_response_cache))
        del _response_cache[oldest]
    _response_cache[get_cache_key(question)] = result

def clear_cache():
    """Clear all cached responses (call when PDF changes)."""
    _response_cache.clear()


# ==========================================================
#  PHASE 1: INGESTION — Parent-Child Chunking
# ==========================================================

@timed("PDF loading (pdfplumber)")
def load_pdf(file_path: str) -> list:
    """Extract text + tables from PDF using pdfplumber."""
    print(f"\n📄 Loading PDF: {file_path}")

    if not os.path.exists(file_path):
        print(f"   ❌ File not found: {file_path}")
        sys.exit(1)

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                if table:
                    for row in table:
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        table_text += " | ".join(cleaned) + "\n"
                    table_text += "\n"

            combined = text
            if table_text.strip():
                combined += "\n\n[TABLE DATA]\n" + table_text

            if combined.strip():
                doc = Document(
                    page_content=combined,
                    metadata={"source": file_path, "page": i},
                )
                pages.append(doc)

    table_count = sum(1 for p in pages if "[TABLE DATA]" in p.page_content)
    print(f"   ✅ Extracted {len(pages)} pages ({table_count} with tables)")
    return pages


@timed("Parent-child chunking")
def chunk_documents(pages: list,
                    parent_size: int = 1200,
                    parent_overlap: int = 200,
                    child_size: int = 400,
                    child_overlap: int = 100) -> tuple:
    """
    Two-tier chunking: large parents for LLM context,
    small children for precise retrieval.
    """
    print(f"\n✂️  Parent-child chunking")
    print(f"   Parent: size={parent_size}, overlap={parent_overlap}")
    print(f"   Child:  size={child_size}, overlap={child_overlap}")

    separators = [
        "\n\n\n", "\n\n", "\n[TABLE DATA]\n",
        "\nPurpose:", "\nProcedure", "\nPolicy:",
        "\nScope:", "\nEligibility:",
        "\n", ". ", " ", "",
    ]

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size, chunk_overlap=parent_overlap,
        separators=separators, length_function=len,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size, chunk_overlap=child_overlap,
        separators=separators, length_function=len,
    )

    parent_chunks = parent_splitter.split_documents(pages)

    for chunk in parent_chunks:
        text = chunk.page_content
        lines = text.split("\n")
        for line in lines[:3]:
            stripped = line.strip()
            if stripped and (stripped.isupper() or "Policy" in stripped
                            or "POLICY" in stripped or "Leave" in stripped):
                chunk.metadata["section"] = stripped
                break
        if "[TABLE DATA]" in text:
            chunk.metadata["has_table"] = True
        chunk.metadata["parent_id"] = str(uuid.uuid4())

    child_chunks = []
    parent_map = {}

    for parent in parent_chunks:
        children = child_splitter.split_documents([parent])
        for child in children:
            child_id = str(uuid.uuid4())
            child.metadata["child_id"] = child_id
            child.metadata["parent_id"] = parent.metadata["parent_id"]
            if "section" in parent.metadata:
                child.metadata["section"] = parent.metadata["section"]
            if "has_table" in parent.metadata:
                child.metadata["has_table"] = parent.metadata["has_table"]
            child_chunks.append(child)
            parent_map[child_id] = parent

    print(f"   ✅ {len(parent_chunks)} parents → {len(child_chunks)} children")

    sections = set(c.metadata.get("section", "") for c in parent_chunks) - {""}
    if sections:
        print(f"   📑 Detected sections: {len(sections)}")
        for s in sorted(sections)[:8]:
            print(f"      • {s[:60]}")

    return child_chunks, parent_map


# ==========================================================
#  PHASE 2: EMBEDDINGS — BGE
# ==========================================================

@timed("Embedding model load")
def create_embeddings() -> HuggingFaceEmbeddings:
    """Load BGE retrieval-optimized embedding model."""
    print("\n🧠 Loading embedding model: BAAI/bge-small-en-v1.5")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    dim = len(embeddings.embed_query("test"))
    print(f"   ✅ Loaded! {dim}-dimensional vectors (retrieval-optimized)")
    return embeddings


# ==========================================================
#  PHASE 3: HYBRID INDEX (FAISS + BM25)
# ==========================================================

@timed("Vector store + BM25 index creation")
def create_hybrid_store(child_chunks: list, embeddings) -> tuple:
    """Build both FAISS and BM25 indexes."""
    print(f"\n💾 Building hybrid index from {len(child_chunks)} child chunks...")

    vector_store = FAISS.from_documents(
        documents=child_chunks, embedding=embeddings,
    )

    tokenized = [chunk.page_content.lower().split() for chunk in child_chunks]
    bm25_index = BM25Okapi(tokenized)

    print(f"   ✅ FAISS: {vector_store.index.ntotal} vectors")
    print(f"   ✅ BM25:  {len(tokenized)} documents indexed")
    return vector_store, bm25_index


def save_hybrid_store(vector_store, bm25_index, child_chunks, parent_map,
                      path: str = "faiss_index"):
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)
    with open(os.path.join(path, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25_index, f)
    with open(os.path.join(path, "child_chunks.pkl"), "wb") as f:
        pickle.dump(child_chunks, f)
    with open(os.path.join(path, "parent_map.pkl"), "wb") as f:
        pickle.dump(parent_map, f)
    print(f"   💿 Saved hybrid index to ./{path}/")


def load_hybrid_store(path: str, embeddings) -> tuple:
    vector_store = FAISS.load_local(
        path, embeddings, allow_dangerous_deserialization=True
    )
    with open(os.path.join(path, "bm25_index.pkl"), "rb") as f:
        bm25_index = pickle.load(f)
    with open(os.path.join(path, "child_chunks.pkl"), "rb") as f:
        child_chunks = pickle.load(f)
    with open(os.path.join(path, "parent_map.pkl"), "rb") as f:
        parent_map = pickle.load(f)
    print(f"   📂 Loaded hybrid index from ./{path}/")
    return vector_store, bm25_index, child_chunks, parent_map


# ==========================================================
#  PHASE 4: RE-RANKER
# ==========================================================

@timed("Cross-encoder model load")
def create_reranker() -> CrossEncoder:
    """Load lightweight cross-encoder for re-ranking."""
    print("\n🎯 Loading re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("   ✅ Re-ranker ready!")
    return reranker


# ==========================================================
#  PHASE 5: HYBRID RETRIEVAL
# ==========================================================
#
#  SPEED OPTIMIZATION: fetch_k=6 instead of 10
#  ─────────────────────────────────────────────
#  Cross-encoder time scales linearly with candidates.
#  10 candidates → ~3s on CPU
#  6 candidates  → ~1.5s on CPU
#  For a 5-6 page doc, 6 is plenty. Scale up for bigger docs.
# ==========================================================

def hybrid_retrieve(question: str,
                    vector_store: FAISS,
                    bm25_index: BM25Okapi,
                    child_chunks: list,
                    parent_map: dict,
                    reranker: CrossEncoder,
                    fetch_k: int = 6,
                    final_k: int = 3) -> list:
    """Hybrid retrieval: FAISS + BM25 → merge → re-rank → parents."""

    # FAISS (semantic)
    faiss_results = vector_store.similarity_search_with_score(question, k=fetch_k)
    faiss_docs = [(doc, float(score)) for doc, score in faiss_results]

    # BM25 (keyword)
    tokenized_query = question.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
    bm25_docs = [
        (child_chunks[i], float(bm25_scores[i]))
        for i in bm25_top_indices
        if bm25_scores[i] > 0
    ]

    # Merge + deduplicate
    seen = set()
    candidates = []
    for doc, score in faiss_docs + bm25_docs:
        h = hashlib.md5(doc.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            candidates.append(doc)

    if not candidates:
        return []

    # Cross-encoder re-ranking
    pairs = [(question, doc.page_content) for doc in candidates]
    ce_scores = reranker.predict(pairs)

    scored = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    top_children = scored[:final_k]

    # Map children → parents
    seen_parents = set()
    parent_docs = []

    for child_doc, ce_score in top_children:
        child_id = child_doc.metadata.get("child_id")
        parent_id = child_doc.metadata.get("parent_id")

        if parent_id not in seen_parents and child_id in parent_map:
            seen_parents.add(parent_id)
            parent = parent_map[child_id]
            parent.metadata["rerank_score"] = float(ce_score)
            parent_docs.append(parent)
        elif child_id not in parent_map:
            child_doc.metadata["rerank_score"] = float(ce_score)
            parent_docs.append(child_doc)

    return parent_docs


# ==========================================================
#  PHASE 6: LLM + PROMPT
# ==========================================================
#
#  SPEED OPTIMIZATIONS:
#  ────────────────────
#  1. qwen2.5:3b instead of 7b → ~2x faster on CPU
#  2. num_ctx=1536 (down from 2048) → ~25% faster
#  3. num_predict=200 (down from 512) → stops sooner
#
#  For a 5-6 page policy doc with 3 parent chunks (~1200
#  chars each), 1536 context is enough. The prompt + context
#  fits in ~1000-1200 tokens, leaving room for the answer.
#
#  When you move to 100+ page docs, bump num_ctx back to
#  2048 or 4096 and use a 7b model.
# ==========================================================

def setup_llm(model: str = "qwen2.5:3b") -> OllamaLLM:
    """Initialize LLM. Local by default, Gemini optional."""
    print(f"\n🤖 Initializing LLM: {model} (via Ollama)")

    # if use_gemini:
    #     try:
    #         from langchain_google_genai import ChatGoogleGenerativeAI
    #         llm = ChatGoogleGenerativeAI(
    #             model="gemini-2.0-flash",
    #             google_api_key=os.getenv("GOOGLE_API_KEY"),
    #             temperature=0.1,
    #             max_output_tokens=300,
    #         )
    #         print("   ✅ Gemini ready!")
    #         return llm
    #     except Exception as e:
    #         print(f"   ⚠️  Gemini failed: {e}")
    #         print("   ↩️  Falling back to Ollama...")

    llm = OllamaLLM(
        model=model,
        temperature=0.1,
        num_predict=200,    # ← tighter cap, faster stop
        num_ctx=1536,       # ← smaller window, faster inference
    )

    print(f"   ✅ LLM ready! (ctx=1536, max_tokens=200)")
    return llm


RAG_PROMPT_TEMPLATE = """You are a precise document assistant. Answer using ONLY the context below.

RULES:
- Include specific numbers, dates, durations, conditions when available.
- Read [TABLE DATA] row-by-row for exact values.
- If the answer is NOT in the context, say: "This information is not covered in the provided documents."
- Be concise. No fluff.

CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:"""


# ==========================================================
#  PHASE 7: ASK
# ==========================================================

def ask(question: str,
        vector_store: FAISS,
        bm25_index: BM25Okapi,
        child_chunks: list,
        parent_map: dict,
        reranker: CrossEncoder,
        llm,
        fetch_k: int = 6,
        final_k: int = 3) -> dict:
    """Full pipeline: hybrid retrieve → re-rank → generate."""

    cached = get_cached(question)
    if cached:
        print(f"\n{'─'*60}")
        print(f"❓  {question}  ⚡ (cached)")
        print(f"{'─'*60}")
        print(f"\n💡 Answer:\n{cached['result']}")
        return cached

    print(f"\n{'─'*60}")
    print(f"❓  {question}")
    print(f"{'─'*60}")

    start = time.time()

    # Retrieve
    retrieve_start = time.time()
    source_docs = hybrid_retrieve(
        question, vector_store, bm25_index, child_chunks,
        parent_map, reranker, fetch_k, final_k,
    )
    retrieve_time = time.time() - retrieve_start

    if not source_docs:
        result = {"result": "No relevant content found.", "source_documents": []}
        print(f"\n💡 Answer:\n{result['result']}")
        return result

    # Build context
    context_parts = []
    for doc in source_docs:
        section = doc.metadata.get("section", "")
        page = doc.metadata.get("page", "?")
        header = f"[Section: {section}, Page {page}]" if section else f"[Page {page}]"
        context_parts.append(f"{header}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # Generate
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    gen_start = time.time()
    answer = llm.invoke(prompt)
    gen_time = time.time() - gen_start

    elapsed = time.time() - start

    result = {"result": answer, "source_documents": source_docs}
    set_cached(question, result)

    # Display
    print(f"\n💡 Answer:\n{answer}")
    print(f"\n📚 Sources ({len(source_docs)} parent chunks):")
    for i, doc in enumerate(source_docs):
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "")
        has_table = " 📊" if doc.metadata.get("has_table") else ""
        score = doc.metadata.get("rerank_score", 0)
        section_tag = f" [{section}]" if section else ""
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"   [{i+1}] Page {page}{section_tag}{has_table} "
              f"(relevance: {score:.3f}): \"{preview}...\"")
    print(f"\n   ⏱️  Retrieval: {retrieve_time:.3f}s | "
          f"Generation: {gen_time:.2f}s | Total: {elapsed:.2f}s")

    return result


def search_only(vector_store, bm25_index, child_chunks, reranker,
                query, k=5):
    """Debug: show hybrid search results without LLM."""
    print(f"\n🔍 Hybrid search: \"{query}\"")

    faiss_results = vector_store.similarity_search_with_score(query, k=k)
    print(f"\n   FAISS (semantic) top-{k}:")
    for i, (doc, score) in enumerate(faiss_results):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"   [{i+1}] L2={score:.4f}: \"{preview}...\"")

    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::-1][:k]
    print(f"\n   BM25 (keyword) top-{k}:")
    for i, idx in enumerate(bm25_top):
        if bm25_scores[idx] > 0:
            preview = child_chunks[idx].page_content[:80].replace("\n", " ")
            print(f"   [{i+1}] Score={bm25_scores[idx]:.4f}: \"{preview}...\"")

    all_docs = [doc for doc, _ in faiss_results]
    for idx in bm25_top:
        if bm25_scores[idx] > 0:
            all_docs.append(child_chunks[idx])

    seen = set()
    unique = []
    for doc in all_docs:
        h = hashlib.md5(doc.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)

    if unique:
        pairs = [(query, doc.page_content) for doc in unique]
        ce_scores = reranker.predict(pairs)
        scored = sorted(zip(unique, ce_scores), key=lambda x: x[1], reverse=True)
        print(f"\n   Re-ranked top-{k}:")
        for i, (doc, score) in enumerate(scored[:k]):
            preview = doc.page_content[:80].replace("\n", " ")
            section = doc.metadata.get("section", "")
            tag = f" [{section}]" if section else ""
            print(f"   [{i+1}] CE={score:.4f}{tag}: \"{preview}...\"")


# ==========================================================
#  MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="RAG PDF Assistant v3")
    parser.add_argument("--pdf", default="documents/WORKPLACE_POLICIES.pdf")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--fetch-k", type=int, default=6)
    parser.add_argument("--gemini", action="store_true",
                        help="Use Gemini API instead of Ollama")
    args = parser.parse_args()

    INDEX_PATH = "faiss_index"

    print("=" * 60)
    print("  RAG PDF Research Assistant v3 — Speed + Precision")
    print("=" * 60)

    embeddings = create_embeddings()
    reranker = create_reranker()

    if os.path.exists(INDEX_PATH) and not args.rebuild:
        try:
            vector_store, bm25_index, child_chunks, parent_map = \
                load_hybrid_store(INDEX_PATH, embeddings)
        except Exception as e:
            print(f"   ⚠️  Failed to load index: {e}")
            args.rebuild = True

    if args.rebuild or not os.path.exists(INDEX_PATH):
        if os.path.exists(INDEX_PATH):
            import shutil
            shutil.rmtree(INDEX_PATH)
            print("   🗑️  Deleted old index.")
        pages = load_pdf(args.pdf)
        child_chunks, parent_map = chunk_documents(pages)
        vector_store, bm25_index = create_hybrid_store(child_chunks, embeddings)
        save_hybrid_store(vector_store, bm25_index, child_chunks, parent_map,
                          INDEX_PATH)

    llm = setup_llm(model=args.model)

    print("\n" + "=" * 60)
    print("  Ready! Ask anything about your document.")
    print("  Commands: 'quit' | '/search <query>' | '/stats'")
    print("=" * 60)

    while True:
        try:
            question = input("\n❓ Ask your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if question.startswith("/search "):
            search_only(vector_store, bm25_index, child_chunks, reranker,
                        question[8:], k=args.fetch_k)
            continue
        if question == "/stats":
            print(f"\n📊 Index Stats:")
            print(f"   Child vectors:  {vector_store.index.ntotal}")
            n_parents = len(set(c.metadata.get('parent_id') for c in child_chunks))
            print(f"   Parent chunks:  {n_parents}")
            print(f"   Dimensions:     {vector_store.index.d}")
            print(f"   LLM:            {args.model}")
            print(f"   Fetch-k:        {args.fetch_k} → final-k: {args.k}")
            print(f"   Cached:         {len(_response_cache)}")
            continue

        ask(question, vector_store, bm25_index, child_chunks,
            parent_map, reranker, llm, args.fetch_k, args.k)


if __name__ == "__main__":
    main()