# 📄 RAG PDF Research Assistant

A fully local, privacy-first RAG (Retrieval-Augmented Generation) system that lets you chat with your PDF documents using open-source LLMs. No API keys. No cloud. Everything runs on your machine.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-latest-1C3C3C?logo=langchain)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What It Does

You give it a PDF → it chunks, embeds, and indexes the content → you ask questions in natural language → it retrieves relevant sections and generates accurate, sourced answers using a local LLM.

Comes with both a **CLI** and a **Streamlit web UI** with real-time streaming.

**Example:**
```
❓ You: What is the maternity leave policy?

💡 Answer: According to the Workplace Policies document, employees are entitled
   to 26 weeks of paid maternity leave after completing 80 days of service...

📚 Sources:
   [1] Page 12 [MATERNITY LEAVE POLICY]: "Maternity Leave Policy — Purpose..."
   [2] Page 13 [ELIGIBILITY] 📊: "All female employees who have..."

⏱️  Response time: 4.82s
```

---

## Features

- **Streaming responses** — tokens appear in real-time, no staring at a loading spinner
- **Table extraction** — reads tables from PDFs using `pdfplumber` (not just plain text)
- **Section-aware chunking** — respects document structure (policy headers, sections, eligibility blocks)
- **Response caching** — repeated questions return instantly
- **Dual interface** — CLI for power users, Streamlit web UI for everyone else
- **Fully local** — runs on CPU, no GPU required, no data leaves your machine

---

## Tech Stack

| Layer | Technology | Why This Choice |
|---|---|---|
| **LLM** | [Ollama](https://ollama.com) + Mistral 7B | Runs locally, no API costs, good quality |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | Fast, 384-dim vectors, great for semantic search |
| **Vector Store** | FAISS (Facebook AI Similarity Search) | Sub-millisecond search, persists to disk |
| **PDF Parsing** | `pdfplumber` | Extracts both text and table structures |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Section-aware separators, configurable overlap |
| **Orchestration** | LangChain `RetrievalQA` chain | Handles retrieval → prompt → generation pipeline |
| **Web UI** | Streamlit | Streaming chat interface with source display |

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│   PDF File   │────▶│  pdfplumber  │────▶│  Pages + Tables   │
└─────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                                                   ▼
                                         ┌───────────────────┐
                                         │ Section-Aware      │
                                         │ RecursiveCharText   │
                                         │ Splitter (1000c,    │
                                         │ 200 overlap)        │
                                         └────────┬──────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  FAISS Index │◀────│ all-MiniLM   │◀────│  Enriched Chunks  │
│  (on disk)   │     │ L6-v2 (384d) │     │  + section meta   │
└──────┬───────┘     └──────────────┘     └──────────────────┘
       │
       │  similarity search (top-k=3)
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Retrieved   │────▶│  HR-Specific │────▶│  Ollama/Mistral  │
│  Chunks      │     │  Prompt      │     │  (local, stream)  │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │  Streamed Answer  │
                                          │  + Sources + Cache │
                                          └──────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/rag-pdf-assistant.git
cd rag-pdf-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Pull an LLM

```bash
ollama pull mistral              # Default — 4.1GB, best quality
# OR for faster responses on CPU:
ollama pull phi3:mini            # 2.3GB, 2-3x faster
ollama pull qwen2.5:3b           # 2GB, good speed/quality balance
```

### 3. Add Your PDF

```bash
mkdir -p documents
cp /path/to/your/file.pdf documents/
```

### 4. Run

**Web UI (recommended):**
```bash
streamlit run app.py
```

**CLI:**
```bash
python rag_assistant.py --pdf documents/your_file.pdf
```

First run builds the FAISS index (~10-30s depending on PDF size). Subsequent runs load instantly from disk.

---

## Usage

### Web UI

Launch with `streamlit run app.py`. Features:

- **Streaming chat** — responses appear token-by-token
- **Source display** — expandable sources with page numbers and section tags
- **Sidebar settings** — switch models, adjust k, rebuild index
- **Response caching** — repeated questions return instantly (⚡ badge)
- **Table indicators** — 📊 marks sources that contain table data

### CLI Commands

```bash
# Basic usage
python rag_assistant.py

# Use a specific PDF
python rag_assistant.py --pdf documents/handbook.pdf

# Force re-index (after updating the PDF or changing chunking)
python rag_assistant.py --rebuild

# Use a different/faster model
python rag_assistant.py --model phi3:mini

# Retrieve fewer chunks (faster responses)
python rag_assistant.py --k 2
```

### Interactive CLI Commands

| Command | Description |
|---|---|
| Type any question | Ask about your document |
| `/search <query>` | Raw FAISS search — no LLM, just retrieval (great for debugging) |
| `/stats` | Show index stats (vector count, dimensions, model, cache size) |
| `quit` | Exit |

---

## Configuration

### CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--pdf` | `documents/WORKPLACE_POLICIES.pdf` | Path to PDF file |
| `--rebuild` | `false` | Force rebuild the FAISS index |
| `--model` | `mistral` | Ollama model name |
| `--k` | `3` | Number of chunks to retrieve per query |

### Performance Tuning

These are already optimized in `setup_llm()` for CPU-only systems:

```python
llm = OllamaLLM(
    model=model,
    temperature=0.1,         # Deterministic for factual Q&A
    num_predict=256,         # Caps answer length, prevents rambling
    num_ctx=2048,            # 40% faster than default 4096
    # num_thread=8,          # Set to your CPU core count
)
```

---

## Optimizations Applied

| Optimization | Impact | Details |
|---|---|---|
| **pdfplumber** for PDF parsing | Reads tables properly | Tables extracted as pipe-separated text, tagged in metadata |
| **Section-aware chunking** | Better retrieval | Splits on `Purpose:`, `Policy:`, `Eligibility:`, etc. |
| **Section metadata** | Traceable sources | Each chunk tagged with detected section title |
| **`num_ctx=2048`** | ~40% faster | Halved context window, sufficient for policy Q&A |
| **`num_predict=256`** | Faster, focused | Prevents runaway generation |
| **`temperature=0.1`** | More accurate | Deterministic outputs for factual content |
| **Domain-specific prompt** | Much better answers | Lists ALL items, cites numbers/deadlines, reads tables |
| **Response caching** | Instant repeats | MD5-based cache, last 50 questions |
| **Streaming (UI)** | ~1s perceived latency | Tokens appear in real-time via `st.write_stream` |
| **`k=3`** (down from 4) | Less noise, faster | 3 focused chunks > 4 noisy ones |

---

## Performance Benchmarks

Tested on AMD Ryzen 7 5700U (CPU only, no discrete GPU):

| Step | Time | Notes |
|---|---|---|
| Embedding model load | ~2-5s | One-time per session |
| PDF load + table extraction | ~1-3s | One-time, cached to disk |
| FAISS index build | ~5-15s | One-time, saved to `faiss_index/` |
| FAISS similarity search | <0.01s | Effectively instant |
| LLM generation (Mistral, CPU) | ~8-12s | Optimized with num_ctx=2048 |
| LLM generation (cached) | <0.01s | Instant on repeated questions |
| Streaming first token | ~1s | Perceived latency with streaming |

---

## Project Structure

```
rag-pdf-assistant/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore                 # Ignores faiss_index/, .env, __pycache__/
├── rag_assistant.py           # Core RAG logic + CLI interface
├── app.py                     # Streamlit web UI (streaming)
├── documents/                 # Your PDF files go here
│   └── sample.pdf
├── faiss_index/               # Auto-generated, git-ignored
│   ├── index.faiss
│   └── index.pkl
└── tests/
    └── test_rag.py
```

---

## Dependencies

```
langchain-community
langchain-huggingface
langchain-ollama
langchain-classic
langchain-text-splitters
langchain-core
faiss-cpu
sentence-transformers
pdfplumber
streamlit
python-dotenv
```

---

## Roadmap

### Completed
- [x] Section-aware chunking with metadata enrichment
- [x] Domain-specific HR prompt template
- [x] pdfplumber for table extraction
- [x] Response caching (MD5-based, in-memory)
- [x] Streamlit web UI with streaming
- [x] Performance tuning (num_ctx, num_predict, k)

### Next Up
- [ ] Hybrid search (BM25 keyword + FAISS vector)
- [ ] Conversation memory (follow-up questions)
- [ ] Multi-PDF support (scan a directory)
- [ ] Cross-encoder re-ranking (fetch 10 → re-rank → keep best 3)
- [ ] Multi-query retrieval (LLM rewrites your question 3 ways)
- [ ] FastAPI REST API wrapper
- [ ] RAGAS evaluation framework

---

## How It Works

**1. Ingestion** — The PDF is loaded using `pdfplumber`, which extracts both regular text and table structures page-by-page. Tables are converted to pipe-separated format and tagged with `[TABLE DATA]` markers. Pages are then split into overlapping chunks using section-aware separators that respect document structure (policy headers, eligibility blocks, etc.). Each chunk is enriched with metadata including page number, detected section title, and whether it contains table data.

**2. Embedding** — Each chunk is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer. Embeddings are normalized for consistent similarity scores.

**3. Indexing** — Vectors are stored in a FAISS flat index (`IndexFlatL2`), persisted to disk so you only pay the embedding cost once.

**4. Retrieval** — Your question is embedded with the same model, then FAISS finds the top-3 most similar chunks. This takes <1ms.

**5. Generation** — Retrieved chunks are injected into a domain-specific prompt template and streamed through the local LLM. Responses are cached for instant retrieval on repeated questions.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection error` to Ollama | Make sure Ollama is running: `ollama serve` |
| Slow responses (>15s) | Try `--model phi3:mini` or lower `--k 2` |
| Wrong/hallucinated answers | Try `--k 5` for more context, or check `/search` results |
| Tables not being read | Run `--rebuild` to re-index with pdfplumber |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your venv |
| Index seems stale | Rebuild with `--rebuild` flag |
| Streamlit won't start | Ensure `rag_assistant.py` is in the same directory as `app.py` |

---

## Contributing

Contributions welcome! Feel free to open issues or submit PRs for any roadmap items.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/hybrid-search`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## License

MIT — use it however you want.

---

## Acknowledgments

Built with [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com/), [FAISS](https://github.com/facebookresearch/faiss), [pdfplumber](https://github.com/jsvine/pdfplumber), [Streamlit](https://streamlit.io/), and [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers).