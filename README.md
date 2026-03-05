# 📄 RAG PDF Research Assistant

A fully local, privacy-first RAG (Retrieval-Augmented Generation) system that lets you chat with your PDF documents using open-source LLMs. No API keys. No cloud. Everything runs on your machine.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-latest-1C3C3C?logo=langchain)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What It Does

You give it a PDF → it chunks, embeds, and indexes the content → you ask questions in natural language → it retrieves relevant sections and generates accurate, sourced answers using a local LLM.

**Example:**
```
❓ You: What is the maternity leave policy?

💡 Answer: According to the Workplace Policies document, employees are entitled
   to 26 weeks of paid maternity leave after completing 80 days of service...

📚 Sources:
   [1] Page 12: "Maternity Leave Policy — Purpose: To provide..."
   [2] Page 13: "Eligibility: All female employees who have..."

⏱️  Response time: 4.82s
```

---

## Tech Stack

| Layer | Technology | Why This Choice |
|---|---|---|
| **LLM** | [Ollama](https://ollama.com) + Mistral 7B | Runs locally, no API costs, good quality |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | Fast, 384-dim vectors, great for semantic search |
| **Vector Store** | FAISS (Facebook AI Similarity Search) | Sub-millisecond search, persists to disk |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Respects sentence boundaries, configurable overlap |
| **Orchestration** | LangChain `RetrievalQA` chain | Handles retrieval → prompt → generation pipeline |
| **PDF Parsing** | `PyPDFLoader` | Reliable page-by-page extraction with metadata |

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   PDF File   │────▶│  PyPDFLoader │────▶│  Page Documents  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                                   ▼
                                         ┌──────────────────┐
                                         │ RecursiveCharText │
                                         │    Splitter       │
                                         │ (1000 chars,      │
                                         │  200 overlap)     │
                                         └────────┬─────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  FAISS Index │◀────│ all-MiniLM   │◀────│  Text Chunks     │
│  (on disk)   │     │ L6-v2 (384d) │     │  (23 vectors)    │
└──────┬───────┘     └──────────────┘     └──────────────────┘
       │
       │  similarity search (top-k=4)
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Retrieved   │────▶│  Prompt      │────▶│  Ollama/Mistral  │
│  Chunks      │     │  Template    │     │  (local LLM)     │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │  Answer + Sources │
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
ollama pull mistral              # Default — 4.1GB, good quality
# OR for faster responses on CPU:
ollama pull phi3:mini            # 2.3GB, 2-3x faster
ollama pull gemma2:2b            # 1.6GB, fastest option
```

### 3. Add Your PDF

```bash
mkdir -p documents
cp /path/to/your/file.pdf documents/
```

### 4. Run

```bash
python rag_assistant.py --pdf documents/your_file.pdf
```

First run builds the FAISS index (takes ~10-30s depending on PDF size). Subsequent runs load instantly from disk.

---

## Usage

### CLI Commands

```bash
# Basic usage
python rag_assistant.py

# Use a specific PDF
python rag_assistant.py --pdf documents/handbook.pdf

# Force re-index (after updating the PDF)
python rag_assistant.py --rebuild

# Use a different/faster model
python rag_assistant.py --model phi3:mini

# Retrieve fewer chunks (faster responses)
python rag_assistant.py --k 2
```

### Interactive Commands

| Command | Description |
|---|---|
| Type any question | Ask about your document |
| `/search <query>` | Raw FAISS search — no LLM, just retrieval (great for debugging) |
| `/stats` | Show index stats (vector count, dimensions, model) |
| `quit` | Exit |

---

## Configuration

### CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--pdf` | `documents/WORKPLACE_POLICIES.pdf` | Path to PDF file |
| `--rebuild` | `false` | Force rebuild the FAISS index |
| `--model` | `mistral` | Ollama model name |
| `--k` | `4` | Number of chunks to retrieve per query |

### Performance Tuning (in code)

In `setup_llm()`, you can uncomment and adjust:

```python
llm = OllamaLLM(
    model=model,
    temperature=0.2,
    num_predict=512,         # Max tokens in response
    # num_ctx=2048,          # Context window (lower = faster)
    # num_thread=8,          # CPU threads (match your core count)
    # num_gpu=1,             # GPU layers (NVIDIA only)
)
```

### Chunking Parameters (in code)

```python
chunk_documents(pages, chunk_size=1000, chunk_overlap=200)
# chunk_size  → larger = more context per chunk, fewer chunks
# chunk_overlap → prevents information loss at chunk boundaries
```

---

## Performance Benchmarks

Tested on a consumer laptop (no GPU):

| Step | Time | Notes |
|---|---|---|
| Embedding model load | ~2-5s | One-time per session |
| PDF load + chunking | ~1-3s | One-time, cached to disk |
| FAISS index build | ~5-15s | One-time, saved to `faiss_index/` |
| FAISS similarity search | <0.01s | Effectively instant |
| LLM generation (Mistral 7B, CPU) | 5-20s | **The bottleneck** — use a smaller model for speed |
| LLM generation (phi3:mini, CPU) | 2-8s | 2-3x faster than Mistral |
| LLM generation (any model, GPU) | 0.5-3s | 10-50x faster with NVIDIA GPU |

---

## Project Structure

```
rag-pdf-assistant/
├── README.md                  # You're reading this
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore                 # Ignores faiss_index/, .env, __pycache__/
├── rag_assistant.py           # Main CLI application
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
pypdf
python-dotenv
```

---

## Roadmap

Planned improvements, roughly in priority order:

### Speed & Performance
- [ ] Smaller model support (`phi3:mini`, `gemma2:2b`) for faster CPU inference
- [ ] GPU acceleration documentation
- [ ] Context window tuning (`num_ctx`)

### Answer Quality
- [ ] Section-aware chunking (respect document structure)
- [ ] Domain-specific prompt templates
- [ ] Hybrid search (BM25 keyword + FAISS vector)
- [ ] Multi-query retrieval (LLM rewrites your question 3 ways)
- [ ] Cross-encoder re-ranking (fetch 10 chunks → re-rank → keep best 4)

### Features
- [ ] Streamlit web UI
- [ ] Multi-PDF support (scan a directory)
- [ ] Conversation memory (follow-up questions)
- [ ] FastAPI REST API wrapper

### Evaluation
- [ ] RAGAS evaluation framework (faithfulness, relevancy, precision)
- [ ] Test dataset for automated quality checks

---

## How It Works (Under the Hood)

**1. Ingestion** — The PDF is loaded page-by-page using `PyPDFLoader`, preserving page number metadata. Pages are then split into overlapping chunks of ~1000 characters using `RecursiveCharacterTextSplitter`, which tries to split at natural boundaries (paragraphs → sentences → words).

**2. Embedding** — Each chunk is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer. This captures the semantic meaning of the text — chunks about similar topics end up near each other in vector space.

**3. Indexing** — Vectors are stored in a FAISS flat index (`IndexFlatL2`), which performs exact nearest-neighbor search. The index is persisted to disk so you only pay the embedding cost once.

**4. Retrieval** — When you ask a question, your question is embedded using the same model, then FAISS finds the top-k most similar chunks by L2 distance. This takes <1ms even for thousands of vectors.

**5. Generation** — The retrieved chunks are injected into a prompt template along with your question, then sent to the local LLM (Mistral via Ollama). The LLM generates an answer grounded in the retrieved context.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection error` to Ollama | Make sure Ollama is running: `ollama serve` |
| Slow responses (>15s) | Switch to a smaller model: `--model phi3:mini` |
| Wrong/hallucinated answers | Try `--k 6` to retrieve more context, or check `/search` results |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your venv |
| Index seems stale | Rebuild with `--rebuild` flag |

---

## Contributing

Contributions are welcome! Feel free to open issues or submit PRs for any of the roadmap items above.

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

Built with [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com/), [FAISS](https://github.com/facebookresearch/faiss), and [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers).