# multi-agent-rag
<div align="center">

# 🧠 Multi-Agent RAG System

**A production-grade Retrieval-Augmented Generation pipeline with hybrid search, cross-encoder re-ranking, semantic caching, and auto-evaluation.**


---

## What is this?

Most RAG tutorials show a single-step pipeline: embed → retrieve → generate. This project goes significantly further — it implements a **five-agent sequential pipeline** where each agent has a single responsibility, making the system modular, testable, and production-ready.

Ask it a question about your uploaded documents. It rewrites your query, runs hybrid search, re-ranks candidates with a cross-encoder, reasons over the top chunks, streams the answer token-by-token, and then asynchronously evaluates its own answer quality — all without you seeing any of the machinery.

---

## Pipeline Architecture

```
User Query
     │
     ▼
┌─────────────────────────────┐
│  1. Query Rewriter Agent    │  Expands vague queries into search-optimised versions
│     "what's a bst?" ──────► │  "What is a Binary Search Tree, its properties..."
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  2. Hybrid Retrieval Agent  │  BM25 keyword search + dense vector search
│     ChromaDB + BM25Okapi    │  Merged via Reciprocal Rank Fusion (k=60)
│     → 16 candidates         │  Catches exact terms AND semantic meaning
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  3. Re-ranker Agent         │  Cross-encoder style: scores (query, chunk) jointly
│     16 → top 6 chunks       │  Same principle as Cohere Rerank / MS MARCO
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  4. Reasoning Agent         │  LLM scores each chunk 0–10 for relevance
│     6 → final 3–4 chunks    │  Keeps only chunks scoring ≥ 6
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  5. Response Agent          │  Streams answer token-by-token via SSE
│     Cites sources           │  Uses original query for natural phrasing
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  6. Eval Agent (async)      │  Scores faithfulness / relevance / completeness
│     Runs in background      │  Patches log entry when done — never blocks response
└─────────────────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| **Hybrid Search** | BM25 + vector similarity merged with Reciprocal Rank Fusion |
| **Cross-Encoder Re-ranking** | Joint (query, chunk) scoring — same principle as Cohere Rerank |
| **Query Rewriting** | Expands vague queries; resolves follow-ups using conversation history |
| **Streaming Responses** | Token-by-token SSE stream with blinking cursor in UI |
| **Semantic Cache** | Cosine similarity cache on rewritten queries — ~50ms hits vs ~3s pipeline |
| **Auto-Evaluation** | Async LLM-as-judge scoring: faithfulness, relevance, completeness |
| **File Upload** | Drag-and-drop PDF/DOCX/PPTX/TXT — ingested live, no restart needed |
| **Analytics Dashboard** | Response time p90, eval scores, source hit frequency, filter rates |
| **Conversation Memory** | Last 5 turns injected into rewriter for follow-up resolution |
| **Query Logging** | Every query logged with full metadata to `query_logs.json` |

---

## Project Structure

```
multi-agent-rag/
│
├── agents/
│   ├── query_rewriter_agent.py   # Rewrites queries for better retrieval
│   ├── retrieval_agent.py        # Hybrid BM25 + vector search with RRF
│   ├── reranker_agent.py         # Cross-encoder style re-ranking
│   ├── reasoning_agent.py        # LLM relevance scoring and filtering
│   ├── response_agent.py         # Streaming response generation
│   └── eval_agent.py             # Async LLM-as-judge evaluation
│
├── frontend/
│   ├── index.html                # Chat UI with drag-and-drop upload
│   └── analytics.html            # Live analytics dashboard
│
├── orchestrator.py               # Coordinates all agents in sequence
├── vector_store.py               # ChromaDB collection setup
├── bm25_store.py                 # Persistent BM25 index manager
├── query_cache.py                # Semantic similarity cache
├── ingest.py                     # Document extraction and chunking
├── logger.py                     # Query logging and stats computation
├── main.py                       # FastAPI server and all endpoints
├── requirements.txt
└── .env                          # API keys (not committed)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Atishjha/multi-agent-rag.git
cd multi-agent-rag
pip install -r requirements.txt
```

### 2. Set up API keys

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

Get a free Groq key at [console.groq.com](https://console.groq.com) — no credit card needed.

### 3. Add documents and ingest

```bash
# Drop your PDFs, DOCX, PPTX, or TXT files into the docs/ folder
mkdir docs
cp your-document.pdf docs/

python ingest.py
```

### 4. Start the server

```bash
uvicorn main:app --reload
```

### 5. Open the UI

| URL | What it is |
|---|---|
| `http://localhost:8000` | Chat interface |
| `http://localhost:8000/analytics.html` | Analytics dashboard |
| `http://localhost:8000/docs` | FastAPI interactive API docs |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Non-streaming query (returns full JSON) |
| `POST` | `/query/stream` | Streaming query (SSE, token by token) |
| `POST` | `/upload` | Upload and ingest a document |
| `GET` | `/stats` | Aggregated analytics and eval metrics |
| `GET` | `/logs?limit=N` | Raw query log entries |
| `DELETE` | `/logs` | Clear all query logs |
| `GET` | `/cache/stats` | Cache size and hit threshold |
| `DELETE` | `/cache` | Clear the semantic cache |
| `GET` | `/health` | Server health check |

### Example query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the time complexity of merge sort?"}'
```

```json
{
  "query": "What is the time complexity of merge sort?",
  "rewritten_query": "What is the time and space complexity of merge sort algorithm, and how does it compare to other sorting algorithms?",
  "answer": "Merge sort has a time complexity of O(n log n) in all cases...",
  "sources": ["Data Structures and Algorithms Made Easy.pdf"],
  "chunks_retrieved": 16,
  "chunks_reranked": 6,
  "chunks_used": 3,
  "elapsed_seconds": 2.14,
  "from_cache": false
}
```

---

## NLP Techniques

### Hybrid Search with Reciprocal Rank Fusion

Pure vector search fails on exact terms — class names, error codes, algorithm identifiers. BM25 catches these. The two ranked lists are merged using RRF:

```
score(doc) = 1/(60 + rank_vector) + 1/(60 + rank_bm25)
```

Documents appearing in both lists score highest. `k=60` is the standard constant from Robertson & Zaragoza (2009), used by Elasticsearch, Weaviate, and Pinecone.

### Cross-Encoder Re-ranking

A bi-encoder (standard RAG) embeds query and chunk **separately**. A cross-encoder sees them **together**, letting every query token attend to every chunk token. This agent implements that principle by passing `(query + chunk)` jointly to the LLM — more accurate than cosine similarity, no PyTorch required.

### Semantic Query Cache

Cache keys are **rewritten queries**, not raw user input. This is critical:

```
"what's a bst?"          ─┐
"explain binary search"   ─┤─► same rewritten query ─► same cache entry
"BST definition"          ─┘
```

Threshold is 0.92 cosine similarity. Cache hits return in ~50ms vs ~3s for the full pipeline.

### LLM-as-Judge Evaluation

After every response, an async agent scores the answer on:
- **Faithfulness** — does every claim appear in the source context?
- **Relevance** — does the answer address what was actually asked?
- **Completeness** — are all parts of the question covered?

Scores are stored in the query log and visualised in the analytics dashboard.

---

## Requirements

```
fastapi
uvicorn
chromadb
groq
rank-bm25
pypdf
python-docx
python-pptx
python-dotenv
aiofiles
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Configuration

| Variable | Location | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | `.env` | — | Required. Get free at console.groq.com |
| `top_k` | `orchestrator.py` | 16 | Retrieval candidates per query |
| `score_threshold` | `reranker_agent.py` | 0.4 | Min re-ranker score to keep chunk |
| `DEFAULT_THRESHOLD` | `query_cache.py` | 0.92 | Min similarity for cache hit |
| `MAX_FILE_SIZE` | `main.py` | 200MB | Max upload size |
| `chunk_size` | `ingest.py` | 500 words | Chunk size for document splitting |
| `overlap` | `ingest.py` | 50 words | Overlap between adjacent chunks |

---

## Built By

**Atish Kumar Jha**  
B.Tech CSE · KIIT University · 2023–2027  
[LinkedIn](https://linkedin.com/in/atish-jha-901314308) · [GitHub](https://github.com/Atishjha)

---

<div align="center">
  <sub>Built from scratch — no LangChain, no LlamaIndex, no magic wrappers.</sub>
</div>
