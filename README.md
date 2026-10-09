# RAG Chatbot 🤖

> **Retrieval-Augmented Generation chatbot** built with LangChain, Google Gemini, and ChromaDB.  
> Upload your documents and ask questions — answers are grounded in your data, not hallucinated.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.0-orange)

---

## ✨ Features

- **Multi-format document support** — PDF, TXT, DOCX, CSV
- **Semantic search** — ChromaDB vector store with Gemini embeddings
- **Anti-hallucination prompts** — 10 engineered strategies (see below)
- **Source citations** — every claim traces back to your documents
- **Conversational memory** — follow-up questions with context
- **Beautiful UI** — glassmorphic dark theme with animations
- **Query reformulation** — optimizes your questions for better retrieval

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Reformulation │  ← Improves retrieval accuracy
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  ChromaDB Retrieval  │  ← Semantic similarity search
│  (Top-K + threshold) │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Context Formatting  │  ← Source metadata + boundaries
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Gemini Generation   │  ← Grounded answer with citations
│  (Anti-hallucination │
│   prompt engineering) │
└─────────────────────┘
```

---

## 🛡️ Prompt Engineering Strategies

| #  | Strategy                  | How It Reduces Hallucinations |
|----|---------------------------|-------------------------------|
| 1  | **Grounding Instructions** | Explicitly limits LLM to provided context only |
| 2  | **Chain-of-Thought**      | Step-by-step reasoning improves factual accuracy |
| 3  | **Confidence Calibration** | Model expresses uncertainty instead of guessing |
| 4  | **Source Attribution**     | Forces `[Source: filename]` citations |
| 5  | **Negative Instruction**  | "Do NOT make up information" constraint |
| 6  | **Role Prompting**        | "Precise research assistant" focuses behavior |
| 7  | **Output Format Control** | Structured output reduces free-form fabrication |
| 8  | **Context Boundaries**    | `<context>` XML tags delineate source material |
| 9  | **Fallback Behavior**     | Defined response when context is insufficient |
| 10 | **Query Reformulation**   | Better queries → better retrieval → less gaps |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
cd RAG-chatbot
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy and edit the environment file
cp .env.example .env
# Add your Google Gemini API key
```

Get your API key from: https://aistudio.google.com/apikey

### 3. Run

```bash
python app.py
```

Open **http://localhost:8000** in your browser.

### 4. Use

1. **Upload** documents via the sidebar (drag & drop or browse)
2. **Ask** questions in the chat input
3. **Review** grounded answers with source citations

---

## 📁 Project Structure

```
RAG-chatbot/
├── app.py                  # FastAPI server & API routes
├── config.py               # Settings & environment config
├── document_processor.py   # Document loading & chunking
├── vector_store.py         # ChromaDB vector store manager
├── rag_chain.py            # LangChain RAG pipeline
├── prompts.py              # Prompt engineering templates
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── static/
    ├── index.html          # Frontend UI
    ├── styles.css          # Dark theme styles
    └── app.js              # Frontend logic
```

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Your Gemini API key (required) |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | LLM model for generation |
| `EMBEDDING_MODEL` | `models/gemini-embedding-2-preview` | Embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum relevance score |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload & ingest a document |
| `POST` | `/api/chat` | Send a question, get a grounded answer |
| `GET` | `/api/stats` | Vector store statistics |
| `POST` | `/api/clear` | Clear all documents and history |
| `POST` | `/api/clear-history` | Clear chat history only |
