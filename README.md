# AskDocs — RAG + Agent Document Assistant

AskDocs is a document question-answering API built with FastAPI. Upload a document (PDF or raw text) and ask questions about it — it answers **grounded in the document's contents** using Retrieval-Augmented Generation (RAG), and can **take actions** (search, date calculations) through a tool-calling agent.

## Features

- Ingest PDFs or raw text into a persistent vector store
- Semantic search over your documents (meaning-based, not keyword)
- RAG question-answering grounded in sources — refuses to answer what isn't in the docs
- A tool-calling agent that decides when to search and when to compute
- FastAPI backend with auto-generated interactive docs at `/docs`

## Tech Stack

- **FastAPI** — API framework
- **Google Gemini** — LLM (`gemini-3.8-flash`) + embeddings (`gemini-embedding-001`)
- **ChromaDB** — persistent vector database
- **pypdf** — PDF text extraction
- **Docker** — containerization

## How It Works

1. **Ingest** — documents are split into overlapping chunks, embedded with Gemini, and stored in ChromaDB.
2. **Retrieve** — a question is embedded and the most similar chunks are pulled from ChromaDB.
3. **Augment + Generate** — retrieved chunks are injected into the prompt; Gemini answers using *only* that context (this is what prevents hallucination).
4. **Agent** — for questions that need action (e.g. "how many years of experience is this?"), a tool-calling agent autonomously decides to search the documents and compute from today's date.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Plain LLM chat (no documents) |
| POST | `/ingest` | Ingest raw text |
| POST | `/ingest-file` | Ingest a PDF |
| POST | `/search` | Semantic search — returns matching chunks |
| POST | `/ask` | RAG answer grounded in documents |
| POST | `/agent` | Agent with tools (document search + date) |
| POST | `/reset` | Clear the vector store |

## Setup

```bash
git clone https://github.com/Chirag076/askdocs.git
cd askdocs
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key (get one free at https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your_key_here
```

Run:

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/docs** to try it in the browser.

## Run with Docker

```bash
docker build -t askdocs .
docker run -p 8000:8000 --env-file .env askdocs
```

## What I Learned

- How RAG grounds LLM answers in real documents to prevent hallucination.
- The difference between a fixed RAG pipeline (`/ask`) and an autonomous tool-calling agent (`/agent`) that decides its own steps.
- Trade-offs in chunking, retrieval depth (top-k), and strict-grounding prompts — e.g. a strict RAG pipeline correctly refuses to *calculate* answers that aren't stated, which is exactly the gap an agent fills.

## Possible Improvements

- Re-ranking retrieved chunks for better precision
- Streaming responses
- A web frontend
- An evaluation set to catch quality regressions
- Authentication and per-user document collections
