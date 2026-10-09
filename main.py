import io
import os
import uuid
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import chromadb
from datetime import date
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="AskDocs")

CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

# --- ChromaDB: a real vector store that persists to ./chroma_db on disk ---
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("askdocs")

def embed(text: str) -> list[float]:
    result = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values

def search_documents(query: str) -> str:
    """Search the user's uploaded documents and return the most relevant passages.

    Use this whenever the answer might be in the user's documents, such as their
    resume, notes, or any file they ingested.

    Args:
        query: A natural-language description of what to look for.
    """
    print(f"[tool] search_documents({query!r})")      # so you can SEE it fire
    chunks = retrieve(query, k=5)
    return "\n\n".join(chunks) if chunks else "No relevant documents found."

def get_today_date() -> str:
    """Return today's date in YYYY-MM-DD format.

    Use this when you need the current date, e.g. to calculate how long ago
    something happened or someone's total years of experience.
    """
    print("[tool] get_today_date()")
    return date.today().isoformat()

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def store_chunks(chunks: list[str]):
    for c in chunks:
        collection.add(
            ids=[str(uuid.uuid4())],      # every chunk needs a unique id
            embeddings=[embed(c)],        # we still embed with Gemini...
            documents=[c],                # ...Chroma just stores + indexes it
        )

def retrieve(query: str, k: int = 3) -> list[str]:
    results = collection.query(query_embeddings=[embed(query)], n_results=k)
    return results["documents"][0]        # Chroma does the similarity search for us

# --- endpoints ---
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    response = client.models.generate_content(model=CHAT_MODEL, contents=req.message)
    return {"reply": response.text}

class IngestRequest(BaseModel):
    text: str

@app.post("/ingest")
def ingest(req: IngestRequest):
    chunks = chunk_text(req.text)
    store_chunks(chunks)
    return {"chunks_added": len(chunks), "total_in_db": collection.count()}

@app.post("/ingest-file")
async def ingest_file(file: UploadFile):
    data = await file.read()
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(text)
    store_chunks(chunks)
    return {"filename": file.filename, "chunks_added": len(chunks), "total_in_db": collection.count()}

class SearchRequest(BaseModel):
    query: str

@app.post("/search")
def search(req: SearchRequest):
    return {"results": retrieve(req.query)}

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: AskRequest):
    chunks = retrieve(req.question)
    context = "\n\n".join(chunks)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.question}"
    )
    response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return {"answer": response.text, "sources": chunks}

@app.post("/reset")
def reset():
    chroma.delete_collection("askdocs")
    global collection
    collection = chroma.get_or_create_collection("askdocs")
    return {"status": "cleared"}

@app.post("/agent")
def agent(req: AskRequest):
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=req.question,
        config=types.GenerateContentConfig(
            tools=[search_documents, get_today_date],   # hand the model its toolbox
        ),
    )
    return {"answer": response.text}