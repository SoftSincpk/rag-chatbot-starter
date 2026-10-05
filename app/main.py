"""FastAPI server: chat endpoint, document upload, re-indexing and a minimal web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import settings
from .rag import RAG, SUPPORTED

app = FastAPI(title="RAG Chatbot Starter", version="1.0.0")
rag = RAG(settings)
STATIC = Path(__file__).resolve().parent.parent / "static"


@app.on_event("startup")
def startup() -> None:
    if not rag.load() and settings.llm_api_key:
        rag.build_index()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(rag.chunks)}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    return rag.answer(req.question, [m.model_dump() for m in req.history])


@app.post("/documents")
async def upload(file: UploadFile) -> dict:
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, f"Supported types: {', '.join(sorted(SUPPORTED))}")
    dest = Path(settings.docs_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())
    return {"saved": name, "chunks": rag.build_index()}


@app.post("/reindex")
def reindex() -> dict:
    return {"chunks": rag.build_index()}
