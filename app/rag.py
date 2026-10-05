"""A small, dependency-light RAG pipeline: load -> chunk -> embed -> retrieve -> answer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

from .config import Settings

SUPPORTED = {".txt", ".md", ".pdf"}

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Cite sources in square brackets, like [handbook.md]."
)


@dataclass
class Chunk:
    text: str
    source: str


def load_documents(docs_dir: str | Path) -> list[tuple[str, str]]:
    """Return (filename, text) for every supported file in docs_dir."""
    docs = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            docs.append((path.name, text))
    return docs


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        while len(para) > size:  # very long paragraph: hard-split it
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:size])
            para = para[size - overlap :]
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}" if tail else para
    if current:
        chunks.append(current)
    return chunks


class RAG:
    def __init__(self, settings: Settings, client: OpenAI | None = None):
        self.s = settings
        self.client = client or OpenAI(
            api_key=settings.llm_api_key or "missing-key", base_url=settings.llm_base_url
        )
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    # ---------- indexing ----------
    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), 64):
            resp = self.client.embeddings.create(model=self.s.embedding_model, input=texts[i : i + 64])
            vectors.extend(d.embedding for d in resp.data)
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)

    def build_index(self) -> int:
        chunks = [
            Chunk(text=c, source=name)
            for name, text in load_documents(self.s.docs_dir)
            for c in chunk_text(text, self.s.chunk_size, self.s.chunk_overlap)
        ]
        self.chunks = chunks
        self.vectors = self._embed([c.text for c in chunks]) if chunks else np.zeros((0, 0), np.float32)
        self.save()
        return len(chunks)

    def save(self) -> None:
        data = {"chunks": [asdict(c) for c in self.chunks], "vectors": self.vectors.tolist()}
        Path(self.s.index_path).write_text(json.dumps(data))

    def load(self) -> bool:
        path = Path(self.s.index_path)
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self.vectors = np.asarray(data["vectors"], dtype=np.float32)
        return True

    # ---------- querying ----------
    def retrieve(self, question: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = self._embed([question])[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[: k or self.s.top_k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        hits = self.retrieve(question)
        context = "\n\n---\n\n".join(f"[{c.source}]\n{c.text}" for c, _ in hits) or "(no documents indexed)"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += (history or [])[-6:]
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
        resp = self.client.chat.completions.create(model=self.s.chat_model, messages=messages, temperature=0.2)
        return {
            "answer": resp.choices[0].message.content,
            "sources": [{"source": c.source, "score": round(s, 3), "preview": c.text[:200]} for c, s in hits],
        }
