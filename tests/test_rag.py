"""Offline tests: a fake OpenAI client stands in for the real API."""

from types import SimpleNamespace

import numpy as np

from app.config import Settings
from app.rag import RAG, chunk_text

VOCAB = ["refund", "shipping", "warranty", "hours", "price"]


def fake_embed(text: str) -> list[float]:
    words = text.lower()
    return [float(words.count(w)) + 0.01 for w in VOCAB]


class FakeClient:
    def __init__(self):
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.last_messages = None

    def _embed(self, model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=fake_embed(t)) for t in input])

    def _chat(self, model, messages, temperature):
        self.last_messages = messages
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Refunds take 5 days [policy.md]"))])


def make_rag(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.md").write_text("Our refund policy: refund within 5 days.\n\nShipping is free over $50.")
    (docs / "hours.txt").write_text("Office hours are 9 to 5. Hours on weekends vary.")
    s = Settings(docs_dir=str(docs), index_path=str(tmp_path / "index.json"), chunk_size=60, chunk_overlap=10)
    return RAG(s, client=FakeClient())


def test_chunk_text_respects_size():
    text = "\n\n".join(["word " * 30] * 5)
    chunks = chunk_text(text, size=100, overlap=20)
    assert chunks and all(len(c) <= 100 for c in chunks)


def test_index_retrieve_and_answer(tmp_path):
    rag = make_rag(tmp_path)
    assert rag.build_index() >= 2
    top_chunk, _ = rag.retrieve("what is the refund policy?", k=1)[0]
    assert top_chunk.source == "policy.md"

    result = rag.answer("refund?")
    assert "[policy.md]" in result["answer"]
    assert result["sources"][0]["source"] == "policy.md"
    assert "Context:" in rag.client.last_messages[-1]["content"]


def test_index_persists(tmp_path):
    rag = make_rag(tmp_path)
    n = rag.build_index()
    fresh = RAG(rag.s, client=FakeClient())
    assert fresh.load() and len(fresh.chunks) == n
    assert np.allclose(np.linalg.norm(fresh.vectors, axis=1), 1.0, atol=1e-5)
