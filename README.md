# RAG Chatbot Starter

A small, readable starter for building a chatbot that answers questions from **your own documents** (retrieval-augmented generation, or RAG).

Drop in `.md`, `.txt` or `.pdf` files, and the bot answers from them and cites its sources. It works with any OpenAI-compatible API: OpenAI, Groq, OpenRouter, Together, or a local model through Ollama.

Built and maintained by [SoftSinc Technologies](https://www.softsincpk.com).

## Features

- **Bring your own documents:** Markdown, plain text and PDF
- **Cited answers:** every reply lists the documents it used
- **Any LLM provider:** set `LLM_BASE_URL` to switch providers, with no code changes
- **Upload from the browser:** add documents in the chat UI and the index rebuilds
- **Minimal dependencies:** FastAPI, NumPy and the OpenAI SDK, with no vector database to run
- **Docker-ready**, with offline tests that don't need an API key

## How it works

```
documents ──► chunk (paragraph-aware, with overlap) ──► embed ──► index.json
                                                                   │
question ──► embed ──► cosine similarity ──► top-k chunks ──► LLM ──► answer + sources
```

The whole pipeline is about 130 lines in [`app/rag.py`](app/rag.py), so it's easy to read and adapt.

## Quick start

```bash
git clone https://github.com/SoftSincpk/rag-chatbot-starter.git
cd rag-chatbot-starter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your API key
uvicorn app.main:app --reload
```

Open http://localhost:8000 and ask: *"What is the refund policy?"*

### With Docker

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 --env-file .env rag-chatbot
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | | API key for your provider |
| `LLM_BASE_URL` | OpenAI | Any OpenAI-compatible endpoint |
| `CHAT_MODEL` | `gpt-4o-mini` | Model used to write answers |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model used for embeddings |
| `DOCS_DIR` | `data` | Folder of documents to index |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Chunking, in characters |
| `TOP_K` | `4` | Number of chunks sent to the model |

Your provider must offer an embeddings endpoint. For local use, Ollama works with models such as `nomic-embed-text` (embeddings) and `llama3.1` (chat).

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | `{"question": "...", "history": []}` returns `{answer, sources}` |
| `POST` | `/documents` | Upload a file (multipart) and rebuild the index |
| `POST` | `/reindex` | Rebuild the index from `DOCS_DIR` |
| `GET` | `/health` | Status and number of indexed chunks |

Interactive docs are available at `/docs`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake model client, so they run offline and for free.

## Taking it to production

This starter keeps things simple on purpose. For production you'll usually want:

- A vector database (pgvector, Qdrant, Pinecone) once you pass a few thousand chunks
- Authentication and rate limiting on the API
- An evaluation set of real questions to measure answer quality
- Streaming responses for a faster-feeling UI

Need a production AI assistant built for your business? [Talk to SoftSinc](https://www.softsincpk.com) or email info@softsincpk.com.

## License

[MIT](LICENSE) © SoftSinc Technologies
