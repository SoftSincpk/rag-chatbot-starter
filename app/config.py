"""App settings, read from environment variables or a .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Any OpenAI-compatible API works: OpenAI, Groq, Together, OpenRouter, Ollama...
    llm_api_key: str = ""
    llm_base_url: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    docs_dir: str = "data"
    index_path: str = "index.json"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4


settings = Settings()
