from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str
    ollama_embeddings_model: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    fallback_top_k: int = 4
    corpus_language: str = "english"
    temperature: float = 0.1
    reasoning: bool = False
    persist_dir: Path = Path("./chroma_db")
    collection: str = "firstTry"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
