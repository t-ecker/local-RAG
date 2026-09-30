import re
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_chat_model: str
    ollama_embeddings_model: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    corpus_language: str = "english"
    temperature: float = 0.1
    reasoning: bool = False
    persist_dir: Path = Path("./chroma_db")
    reranker_model: str
    rerank_pool_size: int = 20
    hybrid_pool_size: int = 10

    @property
    def collection_name(self) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]", "-", self.ollama_embeddings_model)
