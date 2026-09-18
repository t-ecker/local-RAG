from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from local_rag.config import Settings
from local_rag.prompts import build_messages
from local_rag.store import DocumentStore


class RagEngine:
    def __init__(self, settings: Settings, document_store: DocumentStore) -> None:
        self._llm = ChatOllama(
            model=settings.ollama_model,
            temperature=settings.temperature,
            base_url=settings.ollama_base_url,
            reasoning=settings.reasoning,
        )
        self._store = document_store

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return self._store.retrieve_semantic(question)

    def stream_answer(self, question: str, retrieved_chunks) -> Iterator[str]:
        messages = build_messages(question, retrieved_chunks)
        for chunk in self._llm.stream(messages):
            yield chunk.text
