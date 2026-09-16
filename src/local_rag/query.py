from collections.abc import Iterator

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
        )
        self._store = document_store

    def stream_answer(self, question: str) -> Iterator[str]:
        results = self._store.retrieve_semantic(question)
        messages = build_messages(question, results)
        for chunk in self._llm.stream(messages):
            yield chunk.text
