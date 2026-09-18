from collections.abc import Iterator

import httpx
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from ollama import ResponseError

from local_rag.config import Settings
from local_rag.errors import (
    NoDocumentsError,
    OllamaUnavailableError,
)
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
        self._settings = settings

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        if self._store.get_stored_chunk_amount() == 0:
            raise NoDocumentsError("There are no documents in the vector store yet")
        return self._store.retrieve_semantic(question)

    def stream_answer(self, question: str, retrieved_chunks) -> Iterator[str]:
        messages = build_messages(question, retrieved_chunks)
        try:
            for chunk in self._llm.stream(messages):
                yield chunk.text
        except (httpx.ConnectError, ConnectionError) as e:
            raise OllamaUnavailableError("Ollama not available") from e
        except ResponseError as e:
            if e.status_code == 404:
                raise OllamaUnavailableError(
                    "Chat-Model not available. Did you pull "
                    f"{self._settings.ollama_model}?"
                ) from e
            raise
