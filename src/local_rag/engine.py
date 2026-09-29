from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from local_rag.config import Settings
from local_rag.errors import (
    InvalidRetrievalModeError,
    NoDocumentsError,
    NoSelectionError,
)
from local_rag.prompts import build_messages
from local_rag.providers import get_llm, provider_errors
from local_rag.retrievers import RETRIEVERS
from local_rag.store import DocumentStore


class RagEngine:
    def __init__(self, settings: Settings, store: DocumentStore) -> None:
        self._llm: BaseChatModel = get_llm(settings)
        self._store = store
        self._settings = settings

    def retrieve(
        self,
        question: str,
        selected_sources: list[str],
        mode: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        if self._store.count_chunks() == 0:
            raise NoDocumentsError("There are no documents in the vector store yet")
        if len(selected_sources) == 0:
            raise NoSelectionError("There are no documents selected")
        if k is None:
            k = self._settings.top_k
        if mode not in RETRIEVERS:
            raise InvalidRetrievalModeError(f"mode: {mode} is not supported")
        return RETRIEVERS[mode](
            self._store, self._settings, question, selected_sources, k
        )

    def stream_answer(
        self, question: str, retrieved_chunks: list[tuple[Document, float]]
    ) -> Iterator[str]:
        messages = build_messages(question, retrieved_chunks)
        with provider_errors(self._settings.ollama_chat_model):
            for chunk in self._llm.stream(messages):
                yield chunk.text
