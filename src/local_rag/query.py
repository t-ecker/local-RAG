from collections.abc import Iterator

import bm25s
import httpx
import Stemmer  # ty: ignore[unresolved-import]
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from ollama import ResponseError

from local_rag.config import Settings
from local_rag.errors import (
    InvalidRetrieveModeError,
    NoDocumentsError,
    NoSelectionError,
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

    def retrieve(
        self,
        question: str,
        selected_documents: list[str],
        mode: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        if self._store.get_stored_chunk_amount() == 0:
            raise NoDocumentsError("There are no documents in the vector store yet")
        if len(selected_documents) == 0:
            raise NoSelectionError("There are no documents selected")
        if k is None:
            k = self._settings.fallback_top_k
        if mode == "semantic":
            return self._store.retrieve_semantic(question, selected_documents, k)
        if mode == "lexical":
            corpus = self._store.get_documents(selected_documents)
            language = self._settings.corpus_language
            stemmer = Stemmer.Stemmer(language)
            corpus_tokens = bm25s.tokenize(
                [c.page_content for c in corpus], stopwords=language, stemmer=stemmer
            )
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
            query_tokens = bm25s.tokenize(question, stemmer=stemmer)
            chunks, scores = retriever.retrieve(
                query_tokens, k=min(k, len(corpus)), corpus=corpus, return_as="tuple"
            )
            results: list[tuple[Document, float]] = []
            for i in range(len(chunks[0])):
                if scores[0][i] > 0:
                    results.append((chunks[0][i], float(scores[0][i])))
            return results
        raise InvalidRetrieveModeError(f"mode: {mode} is not supported")

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
