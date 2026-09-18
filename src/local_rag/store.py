from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from local_rag.config import Settings


class DocumentStore:
    def __init__(self, settings: Settings) -> None:
        self._embeddings = OllamaEmbeddings(
            model=settings.ollama_embeddings_model, base_url=settings.ollama_base_url
        )
        self._chroma = Chroma(
            collection_name=settings.collection,
            embedding_function=self._embeddings,
            persist_directory=str(settings.persist_dir),
        )
        self._settings = settings

    def add_in_batches(self, chunks: list[Document], batch_size: int = 100) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self._chroma.add_documents(documents=batch)
            print(f"embedded: {i + len(batch)}/{len(chunks)}")

    def retrieve_semantic(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        if k is None:
            k = self._settings.top_k
        return self._chroma.similarity_search_with_relevance_scores(query, k)

    def is_present(self, source: str) -> bool:
        return bool(self._chroma.get(where={"source": source}, limit=1)["ids"])

    def replace_document(self, source: str, chunks: list[Document]) -> None:
        print(f"replacing {source}")
        self._chroma.delete(where={"source": source})
        self.add_in_batches(chunks)

    def delete(self) -> None:
        entries = self._chroma.get()
        ids: list[str] = entries["ids"]
        if ids:
            self._chroma.delete(ids)

    def get_stored_document_amount(self) -> int:
        data = self._chroma.get(include=["metadatas"])
        sources = {m["source"] for m in data["metadatas"]}
        return len(sources)

    def get_stored_chunk_amount(self) -> int:
        return self._chroma._collection.count()
