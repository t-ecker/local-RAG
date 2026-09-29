from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from local_rag.config import Settings
from local_rag.errors import NoDocumentsError
from local_rag.providers import get_embeddings, provider_errors


class DocumentStore:
    def __init__(self, settings: Settings) -> None:
        self._embeddings: Embeddings = get_embeddings(settings)
        self._chroma = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(settings.persist_dir),
        )
        self._settings = settings

    def add_chunks(self, chunks: list[Document], batch_size: int = 100) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            with provider_errors(self._settings.ollama_embeddings_model):
                self._chroma.add_documents(documents=batch)

    def similarity_search(
        self, question: str, selected_sources: list[str], k: int
    ) -> list[tuple[Document, float]]:
        with provider_errors(self._settings.ollama_embeddings_model):
            return self._chroma.similarity_search_with_relevance_scores(
                question, k, filter={"source": {"$in": selected_sources}}
            )

    def contains(self, source: str) -> bool:
        return bool(self._chroma.get(where={"source": source}, limit=1)["ids"])

    def delete_all(self) -> None:
        ids = self._chroma.get()["ids"]
        if ids:
            self._chroma.delete(ids)
        else:
            raise NoDocumentsError(
                "there are no documents in the vector store to delete"
            )

    def delete_source(self, source: str) -> None:
        ids = self._chroma.get(where={"source": source})["ids"]
        if ids:
            self._chroma.delete(ids)

    def list_sources(self) -> set[str]:
        data = self._chroma.get(include=["metadatas"])
        sources: set[str] = {m["source"] for m in data["metadatas"]}
        return sources

    def count_sources(self) -> int:
        return len(self.list_sources())

    def count_chunks(self) -> int:
        return self._chroma._collection.count()

    def get_chunks(self, sources: list[str]) -> list[Document]:
        srcs: list[str | float] = list(sources)
        data = self._chroma.get(where={"source": {"$in": srcs}})
        chunks = []
        for i in range(len(data["ids"])):
            chunks.append(
                Document(
                    page_content=data["documents"][i],
                    metadata=data["metadatas"][i],
                    id=data["ids"][i],
                )
            )
        return chunks
