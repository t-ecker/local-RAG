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

    def add_in_batches(self, chunks: list[Document], batch_size: int = 100) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            with provider_errors(self._settings.ollama_embeddings_model):
                self._chroma.add_documents(documents=batch)
            print(f"embedded: {i + len(batch)}/{len(chunks)}")

    def retrieve_semantic(
        self, query: str, selected_documents: list[str], k: int
    ) -> list[tuple[Document, float]]:
        with provider_errors(self._settings.ollama_embeddings_model):
            return self._chroma.similarity_search_with_relevance_scores(
                query, k, filter={"source": {"$in": selected_documents}}
            )

    def is_present(self, source: str) -> bool:
        return bool(self._chroma.get(where={"source": source}, limit=1)["ids"])

    def replace_document(self, source: str, chunks: list[Document]) -> None:
        print(f"replacing {source}")
        self._chroma.delete(where={"source": source})
        self.add_in_batches(chunks)

    def delete_all(self) -> None:
        entries = self._chroma.get()
        ids: list[str] = entries["ids"]
        if ids:
            self._chroma.delete(ids)
        else:
            raise NoDocumentsError(
                "there are no documents in the vectore store to delete"
            )

    def delete_document(self, source: str) -> None:
        entries = self._chroma.get(where={"source": source})
        self._chroma.delete(entries["ids"])

    def get_stored_documents(self) -> set:
        data = self._chroma.get(include=["metadatas"])
        sources: set[str] = {m["source"] for m in data["metadatas"]}
        return sources

    def get_stored_document_amount(self) -> int:
        return len(self.get_stored_documents())

    def get_stored_chunk_amount(self) -> int:
        return self._chroma._collection.count()

    def get_documents(self, sources: list[str]) -> list[Document]:
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
