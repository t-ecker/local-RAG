from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from local_rag.config import Settings


class DocumentStore:
    def __init__(self, settings: Settings):
        self.embeddings = OllamaEmbeddings(
            model=settings.ollama_embeddings_model, base_url=settings.ollama_base_url
        )
        self.store = Chroma(
            collection_name=settings.collection,
            embedding_function=self.embeddings,
            persist_directory=str(settings.persist_dir),
        )
        self.settings = settings

    def add_in_batches(self, splits, batch_size=100):
        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            self.store.add_documents(documents=batch)
            print(f"eingebettet: {i + len(batch)}/{len(splits)}")

    def retrieve_by_vector(self, prompt: str, k: int | None = None):
        if k is None:
            k = self.settings.top_k
        return self.store.similarity_search_with_score(prompt, k)
