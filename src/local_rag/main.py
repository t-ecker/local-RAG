from local_rag.config import Settings
from local_rag.ingest import ingest_documents
from local_rag.llm import LlmEngine
from local_rag.store import DocumentStore


def main() -> None:
    print("local-RAG!\n\n")
    settings = Settings()
    store = DocumentStore(settings=settings)
    engine = LlmEngine(settings=settings, documentStore=store)

    docs = ingest_documents(settings)
    store.add_in_batches(docs)

    while True:
        prompt = input("what do you wanna know? ")
        if prompt == "q":
            break
        for token in engine.ask_llm(prompt):
            print(token, end="", flush=True)
        print("")


if __name__ == "__main__":
    main()
