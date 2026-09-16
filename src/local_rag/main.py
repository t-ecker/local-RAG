from local_rag.config import Settings
from local_rag.ingest import load_chunks
from local_rag.query import RagEngine
from local_rag.store import DocumentStore


def main() -> None:
    print("local-RAG!\n\n")
    settings = Settings()
    store = DocumentStore(settings=settings)
    rag_engine = RagEngine(settings=settings, document_store=store)

    pdf: str = "./sample_docs/1.pdf"
    if not store.is_present(pdf):
        print(f"adding {pdf}")
        chunks = load_chunks(settings, pdf)
        store.add_in_batches(chunks)

    while True:
        prompt = input("what do you wanna know? ")
        if prompt == "q":
            break
        for token in rag_engine.stream_answer(prompt):
            print(token, end="", flush=True)
        print("")


if __name__ == "__main__":
    main()
