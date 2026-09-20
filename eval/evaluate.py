import json
from pathlib import Path

from langchain_core.documents import Document

from local_rag.config import Settings
from local_rag.ingest import load_chunks
from local_rag.query import RagEngine
from local_rag.store import DocumentStore

TOP_K_LIST = [1, 3, 5]


def is_chunk_correct(test, chunk: Document):
    metadata = chunk.metadata
    if Path(metadata["source"]).name == test["expected_source"]:
        chunk_start = metadata["start_index"]
        chunk_end = chunk_start + len(chunk.page_content)
        overlaps = (
            chunk_start < test["answer_char_end"]
            and chunk_end > test["answer_char_start"]
        )
        if overlaps:
            if Path(metadata["source"]).suffix.lower() == ".pdf":
                if metadata["page"] == test["expected_page"]:
                    return True
            else:
                return True
    return False


def check_chunks(test, retrieved_chunks):
    for i, (chunk, _) in enumerate(retrieved_chunks, start=1):
        if is_chunk_correct(test, chunk):
            return i
    return None


def evaluate_retrieval(data_set, retrieve_fn):
    hits_recall: dict[int, float] = dict.fromkeys(TOP_K_LIST, 0.0)
    hits_mrr: dict[int, float] = dict.fromkeys(TOP_K_LIST, 0.0)
    amount_valid_tests: int = 0
    for test in data_set:
        if test["expected_source"] is None:
            continue
        amount_valid_tests += 1
        retrieved_chunks = retrieve_fn(test["question"], max(TOP_K_LIST))
        for top_k in TOP_K_LIST:
            if (i := check_chunks(test, retrieved_chunks[:top_k])) is not None:
                hits_recall[top_k] += 1.0
                hits_mrr[top_k] += 1 / i
    scores = []
    for top_k in TOP_K_LIST:
        scores.append(
            {
                "k": top_k,
                "recall@k": hits_recall[top_k] / amount_valid_tests,
                "mrr@k": hits_mrr[top_k] / amount_valid_tests,
            }
        )
    return scores


def main():
    settings_base = Settings()
    settings_updated = settings_base.model_copy(update={"collection": "eval"})
    vector_store = DocumentStore(settings_updated)
    rag_engine = RagEngine(settings_updated, vector_store)

    corpus_paths = [str(path) for path in Path("./eval/corpus/").iterdir()]

    if vector_store.get_stored_chunk_amount() == 0:
        for path in corpus_paths:
            vector_store.add_in_batches(load_chunks(settings_updated, str(path)))

    with open("./eval/eval_set_resolved.json") as file:
        data_set = json.load(file)

    def retrieve_semantic(question: str, top_k: int):
        return rag_engine.retrieve(question, corpus_paths, top_k)

    scores = evaluate_retrieval(data_set, retrieve_semantic)
    for row in scores:
        print(f"Recall@{row['k']}: {row['recall@k']:.3f}")
        print(f"MRR@{row['k']}: {row['mrr@k']:.3f}")


if __name__ == "__main__":
    main()
