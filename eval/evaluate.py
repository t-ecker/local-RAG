import json
from pathlib import Path

from langchain_core.documents import Document

from local_rag.config import Settings
from local_rag.ingest import load_chunks
from local_rag.query import RagEngine
from local_rag.store import DocumentStore

TOP_K = 4


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


def get_recall_at_k(test, retrieved_chunks):
    for chunk, _ in retrieved_chunks:
        if is_chunk_correct(test, chunk):
            return 1.0
    print(f"failed at test: {test['question']}")
    print(
        f"  expected: {test['expected_source']} p={test['expected_page']} "
        f"chars={test['answer_char_start']}-{test['answer_char_end']}"
    )
    for chunk, score in retrieved_chunks:
        metadata = chunk.metadata
        start = metadata["start_index"]
        end = start + len(chunk.page_content)
        print(
            f"  got:      {Path(metadata['source']).name} "
            f"p={metadata.get('page')} chars={start}-{end} score={score:.3f}"
        )
    return 0.0


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

    recall_at_k_score: float = 0.0
    amount_valid_tests: int = 0

    for test in data_set:
        retrieved_chunks = rag_engine.retrieve(
            test["question"],
            corpus_paths,
            TOP_K,
        )
        if test["expected_source"] is None:
            continue
        amount_valid_tests += 1
        recall_at_k_score += get_recall_at_k(test, retrieved_chunks)

    recall_at_k_score /= amount_valid_tests
    # recall_at_k_score = 1 - recall_at_k_score

    print(
        f"Recall@{TOP_K}: {recall_at_k_score:.3f}  (on {amount_valid_tests} answerable questions)"
    )


if __name__ == "__main__":
    main()
