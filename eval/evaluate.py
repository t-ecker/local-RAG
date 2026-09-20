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
    for chunk, _ in retrieved_chunks:
        if is_chunk_correct(test, chunk):
            return 1.0
    # print(f"failed at test: {test['question']}")
    # print(
    #     f"  expected: {test['expected_source']} p={test['expected_page']} "
    #     f"chars={test['answer_char_start']}-{test['answer_char_end']}"
    # )
    # for chunk, score in retrieved_chunks:
    #     metadata = chunk.metadata
    #     start = metadata["start_index"]
    #     end = start + len(chunk.page_content)
    #     print(
    #         f"  got:      {Path(metadata['source']).name} "
    #         f"p={metadata.get('page')} chars={start}-{end} score={score:.3f}"
    #     )
    return 0.0


def get_recall_at_k_score(data_set, corpus_paths, rag_engine):
    hits: dict[int, float] = dict.fromkeys(TOP_K_LIST, 0.0)
    amount_valid_tests: int = 0
    for test in data_set:
        if test["expected_source"] is None:
            continue
        amount_valid_tests += 1
        retrieved_chunks = rag_engine.retrieve(
            test["question"],
            corpus_paths,
            max(TOP_K_LIST),
        )
        for top_k in TOP_K_LIST:
            hits[top_k] += check_chunks(test, retrieved_chunks[:top_k])
    scores = []
    for top_k in TOP_K_LIST:
        scores.append(
            {
                "k-value": top_k,
                "recall@k": hits[top_k] / amount_valid_tests,
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

    recall_at_top_k_scores = get_recall_at_k_score(data_set, corpus_paths, rag_engine)
    for row in recall_at_top_k_scores:
        print(f"Recall@{row['k-value']}: {row['recall@k']:.3f}")


if __name__ == "__main__":
    main()
