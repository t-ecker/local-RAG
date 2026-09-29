import json
from functools import partial
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from local_rag.config import Settings
from local_rag.ingest import load_chunks
from local_rag.retrievers import RETRIEVERS
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


def find_hit_rank(test, retrieved_chunks):
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
        retrieved_chunks = retrieve_fn(test["question"])
        for top_k in TOP_K_LIST:
            if (i := find_hit_rank(test, retrieved_chunks[:top_k])) is not None:
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


def print_table(results: dict[str, list[dict]]) -> None:
    rows = []
    for mode, scores in results.items():
        row = {"mode": mode}
        row.update({f"R@{s['k']}": s["recall@k"] for s in scores})
        row.update({f"MRR@{s['k']}": s["mrr@k"] for s in scores})
        rows.append(row)

    table = pd.DataFrame(rows).set_index("mode")
    print(table.to_string(float_format="%.3f"))


def main():
    settings_base = Settings()
    settings_updated = settings_base.model_copy(
        update={"persist_dir": Path("./eval/chroma_db")}
    )
    store = DocumentStore(settings_updated)

    corpus_paths = [str(path) for path in Path("./eval/corpus/").iterdir()]

    if store.count_chunks() == 0:
        for path in corpus_paths:
            store.add_chunks(load_chunks(settings_updated, str(path)))

    with open("./eval/eval_set_resolved.json") as file:
        data_set = json.load(file)

    results = {}
    for mode, retriever in RETRIEVERS.items():
        retrieve_fn = partial(
            retriever,
            store,
            settings_updated,
            selected_sources=corpus_paths,
            k=max(TOP_K_LIST),
        )
        results[mode] = evaluate_retrieval(data_set, retrieve_fn)

    print_table(results)


if __name__ == "__main__":
    main()
