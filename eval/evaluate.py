import json
import sys
from functools import partial
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from local_rag.config import Settings
from local_rag.errors import LocalRagError
from local_rag.ingest import load_chunks
from local_rag.loaders import LOADERS
from local_rag.retrievers import RETRIEVERS
from local_rag.store import DocumentStore

TOP_K_LIST = [1, 3, 5]

EVAL_DIR = Path("./eval")
CORPUS_DIR = EVAL_DIR / "corpus"
DATASET_PATH = EVAL_DIR / "eval_set_resolved.json"
PERSIST_DIR = EVAL_DIR / "chroma_db"


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


def index_corpus(
    store: DocumentStore, settings: Settings, corpus_paths: list[str]
) -> None:
    if store.count_chunks() > 0:
        store.delete_all()
    for path in corpus_paths:
        store.add_chunks(load_chunks(settings, path))


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as file:
        return json.load(file)


def evaluate_all_modes(
    store: DocumentStore,
    settings: Settings,
    data_set: list[dict],
    corpus_paths: list[str],
) -> dict[str, list[dict]]:
    results = {}
    for mode, retriever in RETRIEVERS.items():
        retrieve_fn = partial(
            retriever,
            store,
            settings,
            selected_sources=corpus_paths,
            k=max(TOP_K_LIST),
        )
        results[mode] = evaluate_retrieval(data_set, retrieve_fn)
    return results


def main():
    settings_base = Settings()
    settings_updated = settings_base.model_copy(update={"persist_dir": PERSIST_DIR})
    store = DocumentStore(settings_updated)
    corpus_paths = [
        str(path) for path in CORPUS_DIR.iterdir() if path.suffix.lower() in LOADERS
    ]

    index_corpus(store, settings_updated, corpus_paths)
    results = evaluate_all_modes(store, settings_updated, load_dataset(), corpus_paths)
    print_table(results)


if __name__ == "__main__":
    try:
        main()
    except LocalRagError as e:
        sys.exit(f"eval failed: {e}")
