import operator

import bm25s
import Stemmer  # ty: ignore[unresolved-import]
from langchain_core.documents import Document

from local_rag.config import Settings
from local_rag.store import DocumentStore


def retrieve_semantic(
    store: DocumentStore,
    settings: Settings,
    question: str,
    selected_documents: list[str],
    k: int,
) -> list[tuple[Document, float]]:
    return store.retrieve_semantic(question, selected_documents, k)


def retrieve_lexical(
    store: DocumentStore,
    settings: Settings,
    question: str,
    selected_documents: list[str],
    k: int,
) -> list[tuple[Document, float]]:
    corpus = store.get_documents(selected_documents)
    language = settings.corpus_language
    stemmer = Stemmer.Stemmer(language)
    corpus_tokens = bm25s.tokenize(
        [c.page_content for c in corpus], stopwords=language, stemmer=stemmer
    )
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    query_tokens = bm25s.tokenize(question, stemmer=stemmer)
    chunks, scores = retriever.retrieve(
        query_tokens, k=min(k, len(corpus)), corpus=corpus, return_as="tuple"
    )
    results: list[tuple[Document, float]] = []
    for i in range(len(chunks[0])):
        if scores[0][i] > 0:
            results.append((chunks[0][i], float(scores[0][i])))
    return results


def retrieve_hybrid(
    store: DocumentStore,
    settings: Settings,
    question: str,
    selected_documents: list[str],
    k: int,
) -> list[tuple[Document, float]]:
    retrieved_chunks: list[list[Document]] = []

    results_semantic = retrieve_semantic(
        store, settings, question, selected_documents, k * 2
    )
    results_lexical = retrieve_lexical(
        store, settings, question, selected_documents, k * 2
    )
    retrieved_chunks.append([chunk for chunk, score in results_semantic])
    retrieved_chunks.append([chunk for chunk, score in results_lexical])

    scores: dict[str, float] = {}
    all_chunks = {}

    for chunk_list in retrieved_chunks:
        for i, chunk in enumerate(chunk_list):
            assert chunk.id is not None
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (60 + i + 1)
            all_chunks[chunk.id] = chunk

    top_k = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)[:k]
    return [(all_chunks[id], score) for id, score in top_k]


RETRIEVERS = {
    "semantic": retrieve_semantic,
    "lexical": retrieve_lexical,
    "hybrid": retrieve_hybrid,
}
