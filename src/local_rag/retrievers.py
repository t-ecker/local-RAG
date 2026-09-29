import operator

import bm25s
import Stemmer  # ty: ignore[unresolved-import]
from langchain_core.documents import Document

from local_rag.config import Settings
from local_rag.providers import get_reranker
from local_rag.store import DocumentStore

RRF_K = 60


def retrieve_semantic(
    store: DocumentStore,
    settings: Settings,
    question: str,
    selected_sources: list[str],
    k: int,
) -> list[tuple[Document, float]]:
    return store.similarity_search(question, selected_sources, k)


def retrieve_lexical(
    store: DocumentStore,
    settings: Settings,
    question: str,
    selected_sources: list[str],
    k: int,
) -> list[tuple[Document, float]]:
    corpus = store.get_chunks(selected_sources)
    language = settings.corpus_language
    stemmer = Stemmer.Stemmer(language)
    corpus_tokens = bm25s.tokenize(
        [c.page_content for c in corpus], stopwords=language, stemmer=stemmer
    )
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    query_tokens = bm25s.tokenize(question, stopwords=language, stemmer=stemmer)
    chunks, scores = bm25.retrieve(
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
    selected_sources: list[str],
    k: int,
) -> list[tuple[Document, float]]:
    rankings: list[list[Document]] = []

    results_semantic = retrieve_semantic(
        store, settings, question, selected_sources, settings.hybrid_pool_size
    )
    results_lexical = retrieve_lexical(
        store, settings, question, selected_sources, settings.hybrid_pool_size
    )
    rankings.append([chunk for chunk, score in results_semantic])
    rankings.append([chunk for chunk, score in results_lexical])

    scores: dict[str, float] = {}
    chunks_by_id = {}

    for ranking in rankings:
        for i, chunk in enumerate(ranking):
            assert chunk.id is not None
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (RRF_K + i + 1)
            chunks_by_id[chunk.id] = chunk

    top_k = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)[:k]
    return [(chunks_by_id[chunk_id], score) for chunk_id, score in top_k]


def retrieve_hybrid_rerank(
    store: DocumentStore,
    settings: Settings,
    question: str,
    selected_sources: list[str],
    k: int,
) -> list[tuple[Document, float]]:
    candidates = retrieve_hybrid(
        store, settings, question, selected_sources, settings.rerank_pool_size
    )

    model = get_reranker(settings.reranker_model)
    ranks = model.rank(question, [c.page_content for c, _ in candidates], top_k=k)
    results = []
    for rank in ranks:
        chunk, _ = candidates[int(rank["corpus_id"])]
        results.append((chunk, float(rank["score"])))
    return results


RETRIEVERS = {
    "semantic": retrieve_semantic,
    "lexical": retrieve_lexical,
    "hybrid": retrieve_hybrid,
    "hybrid_rerank": retrieve_hybrid_rerank,
}
