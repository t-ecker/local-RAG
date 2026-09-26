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


RETRIEVERS = {
    "semantic": retrieve_semantic,
    "lexical": retrieve_lexical,
}
