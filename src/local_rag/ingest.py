from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_rag.config import Settings
from local_rag.errors import NoExtractableTextError
from local_rag.loaders import load_file


def split_into_chunks(settings: Settings, pages: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(pages)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    return chunks


def load_chunks(settings: Settings, path: str) -> list[Document]:
    pages = load_file(path)
    chunks = split_into_chunks(settings, pages)
    if not chunks:
        raise NoExtractableTextError("no extractable text")
    return chunks
