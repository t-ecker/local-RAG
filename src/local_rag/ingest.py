from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_rag.config import Settings
from local_rag.errors import NoExtractableTextError
from local_rag.loaders import file_loading_router


def split_pages(settings: Settings, pages: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(pages)
    chunks = [d for d in chunks if d.page_content.strip()]
    return chunks


def load_chunks(settings: Settings, path: str) -> list[Document]:
    pages = file_loading_router(path)
    chunks = split_pages(settings, pages)
    if not chunks:
        raise NoExtractableTextError("no extractable text")
    print(f"Pages: {len(pages)}, Chunks: {len(chunks)}")
    return chunks
