import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_rag.config import Settings


def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append(
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": file_path, "page": i},
            )
        )
    return pages


def split_pages(settings: Settings, docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    chunks = [d for d in chunks if d.page_content.strip()]
    return chunks


def load_chunks(settings: Settings, pdf_path: str) -> list[Document]:
    pages = load_pdf_pages(pdf_path)
    chunks = split_pages(settings, pages)
    print(f"Pages: {len(pages)}, Chunks: {len(chunks)}")
    return chunks
