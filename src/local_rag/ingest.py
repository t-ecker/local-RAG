import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_rag.config import Settings


def load_pdf_pages(filePath: str) -> list[Document]:
    reader = pypdf.PdfReader(filePath)
    documents = []
    for i, page in enumerate(reader.pages, start=1):
        documents.append(
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": filePath, "page": i},
            )
        )
    return documents


def chunk_text(docs: list[Document], settings: Settings):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    chunks = [d for d in chunks if d.page_content.strip()]
    return chunks


def ingest_documents(settings: Settings) -> list[Document]:
    filePath = "./sample_docs/GlobalJusticeReport.pdf"
    docs = load_pdf_pages(filePath)
    chunks = chunk_text(docs, settings)
    print(f"Seiten: {len(docs)}, Chunks: {len(chunks)}")
    return chunks
