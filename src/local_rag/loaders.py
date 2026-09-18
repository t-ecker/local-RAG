from pathlib import Path

import pypdf
from langchain_core.documents import Document

from local_rag.errors import DocumentLoadError


def pdf_loader(path: str) -> list[Document]:
    reader = pypdf.PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append(
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": path, "page": i},
            )
        )
    return pages


def txt_and_md_loader(path: str) -> list[Document]:
    content = Path(path).read_text(encoding="utf-8")
    return [Document(content, metadata={"source": path})]


def file_loading_router(path: str) -> list[Document]:
    p = Path(path)
    if not p.is_file():
        raise DocumentLoadError(f"file not found: {path}")
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return pdf_loader(path)
    if suffix == ".txt" or suffix == ".md":
        return txt_and_md_loader(path)
    raise DocumentLoadError("filetype not supported (use .pdf .txt or .md)")
