from pathlib import Path

import pypdf
from langchain_core.documents import Document

from local_rag.errors import DocumentLoadError


def load_pdf(path: str) -> list[Document]:
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


def load_text(path: str) -> list[Document]:
    content = Path(path).read_text(encoding="utf-8")
    return [Document(content, metadata={"source": path})]


def load_file(path: str) -> list[Document]:
    file_path = Path(path)
    if not file_path.is_file():
        raise DocumentLoadError(f"file not found: {path}")
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".txt" or suffix == ".md":
        return load_text(path)
    raise DocumentLoadError("filetype not supported (use .pdf .txt or .md)")
