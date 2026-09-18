from langchain_core.documents import Document

SYSTEM_PROMPT = """You are an assistant that answers questions based only on \
the provided context.

Rules:
- Use only information from the context below. Do not rely on prior knowledge.
- If the answer is not in the context, say clearly: "That is not covered in \
the provided documents." Do not make anything up.
- Answer concisely and precisely.

Context:
{context}"""


def build_messages(
    question: str, retrieved_chunks: list[tuple[Document, float]]
) -> list[tuple[str, str]]:
    context = ""
    for chunk, score in retrieved_chunks:
        page = chunk.metadata.get("page")
        page_label = f"page {page}, " if page is not None else ""
        context += (
            f"\n\n[Source: {chunk.metadata.get('source')}, "
            f"{page_label}score={score:.3f}]\n"
            f"{chunk.page_content}"
        )
    print(f"\ncontext: {context}\n\n\n")

    return [
        ("system", SYSTEM_PROMPT.format(context=context)),
        ("human", question),
    ]
