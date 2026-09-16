from langchain_core.documents import Document

SYSTEM_PROMPT = """You are an assistant that answers questions based only on \
the provided context.

Rules:
- Use only information from the context below. Do not rely on prior knowledge.
- If the answer is not in the context, say clearly: "That is not covered in \
the provided documents." Do not make anything up.
- Cite the source your answer relies on (file name and page).
- Answer concisely and precisely.

Context:
{context}"""


def build_messages(
    question: str, results: list[tuple[Document, float]]
) -> list[tuple[str, str]]:
    context = ""
    for doc, score in results:
        context += (
            f"\n\n[Source: {doc.metadata.get('source')}, "
            f"page {doc.metadata.get('page')}, score={score:.3f}]\n"
            f"{doc.page_content}"
        )
    print(f"\ncontext: {context}\n\n\n")

    return [
        ("system", SYSTEM_PROMPT.format(context=context)),
        ("human", question),
    ]
