from langchain_core.documents import Document


def build_messages(
    prompt: str, results: list[tuple[Document, float]]
) -> list[tuple[str, str]]:
    context = ""
    for d, score in results:
        context += f"\n\n[Source: {d.metadata.get('source')}, page {d.metadata.get('page')}, score={score:3f}]\n{d.page_content}"
    print(f"\ncontext: {context}\n\n\n")

    system_prompt = f"""You are an assistant that answers questions based only on the provided context.
        Rules:
        - Use only information from the context below. Do not rely on prior knowledge.
        - If the answer is not in the context, say clearly: "That is not covered in the provided documents." Do not make anything up.
        - Cite the source your answer relies on (file name and page).
        - Answer concisely and precisely.

        Context:
        {context}"""

    messages = [
        ("system", system_prompt),
        ("human", prompt),
    ]
    return messages
