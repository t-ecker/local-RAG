import os

import pypdf
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


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


filePath = "./sample_docs/GlobalJusticeReport.pdf"
docs = load_pdf_pages(filePath)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
all_splits = [d for d in all_splits if d.page_content.strip()]

print(f"Seiten: {len(docs)}, Chunks: {len(all_splits)}")

embeddingsModel = os.getenv("OLLAMA_EMBEDDINGS_MODEL")
if embeddingsModel is None:
    raise RuntimeError("embeddings model is missing")
embeddings = OllamaEmbeddings(
    model=embeddingsModel, base_url=os.getenv("OLLAMA_BASE_URL")
)

vector_store = Chroma(
    collection_name="firstTry",
    embedding_function=embeddings,
    persist_directory="./chroma_db/",
)


def add_in_batches(vector_store, splits, batch_size=100):
    for i in range(0, len(splits), batch_size):
        batch = splits[i : i + batch_size]
        vector_store.add_documents(documents=batch)
        print(f"eingebettet: {i + len(batch)}/{len(splits)}")


add_in_batches(vector_store, all_splits)

chatModel = os.getenv("OLLAMA_MODEL")
if chatModel is None:
    raise RuntimeError("chat model is missing")
llm = ChatOllama(
    model=chatModel,
    temperature=0.1,
    base_url=os.getenv("OLLAMA_BASE_URL"),
)

while True:
    prompt = input("what do you wanna know? ")
    if prompt == "q":
        break

    results = vector_store.similarity_search_with_score(prompt, k=4)
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

    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
    print("")
