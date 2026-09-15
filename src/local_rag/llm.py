from langchain_ollama import ChatOllama

from local_rag.config import Settings
from local_rag.prompts import build_messages
from local_rag.store import DocumentStore


class LlmEngine:
    def __init__(self, settings: Settings, documentStore: DocumentStore):
        self.llm = ChatOllama(
            model=settings.ollama_model,
            temperature=settings.temperature,
            base_url=settings.ollama_base_url,
        )
        self.store = documentStore

    def ask_llm(self, question: str):
        results = self.store.retrieve_by_vector(question)
        messages = build_messages(question, results)
        for chunk in self.llm.stream(messages):
            yield chunk.content
