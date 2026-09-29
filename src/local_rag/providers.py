from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ollama import ResponseError
from sentence_transformers import CrossEncoder

from local_rag.config import Settings
from local_rag.errors import ModelNotFoundError, ProviderUnavailableError


@contextmanager
def provider_errors(model_name: str) -> Iterator[None]:
    try:
        yield
    except (httpx.ConnectError, ConnectionError) as e:
        raise ProviderUnavailableError("Ollama not available") from e
    except ResponseError as e:
        if e.status_code == 404:
            raise ModelNotFoundError(
                f"Model '{model_name}' not available. Did you pull it?"
            ) from e
        raise


def get_llm(settings: Settings) -> BaseChatModel:
    return ChatOllama(
        model=settings.ollama_model,
        temperature=settings.temperature,
        base_url=settings.ollama_base_url,
        reasoning=settings.reasoning,
    )


def get_embeddings(settings: Settings) -> Embeddings:
    return OllamaEmbeddings(
        model=settings.ollama_embeddings_model, base_url=settings.ollama_base_url
    )


@cache
def get_reranker(model_name: str) -> CrossEncoder:
    print(
        f"Loading reranker model '{model_name}' "
        "(first run only. the output below is normal)",
        flush=True,
    )
    return CrossEncoder(model_name)
