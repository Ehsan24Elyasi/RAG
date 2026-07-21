from __future__ import annotations

from typing import Protocol

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import Settings


class ChatProvider(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class EmbeddingProvider(Protocol):
    fingerprint: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleChatProvider:
    """Synchronous native-role chat client for GapGPT-compatible APIs."""

    def __init__(self, api_key: str | None, model: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key or "local", base_url=base_url)
        self.model = model

    def generate(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0)
        return response.choices[0].message.content or ""


class SentenceTransformerEmbeddingProvider:
    """Local multilingual, L2-normalized embeddings with an explicit version."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.fingerprint = f"sentence-transformers:{model_name}:normalized-v1"
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()


def create_chat_provider(settings: Settings) -> ChatProvider:
    api_key = settings.chat_api_key.get_secret_value() if settings.chat_api_key else None
    return OpenAICompatibleChatProvider(
        api_key=api_key, model=settings.chat_model, base_url=settings.provider_base_url("chat")
    )


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    return SentenceTransformerEmbeddingProvider(settings.embedding_model)


LLMProvider = OpenAICompatibleChatProvider
