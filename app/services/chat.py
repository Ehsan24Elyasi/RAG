from __future__ import annotations

import re
from typing import Protocol

from app.config import Settings
from app.rag.prompting import build_messages
from app.services.metadata import MetadataStore

_CITATION = re.compile(r"\[S(\d+)]")
_GREETING = re.compile(r"^(?:سلام|سلام علیکم|درود|hello|hi)[!،.\s]*$", re.IGNORECASE)
_THANKS = re.compile(r"^(?:ممنون|متشکرم|مرسی|سپاس|تشکر|thanks?|thank you)[!،.\s]*$", re.IGNORECASE)
_FAREWELL = re.compile(r"^(?:خداحافظ|فعلاً|فعلا|بدرود|bye|goodbye)[!،.\s]*$", re.IGNORECASE)
_REFERENTIAL = re.compile(
    r"(?:این|آن|اونو|اون|همون|موردش|شرایطش|جزئیاتش|نسخه(?:ش)?|حجمش|"
    r"امنیتش|نصبش|موبایل|یادداشت(?:‌| )?ها|بیشتر|ادامه|چطور|چند مگ|چی شد)",
    re.IGNORECASE,
)


class ChatClient(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorClient(Protocol):
    def query(self, query_embedding: list[float], top_k: int) -> dict: ...


def _history_aware_query(message: str, history: list[dict[str, str]]) -> str:
    """Expand only short, clearly referential follow-ups without an LLM call."""
    text = message.strip()
    word_count = len(text.split())
    if len(text) > 80 or (not _REFERENTIAL.search(text) and word_count > 3):
        return text
    prior_user_messages = [item.get("content", "").strip() for item in history if item.get("role") == "user"]
    if not prior_user_messages:
        return text
    prior = prior_user_messages[-1][:500]
    if not prior:
        return text
    prior_assistant_messages = [
        item.get("content", "").strip() for item in history if item.get("role") == "assistant"
    ]
    assistant_context = prior_assistant_messages[-1][:500] if prior_assistant_messages else ""
    parts = [f"موضوع قبلی کاربر: {prior}"]
    if assistant_context:
        parts.append(f"پاسخ قبلی دستیار: {assistant_context}")
    parts.append(f"پرسش فعلی: {text}")
    return "\n".join(parts)


class ChatService:
    def __init__(
        self,
        *,
        metadata: MetadataStore,
        embeddings: EmbeddingClient,
        vectors: VectorClient,
        chat: ChatClient,
        settings: Settings,
    ):
        self.metadata = metadata
        self.embeddings = embeddings
        self.vectors = vectors
        self.chat = chat
        self.settings = settings

    def _unknown_response(self, message: str, history: list[dict[str, str]]) -> str:
        base = (
            f"در منابع فعلی {self.settings.company_name} اطلاعات کافی برای پاسخ دقیق به این سؤال پیدا نکردم."
        )
        if len(message.split()) <= 3 and not history:
            return f"{base} لطفاً موضوع یا بخش موردنظرتان را کمی دقیق‌تر بفرمایید."
        return base

    def _deterministic_response(self, message: str) -> str | None:
        text = message.strip()
        if _GREETING.fullmatch(text):
            return f"سلام! من {self.settings.assistant_name}، دستیار پشتیبانی {self.settings.company_name} هستم. چطور می‌توانم کمک کنم؟"
        if _THANKS.fullmatch(text):
            return "خواهش می‌کنم. اگر پرسش دیگری دارید، در خدمتم."
        if _FAREWELL.fullmatch(text):
            return "خداحافظ! امیدوارم روز خوبی داشته باشید."
        return None

    def answer(self, message: str, history: list[dict[str, str]], top_k: int) -> tuple[str, list[dict]]:
        if response := self._deterministic_response(message):
            return response, []
        retrieval_query = _history_aware_query(message, history)
        query_vector = self.embeddings.embed([retrieval_query])
        if len(query_vector) != 1:
            raise RuntimeError("Embedding provider returned an unexpected result.")
        result = self.vectors.query(query_vector[0], min(top_k * 3, 50))
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        active_ids = self.metadata.active_vector_ids([str(item) for item in ids])

        contexts: list[str] = []
        sources: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for index, vector_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else None
            if distance is not None and float(distance) > self.settings.retrieval_max_distance:
                continue
            if str(vector_id) not in active_ids or index >= len(documents):
                continue
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            document_id = str(metadata.get("document_id", ""))
            chunk_index = int(metadata.get("chunk_index", 0))
            if not document_id or (document_id, chunk_index) in seen:
                continue
            seen.add((document_id, chunk_index))
            contexts.append(str(documents[index]))
            sources.append(
                {
                    "id": f"S{len(sources) + 1}",
                    "document_id": document_id,
                    "title": str(metadata.get("source_name", "Support document")),
                    "source_type": str(metadata.get("source_type", "upload")),
                    "url": str(metadata.get("source_url", "")) or None,
                    "chunk_index": chunk_index,
                }
            )
            if len(contexts) >= top_k:
                break
        if not contexts:
            return self._unknown_response(message, history), []

        normalized_history = [(item["role"], item["content"]) for item in history if item.get("content")]
        answer = self.chat.generate(
            build_messages(
                message,
                contexts,
                normalized_history,
                assistant_name=self.settings.assistant_name,
                company_name=self.settings.company_name,
            )
        ).strip()
        if not answer:
            return self._unknown_response(message, history), []
        cited_indexes = {int(match) - 1 for match in _CITATION.findall(answer)}
        cited_sources = [source for index, source in enumerate(sources) if index in cited_indexes]
        return answer, cited_sources
