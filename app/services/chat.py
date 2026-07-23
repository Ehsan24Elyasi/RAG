from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from app.config import Settings
from app.llm.types import GenerationResult
from app.rag.prompting import build_messages, format_support_contacts
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
    def generate(self, messages: list[dict[str, str]]) -> GenerationResult: ...


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorClient(Protocol):
    def query(self, query_embedding: list[float], top_k: int) -> dict: ...


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[dict]
    generation: GenerationResult | None = None


class ChatGenerationError(RuntimeError):
    """A model generation attempt failed after retrieval completed."""

    def __init__(self, latency_ms: int):
        super().__init__("The chat provider failed to generate a response.")
        self.latency_ms = latency_ms


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

    def _support_contacts(self) -> str:
        return format_support_contacts(
            self.settings.support_email,
            self.settings.support_phone,
            self.settings.support_url,
        )

    def _unknown_response(self, message: str, history: list[dict[str, str]]) -> str:
        base = "متأسفم، در حال حاضر نمی‌توانم پاسخ دقیقی برای این مورد ارائه کنم."
        contacts = self._support_contacts()
        if contacts:
            return f"{base}\n\nبرای بررسی دقیق‌تر، لطفاً از یکی از راه‌های رسمی زیر با پشتیبانی ما در تماس باشید:\n{contacts}"
        if len(message.split()) <= 3 and not history:
            return f"{base} لطفاً موضوع یا بخش موردنظرتان را کمی دقیق‌تر بفرمایید."
        return f"{base} اگر جزئیات بیشتری بفرمایید، موضوع را دقیق‌تر بررسی می‌کنم."

    def _deterministic_response(self, message: str, history: list[dict[str, str]]) -> str | None:
        text = message.strip()
        if _GREETING.fullmatch(text):
            if any(item.get("role") == "assistant" for item in history):
                return "سلام! بفرمایید؛ دربارهٔ کدام بخش از خدمات می‌توانم راهنمایی‌تان کنم؟"
            return f"سلام! من {self.settings.assistant_name}، دستیار پشتیبانی {self.settings.company_name} هستم. چطور می‌توانم کمک کنم؟"
        if _THANKS.fullmatch(text):
            return "خواهش می‌کنم. اگر پرسش دیگری دارید، در خدمتم."
        if _FAREWELL.fullmatch(text):
            return "خداحافظ! امیدوارم روز خوبی داشته باشید."
        return None

    def answer(self, message: str, history: list[dict[str, str]], top_k: int) -> tuple[str, list[dict]]:
        result = self.answer_detailed(message, history, top_k)
        return result.answer, result.sources

    def answer_detailed(self, message: str, history: list[dict[str, str]], top_k: int) -> ChatResult:
        if response := self._deterministic_response(message, history):
            return ChatResult(answer=response, sources=[])
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
            return ChatResult(answer=self._unknown_response(message, history), sources=[])

        normalized_history = [(item["role"], item["content"]) for item in history if item.get("content")]
        started = perf_counter()
        try:
            generation = self.chat.generate(
                build_messages(
                    message,
                    contexts,
                    normalized_history,
                    assistant_name=self.settings.assistant_name,
                    company_name=self.settings.company_name,
                    support_contacts=self._support_contacts(),
                )
            )
        except Exception as error:
            latency_ms = round((perf_counter() - started) * 1000)
            raise ChatGenerationError(latency_ms) from error
        answer = generation.text.strip()
        if not answer:
            return ChatResult(
                answer=self._unknown_response(message, history),
                sources=[],
                generation=generation,
            )
        cited_indexes = {int(match) - 1 for match in _CITATION.findall(answer)}
        cited_sources = [source for index, source in enumerate(sources) if index in cited_indexes]
        return ChatResult(answer=answer, sources=cited_sources, generation=generation)
