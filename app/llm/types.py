from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    finish_reason: str | None = None
    usage: GenerationUsage = GenerationUsage()
    provider_request_id: str | None = None
    latency_ms: int | None = None
