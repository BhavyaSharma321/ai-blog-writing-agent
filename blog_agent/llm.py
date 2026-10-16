"""LLM factories and model discovery."""

from __future__ import annotations

import logging
from typing import TypeVar

from langchain_openai import ChatOpenAI
from openai import OpenAI, RateLimitError
from pydantic import BaseModel

from blog_agent.config import IMAGE_MODELS, PROVIDERS, Settings

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

# Model ids containing these are not chat/text models.
_NON_TEXT = ("whisper", "tts", "guard", "embedding", "image", "imagen", "veo", "audio", "live", "aqa",
             "learnlm", "transcribe", "banana", "orpheus", "playai", "compound", "safeguard", "robotics", "computer-use")


class MissingKeyError(RuntimeError):
    pass


def _chat(settings: Settings, temperature: float) -> ChatOpenAI:
    if not settings.llm_api_key:
        raise MissingKeyError(f"No API key set for {PROVIDERS[settings.provider]['label']}.")
    return ChatOpenAI(
        model=settings.model,
        temperature=temperature,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        max_retries=2,
    )


def _with_rate_limit_retry(runnable):
    # Free tiers (e.g. Groq: 8k tokens/min) reset per minute, so back off for up to ~1 minute.
    return runnable.with_retry(
        retry_if_exception_type=(RateLimitError,),
        exponential_jitter_params={"initial": 4, "max": 40},
        stop_after_attempt=8,
    )


def get_llm(settings: Settings, temperature: float = 0.4):
    return _with_rate_limit_retry(_chat(settings, temperature))


def structured(settings: Settings, schema: type[T], temperature: float = 0.2):
    # function_calling is the most forgiving method across models and schema shapes.
    return _with_rate_limit_retry(
        _chat(settings, temperature).with_structured_output(schema, method="function_calling")
    )


def _fetch_model_ids(provider: str, api_key: str) -> list[str]:
    client = OpenAI(api_key=api_key, base_url=PROVIDERS[provider]["base_url"], timeout=15, max_retries=1)
    return sorted(m.id.removeprefix("models/") for m in client.models.list())


def list_text_models(provider: str, api_key: str) -> list[str]:
    """Recommended models first, then any other chat model the key can use."""
    preferred = PROVIDERS[provider]["models"]
    if not api_key:
        return preferred
    try:
        live = [m for m in _fetch_model_ids(provider, api_key) if not any(x in m.lower() for x in _NON_TEXT)]
        if provider == "gemini":
            live = [m for m in live if m.startswith("gemini")]
    except Exception as exc:
        log.warning("Could not list %s models: %s", provider, exc)
        return preferred
    if not live:
        return preferred
    return [m for m in preferred if m in live] + [m for m in live if m not in preferred]


def list_image_models(api_key: str) -> list[str]:
    if not api_key:
        return IMAGE_MODELS
    try:
        live = [m for m in _fetch_model_ids("gemini", api_key) if m.startswith("gemini") and ("image" in m or "banana" in m)]
    except Exception as exc:
        log.warning("Could not list Gemini image models: %s", exc)
        return IMAGE_MODELS
    return [m for m in IMAGE_MODELS if m in live] + [m for m in live if m not in IMAGE_MODELS] or IMAGE_MODELS
