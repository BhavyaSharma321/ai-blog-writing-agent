"""LLM factories."""

from __future__ import annotations

from typing import TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from blog_agent.config import Settings

T = TypeVar("T", bound=BaseModel)


class MissingKeyError(RuntimeError):
    pass


def get_llm(settings: Settings, temperature: float = 0.4) -> ChatOpenAI:
    if not settings.llm_api_key:
        raise MissingKeyError(f"No API key set for LLM provider '{settings.provider}'.")
    return ChatOpenAI(
        model=settings.model,
        temperature=temperature,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        max_retries=5,  # free tiers (Groq/Gemini) rate-limit parallel workers
    )


def structured(settings: Settings, schema: type[T], temperature: float = 0.2):
    # function_calling is the most forgiving method across models and schema shapes.
    return get_llm(settings, temperature).with_structured_output(schema, method="function_calling")
