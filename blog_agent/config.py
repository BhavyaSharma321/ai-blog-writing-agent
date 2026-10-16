"""Runtime settings.

Settings travel through LangGraph's `config["configurable"]["settings"]` instead of
os.environ, so on a shared Streamlit deployment one user's keys never leak into
another user's session.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

load_dotenv()

# Both providers expose an OpenAI-compatible API, so one client handles them.
# `models` is the fallback list shown when the live model list can't be fetched.
PROVIDERS = {
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
        ],
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"],
    },
}
IMAGE_MODELS = ["gemini-2.5-flash-image"]
DEFAULT_IMAGE_MODEL = IMAGE_MODELS[0]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    return value.strip() or default


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "auto"  # auto | groq | gemini
    llm_model: str = ""  # empty -> provider default
    groq_api_key: str = ""
    tavily_api_key: str = ""
    google_api_key: str = ""
    image_model: str = DEFAULT_IMAGE_MODEL
    output_dir: str = str(DEFAULT_OUTPUT_DIR)

    @classmethod
    def from_env(cls, **overrides: str) -> "Settings":
        base = cls(
            llm_provider=_env("LLM_PROVIDER", "auto").lower(),
            llm_model=_env("LLM_MODEL"),
            groq_api_key=_env("GROQ_API_KEY"),
            tavily_api_key=_env("TAVILY_API_KEY"),
            google_api_key=_env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY"),
            image_model=_env("GEMINI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL),
            output_dir=_env("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)),
        )
        # Only apply overrides that are actually filled in.
        return replace(base, **{k: v.strip() for k, v in overrides.items() if v and v.strip()})

    def key_for(self, provider: str) -> str:
        return {"groq": self.groq_api_key, "gemini": self.google_api_key}[provider]

    @property
    def provider(self) -> str:
        """Resolved text-LLM provider; 'auto' picks the first one with a key."""
        if self.llm_provider in PROVIDERS:
            return self.llm_provider
        return next((p for p in PROVIDERS if self.key_for(p)), "groq")

    @property
    def llm_api_key(self) -> str:
        return self.key_for(self.provider)

    @property
    def llm_base_url(self) -> str:
        return PROVIDERS[self.provider]["base_url"]

    @property
    def model(self) -> str:
        return self.llm_model or PROVIDERS[self.provider]["models"][0]

    @property
    def can_research(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def can_make_images(self) -> bool:
        return bool(self.google_api_key)


def get_settings(config: RunnableConfig | None) -> Settings:
    configurable = (config or {}).get("configurable", {})
    settings = configurable.get("settings")
    return settings if isinstance(settings, Settings) else Settings.from_env()
