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

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    return value.strip() or default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    openai_model: str = DEFAULT_OPENAI_MODEL
    tavily_api_key: str = ""
    google_api_key: str = ""
    image_model: str = DEFAULT_IMAGE_MODEL
    output_dir: str = str(DEFAULT_OUTPUT_DIR)

    @classmethod
    def from_env(cls, **overrides: str) -> "Settings":
        base = cls(
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_model=_env("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            tavily_api_key=_env("TAVILY_API_KEY"),
            google_api_key=_env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY"),
            image_model=_env("GEMINI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL),
            output_dir=_env("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)),
        )
        # Only apply overrides that are actually filled in.
        return replace(base, **{k: v.strip() for k, v in overrides.items() if v and v.strip()})

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
