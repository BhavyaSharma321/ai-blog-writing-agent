"""Image backends.

- mermaid:   diagram code rendered to PNG by mermaid.ink (free, no key)
- openverse: openly-licensed real photos from Openverse (free, no key), with attribution
- gemini:    AI-generated images (needs a Google key with image quota / billing)
"""

from __future__ import annotations

import base64
import logging
import re
import threading
from dataclasses import dataclass

import httpx

from blog_agent.config import Settings

log = logging.getLogger(__name__)

USER_AGENT = "ai-blog-writing-agent/1.0 (+https://github.com/BhavyaSharma321/ai-blog-writing-agent)"
MAX_BYTES = 8 * 1024 * 1024
LICENSE_NAMES = {"cc0": "CC0", "pdm": "Public Domain"}


@dataclass
class ImageResult:
    data: bytes
    mime: str
    backend: str
    credit: str = ""  # Markdown attribution, required for licensed photos


def _get(url: str, retries: int = 0, **params) -> httpx.Response:
    for attempt in range(retries + 1):
        try:
            response = httpx.get(
                url, params=params or None, headers={"User-Agent": USER_AGENT}, timeout=40, follow_redirects=True
            )
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == retries:
                raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500 or attempt == retries:
                raise
    raise AssertionError("unreachable")


def _as_image(response: httpx.Response) -> tuple[bytes, str]:
    mime = response.headers.get("content-type", "").split(";")[0].strip()
    if not mime.startswith("image/"):
        raise RuntimeError(f"Expected an image, got {mime or 'unknown content'}")
    if len(response.content) > MAX_BYTES:
        raise RuntimeError("Image is too large")
    return response.content, mime


# ---------------------------------------------------------------- mermaid
_UNQUOTED_LABEL = re.compile(r'\b([A-Za-z][\w-]*)\[(?!")([^\]\n]*)\]')


def _quote_labels(code: str) -> str:
    """A["..."] tolerates (), :, & etc. inside labels; A[...] often doesn't."""
    return _UNQUOTED_LABEL.sub(lambda m: f'{m.group(1)}["{m.group(2).replace(chr(34), chr(39))}"]', code)


def _mermaid_png(code: str) -> ImageResult:
    encoded = base64.urlsafe_b64encode(code.encode("utf-8")).decode("ascii")
    data, mime = _as_image(_get(f"https://mermaid.ink/img/{encoded}", retries=1, type="png", bgColor="!white", width=1200))
    return ImageResult(data, mime, "mermaid")


def render_mermaid(code: str) -> ImageResult:
    code = code.strip().removeprefix("```mermaid").removeprefix("```").removesuffix("```").strip()
    if not code:
        raise RuntimeError("No Mermaid code for this image")
    try:
        return _mermaid_png(code)
    except httpx.HTTPStatusError as exc:
        fixed = _quote_labels(code)
        if exc.response.status_code != 400 or fixed == code:
            raise RuntimeError("Mermaid could not render the diagram code") from exc
        return _mermaid_png(fixed)


# ---------------------------------------------------------------- openverse
_lock = threading.Lock()


def _clean(text: str) -> str:
    return (text or "").replace("[", "(").replace("]", ")").strip()


def _credit(item: dict) -> str:
    license_code = (item.get("license") or "").lower()
    license_name = LICENSE_NAMES.get(license_code, f"CC {license_code.upper()} {item.get('license_version') or ''}".strip())
    title = _clean(item.get("title")) or "Photo"
    creator = _clean(item.get("creator")) or "unknown"
    parts = [f"[{title}]({item['foreign_landing_url']})" if item.get("foreign_landing_url") else title]
    parts.append(f"by [{creator}]({item['creator_url']})" if item.get("creator_url") else f"by {creator}")
    parts.append(f"[{license_name}]({item['license_url']})" if item.get("license_url") else license_name)
    return "Photo: " + " ".join(parts[:2]) + f", {parts[2]}, via Openverse"


def _openverse_results(query: str) -> list[dict]:
    return _get(
        "https://api.openverse.org/v1/images/",
        retries=1,  # the free API is occasionally slow on uncached queries
        q=query,
        page_size=20,
        license_type="commercial",
        mature="false",
    ).json().get("results", [])


def _mentions(item: dict, word: str) -> bool:
    # Only human tags: machine tags (with an `accuracy`, e.g. from Clarifai) are often wrong.
    tags = [t.get("name", "") for t in item.get("tags") or [] if t.get("accuracy") is None]
    text = " ".join([item.get("title") or "", *tags]).lower()
    return word in text


def search_openverse(query: str, used_ids: set[str]) -> ImageResult:
    words = [w for w in query.lower().split() if len(w) > 2]
    if not words:
        raise RuntimeError("No search query for this photo")

    # Openverse's own ranking is loose. The first word is usually the subject
    # ("cricket" in "cricket stadium crowd"): keep only photos that mention it,
    # and if none can be used, search again with a shorter query. If the first word
    # isn't the subject ("historic cricket match"), also try starting from the next word.
    attempts = [(words[s], words[s : s + n]) for s in range(min(2, len(words))) for n in range(len(words) - s, 0, -1)]
    for subject, query_words in attempts:
        results = [r for r in _openverse_results(" ".join(query_words)) if _mentions(r, subject) and r.get("url")]
        results.sort(key=lambda r: sum(_mentions(r, w) for w in words), reverse=True)
        for item in results:
            with _lock:  # don't reuse the same photo twice in one blog
                if item["id"] in used_ids:
                    continue
                used_ids.add(item["id"])
            try:
                data, mime = _as_image(_get(item["url"]))
            except Exception as exc:  # some hosts block hotlinking
                log.info("Skipping Openverse photo %s: %s", item.get("url"), exc)
                continue
            return ImageResult(data, mime, "openverse", credit=_credit(item))
    raise RuntimeError(f"No usable Openverse photo for {query!r}")


# ---------------------------------------------------------------- gemini
def gemini_image(settings: Settings, prompt: str) -> ImageResult:
    from google import genai
    from google.genai import types

    if not settings.google_api_key:
        raise RuntimeError("No Gemini API key")
    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.generate_content(
        model=settings.image_model,
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )
    for candidate in response.candidates or []:
        for part in (candidate.content.parts if candidate.content else None) or []:
            if part.inline_data and part.inline_data.data:
                return ImageResult(part.inline_data.data, part.inline_data.mime_type or "image/png", "gemini")
    raise RuntimeError("Gemini returned no image data")


# ---------------------------------------------------------------- dispatcher
def make_image(settings: Settings, spec: dict, used_photo_ids: set[str]) -> ImageResult:
    """Try the backends for this spec in order; free backends act as fallbacks."""
    # A diagram never falls back to a stock photo: an unrelated photo is worse than none.
    free = ["mermaid"] if spec.get("kind") == "diagram" else ["openverse"]
    order = {
        "gemini": ["gemini", *free],
        "auto": free,
        "mermaid": ["mermaid"],
        "openverse": ["openverse"],
    }.get(settings.image_source, free)

    attempts = {
        "gemini": lambda: gemini_image(settings, spec["prompt"]),
        "mermaid": lambda: render_mermaid(spec.get("mermaid", "")),
        "openverse": lambda: search_openverse(spec.get("search_query", ""), used_photo_ids),
    }
    errors = []
    for backend in order:
        try:
            return attempts[backend]()
        except Exception as exc:
            errors.append(f"{backend}: {str(exc)[:200]}")
    raise RuntimeError(" | ".join(errors))
