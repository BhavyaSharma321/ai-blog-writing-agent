"""Research node: Tavily web search -> ranked, de-duplicated Evidence Pack."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.runnables import RunnableConfig
from tavily import TavilyClient

from blog_agent.config import Settings, get_settings
from blog_agent.schemas import State

log = logging.getLogger(__name__)

RESULTS_PER_QUERY = 5
MAX_EVIDENCE = 8
SNIPPET_CHARS = 400


def _search(client: TavilyClient, query: str, mode: str, recency_days: int) -> list[dict]:
    kwargs: dict = {"query": query, "max_results": RESULTS_PER_QUERY, "search_depth": "basic"}
    if mode == "open_book":
        kwargs.update(topic="news", days=recency_days)
    try:
        results = client.search(**kwargs).get("results", [])
    except Exception as exc:  # one failing query should not kill the run
        log.warning("Tavily search failed for %r: %s", query, exc)
        return []
    return [
        {
            "title": r.get("title") or r.get("url", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or "").strip(),
            "published_at": r.get("published_date"),
            "score": r.get("score"),
        }
        for r in results
        if r.get("url")
    ]


def search_all(settings: Settings, queries: list[str], mode: str, recency_days: int) -> list[dict]:
    client = TavilyClient(api_key=settings.tavily_api_key)
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(queries)))) as pool:
        batches = pool.map(lambda q: _search(client, q, mode, recency_days), queries)
    seen, raw = set(), []
    for batch in batches:
        for item in batch:
            key = item["url"].rstrip("/")
            if key not in seen:
                seen.add(key)
                raw.append(item)
    return raw


def research_node(state: State, config: RunnableConfig) -> dict:
    # Deterministic, no LLM call: Tavily already ranks results by relevance, and skipping an
    # LLM "curation" pass saves ~6k tokens (most of a free Groq minute) per blog.
    settings = get_settings(config)
    raw = search_all(settings, state.get("queries", []), state["mode"], state.get("recency_days", 30))
    raw.sort(key=lambda r: r.get("score") or 0, reverse=True)
    evidence = [{**r, "snippet": r["snippet"][:SNIPPET_CHARS]} for r in raw[:MAX_EVIDENCE]]
    return {"evidence": evidence}
