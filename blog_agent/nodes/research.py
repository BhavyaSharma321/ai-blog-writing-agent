"""Research node: Tavily web search -> LLM-curated Evidence Pack."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from tavily import TavilyClient

from blog_agent.config import Settings, get_settings
from blog_agent.llm import structured
from blog_agent.prompts import RESEARCH_SYSTEM
from blog_agent.schemas import EvidencePack, State
from blog_agent.utils import format_evidence

log = logging.getLogger(__name__)

RESULTS_PER_QUERY = 5
MAX_EVIDENCE = 15


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
    settings = get_settings(config)
    raw = search_all(settings, state.get("queries", []), state["mode"], state.get("recency_days", 30))
    if not raw:
        return {"evidence": []}

    try:
        pack: EvidencePack = structured(settings, EvidencePack).invoke(
            [
                SystemMessage(RESEARCH_SYSTEM.format(max_items=MAX_EVIDENCE)),
                HumanMessage(
                    f"Topic: {state['topic']}\nToday's date: {state['as_of']}\n\n"
                    f"Raw search results:\n{format_evidence(raw, snippet_chars=700)}"
                ),
            ]
        )
        # Guard against hallucinated URLs: keep only URLs that Tavily actually returned.
        known = {r["url"].rstrip("/") for r in raw}
        evidence = [e.model_dump() for e in pack.evidence if e.url.rstrip("/") in known]
    except Exception as exc:
        log.warning("Evidence curation failed, using raw results: %s", exc)
        evidence = []

    return {"evidence": (evidence or raw)[:MAX_EVIDENCE]}
