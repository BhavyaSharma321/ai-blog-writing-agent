"""Router node: decides Closed Book / Hybrid / Open Book and drafts search queries."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from blog_agent.config import get_settings
from blog_agent.llm import structured
from blog_agent.prompts import ROUTER_SYSTEM
from blog_agent.schemas import RouterDecision, State

MAX_QUERIES = 6


def router_node(state: State, config: RunnableConfig) -> dict:
    settings = get_settings(config)
    decision: RouterDecision = structured(settings, RouterDecision).invoke(
        [
            SystemMessage(ROUTER_SYSTEM),
            HumanMessage(f"Topic: {state['topic']}\nToday's date: {state['as_of']}"),
        ]
    )

    research_allowed = state.get("enable_research", True) and settings.can_research
    needs_research = decision.mode != "closed_book" and research_allowed
    reason = decision.reason
    if decision.mode != "closed_book" and not research_allowed:
        reason += " (Research skipped: disabled or no Tavily key.)"

    return {
        "mode": decision.mode,
        "needs_research": needs_research,
        "router_reason": reason,
        "queries": decision.queries[:MAX_QUERIES] if needs_research else [],
    }


def route_after_router(state: State) -> str:
    return "research" if state.get("needs_research") else "orchestrator"
