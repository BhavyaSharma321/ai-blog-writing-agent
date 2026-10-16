"""Main LangGraph: router -> [research] -> orchestrator -> workers (parallel) -> reducer."""

from __future__ import annotations

from datetime import date

from langgraph.graph import END, START, StateGraph

from blog_agent.config import Settings
from blog_agent.nodes.orchestrator import fan_out, orchestrator_node
from blog_agent.nodes.reducer import reducer_node
from blog_agent.nodes.research import research_node
from blog_agent.nodes.router import route_after_router, router_node
from blog_agent.nodes.worker import worker_node
from blog_agent.schemas import State


def build_graph():
    g = StateGraph(State)
    g.add_node("router", router_node)
    g.add_node("research", research_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("worker", worker_node)
    g.add_node("reducer", reducer_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_router, ["research", "orchestrator"])
    g.add_edge("research", "orchestrator")
    g.add_conditional_edges("orchestrator", fan_out, ["worker"])
    g.add_edge("worker", "reducer")
    g.add_edge("reducer", END)
    return g.compile()


def initial_state(
    topic: str,
    *,
    target_words: int = 1500,
    recency_days: int = 30,
    enable_research: bool = True,
    enable_images: bool = True,
    max_images: int = 3,
) -> State:
    return {
        "topic": topic.strip(),
        "as_of": date.today().isoformat(),
        "target_words": target_words,
        "recency_days": recency_days,
        "enable_research": enable_research,
        "enable_images": enable_images,
        "max_images": max_images,
        "evidence": [],
        "sections": [],
    }


def run(topic: str, settings: Settings | None = None, **options) -> State:
    graph = build_graph()
    return graph.invoke(
        initial_state(topic, **options),
        {"configurable": {"settings": settings or Settings.from_env()}},
    )
