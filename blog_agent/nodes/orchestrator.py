"""Orchestrator (planner) node + fan-out to parallel workers."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from blog_agent.config import get_settings
from blog_agent.llm import structured
from blog_agent.prompts import ORCHESTRATOR_SYSTEM
from blog_agent.schemas import Plan, State
from blog_agent.utils import format_evidence


def orchestrator_node(state: State, config: RunnableConfig) -> dict:
    settings = get_settings(config)
    evidence = state.get("evidence", [])
    plan: Plan = structured(settings, Plan).invoke(
        [
            SystemMessage(
                ORCHESTRATOR_SYSTEM.format(
                    target_words=state.get("target_words", 1500),
                    mode=state.get("mode", "closed_book"),
                    as_of=state["as_of"],
                )
            ),
            HumanMessage(f"Topic: {state['topic']}\n\nEvidence:\n{format_evidence(evidence)}"),
        ]
    )

    # Normalise: stable ordering, no citations without evidence.
    tasks = sorted(plan.tasks, key=lambda t: t.id)
    for i, task in enumerate(tasks, 1):
        task.id = i
        task.target_words = max(80, task.target_words)
        if not evidence:
            task.requires_citations = False
    plan.tasks = tasks
    return {"plan": plan}


def fan_out(state: State) -> list[Send]:
    """One Send per section -> LangGraph runs all workers in parallel."""
    plan = state["plan"]
    plan_meta = plan.model_dump(exclude={"tasks"})
    return [
        Send(
            "worker",
            {
                "topic": state["topic"],
                "mode": state.get("mode", "closed_book"),
                "plan": plan_meta,
                "task": task.model_dump(),
                "evidence": state.get("evidence", []),
            },
        )
        for task in plan.tasks
    ]
