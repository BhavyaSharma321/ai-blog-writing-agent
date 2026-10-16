"""Worker node: drafts exactly one section of the blog."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from blog_agent.config import get_settings
from blog_agent.llm import get_llm
from blog_agent.prompts import WORKER_SYSTEM
from blog_agent.utils import format_evidence


def worker_node(payload: dict, config: RunnableConfig) -> dict:
    settings = get_settings(config)
    plan, task, evidence = payload["plan"], payload["task"], payload.get("evidence", [])

    code_rule = (
        "Include at least one short, correct, runnable code snippet in a fenced block with a language tag."
        if task["requires_code"]
        else "Only include code if it is essential."
    )
    if task["requires_citations"] and evidence:
        citation_rule = (
            "Back factual claims with inline Markdown links to the evidence, e.g. ([Source Title](url)). "
            "Only use URLs from the evidence list below."
        )
    else:
        citation_rule = "Do not add links or citations."

    system = WORKER_SYSTEM.format(
        blog_title=plan["blog_title"],
        audience=plan["audience"],
        tone=plan["tone"],
        blog_kind=plan["blog_kind"],
        constraints="\n".join(f"- {c}" for c in plan["constraints"]) or "- (none)",
        section_title=task["title"],
        target_words=task["target_words"],
        code_rule=code_rule,
        citation_rule=citation_rule,
    )
    brief = (
        f"Topic: {payload['topic']}\n"
        f"Section {task['id']}: {task['title']}\n"
        f"Goal: {task['goal']}\n"
        "Must cover:\n" + "\n".join(f"- {b}" for b in task["bullets"])
    )
    if task["requires_citations"] and evidence:
        # Short snippets keep each worker's prompt small (free-tier token limits).
        brief += f"\n\nEvidence:\n{format_evidence(evidence, snippet_chars=200)}"

    response = get_llm(settings, temperature=0.5).invoke([SystemMessage(system), HumanMessage(brief)])
    md = "\n".join(line.rstrip() for line in response.content.strip().splitlines())
    if not md.startswith("## "):
        md = f"## {task['title']}\n\n{md}"
    return {"sections": [(task["id"], md)]}
