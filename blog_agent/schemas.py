"""Pydantic models (structured LLM outputs) and the LangGraph state."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

Mode = Literal["closed_book", "hybrid", "open_book"]


# ---------------------------------------------------------------- router
class RouterDecision(BaseModel):
    needs_research: bool = Field(description="True if web research is needed to write a good blog.")
    mode: Mode = Field(description="closed_book | hybrid | open_book")
    reason: str = Field(description="One sentence explaining the decision.")
    queries: list[str] = Field(description="3-6 targeted web search queries. Empty for closed_book.")


# ---------------------------------------------------------------- research
class EvidenceItem(BaseModel):
    title: str
    url: str
    snippet: str = Field(description="The most useful 1-3 sentences / facts from the source.")
    published_at: Optional[str] = Field(description="Publish date if known (YYYY-MM-DD), else null.")


class EvidencePack(BaseModel):
    evidence: list[EvidenceItem] = Field(description="Relevant, de-duplicated, high-quality sources only.")


# ---------------------------------------------------------------- planner
class Task(BaseModel):
    id: int = Field(description="Section order, starting at 1.")
    title: str = Field(description="Section heading (no leading #).")
    goal: str = Field(description="What the reader should understand / be able to do after this section.")
    bullets: list[str] = Field(description="3-6 concrete points this section must cover.")
    target_words: int = Field(description="Target word count for this section.")
    requires_citations: bool = Field(description="True if the section makes factual claims that must cite evidence.")
    requires_code: bool = Field(description="True if the section should include a code snippet.")


class Plan(BaseModel):
    blog_title: str
    audience: str = Field(description="Who this blog is for.")
    tone: str = Field(description="Writing tone, e.g. 'practical, friendly, precise'.")
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"]
    constraints: list[str] = Field(description="Global rules every writer must follow.")
    tasks: list[Task]


# ---------------------------------------------------------------- images
class ImageSpec(BaseModel):
    after_heading: str = Field(description="Exact text of the '## ' heading the image belongs under.")
    kind: Literal["diagram", "photo"] = Field(description="diagram = process/architecture/flow; photo = real-world subject.")
    alt: str = Field(description="Short alt text.")
    caption: str = Field(description="One-line caption shown under the image.")
    search_query: str = Field(description="2-4 word stock-photo search query for this image.")
    mermaid: str = Field(description="Mermaid diagram code if kind is diagram, otherwise an empty string.")
    prompt: str = Field(description="Detailed prompt for an AI image generation model.")


class ImagePlan(BaseModel):
    images: list[ImageSpec]


# ---------------------------------------------------------------- graph state
class State(TypedDict, total=False):
    # inputs
    topic: str
    as_of: str
    recency_days: int
    target_words: int
    enable_research: bool
    enable_images: bool
    max_images: int

    # router
    mode: Mode
    needs_research: bool
    router_reason: str
    queries: list[str]

    # research
    evidence: list[dict]

    # planner
    plan: Plan

    # workers (fan-in: each worker appends (task_id, markdown))
    sections: Annotated[list[tuple[int, str]], operator.add]

    # reducer
    run_dir: str
    merged_md: str
    md_with_placeholders: str
    image_specs: list[dict]
    final: str
