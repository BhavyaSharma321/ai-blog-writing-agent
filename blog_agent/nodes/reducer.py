"""Reducer sub-graph: merge sections -> decide images -> generate & place images."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from blog_agent import images
from blog_agent.config import get_settings
from blog_agent.llm import structured
from blog_agent.prompts import IMAGE_KIND_RULES, IMAGE_PLANNER_SYSTEM
from blog_agent.schemas import ImagePlan, State
from blog_agent.utils import insert_after_heading, new_run_dir, slugify

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- 1. merge
def merge_node(state: State, config: RunnableConfig) -> dict:
    settings = get_settings(config)
    plan = state["plan"]
    sections = sorted(state.get("sections", []), key=lambda s: s[0])
    body = "\n\n".join(md.strip() for _, md in sections)
    md = f"# {plan.blog_title}\n\n{body}\n"

    cited = list({e["url"].rstrip("/"): e for e in state.get("evidence", []) if e["url"] in md}.values())
    if cited:
        md += "\n## References\n\n" + "\n".join(f"- [{e['title']}]({e['url']})" for e in cited) + "\n"

    run_dir = new_run_dir(settings.output_dir, plan.blog_title)
    (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "evidence.json").write_text(json.dumps(state.get("evidence", []), indent=2), encoding="utf-8")
    meta = {k: state.get(k) for k in ("topic", "as_of", "mode", "router_reason", "queries")}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"merged_md": md, "run_dir": str(run_dir)}


# ---------------------------------------------------------------- 2. decide images
def _outline(md: str, chars: int = 250) -> str:
    """Each section's heading + opening text: enough to place images, ~5x fewer tokens than the full blog."""
    parts = re.split(r"(?m)^(?=## )", md)
    return "\n\n".join(p.strip()[:chars] for p in parts if p.strip().startswith("## "))


def decide_images_node(state: State, config: RunnableConfig) -> dict:
    settings = get_settings(config)
    md = state["merged_md"]
    max_images = state.get("max_images", 3)
    if not (state.get("enable_images", True) and settings.can_make_images and max_images > 0):
        return {"md_with_placeholders": md, "image_specs": []}

    headings = [h[3:].strip() for h in md.split("\n") if h.startswith("## ") and h[3:].strip() != "References"]
    messages = [
        SystemMessage(
            IMAGE_PLANNER_SYSTEM.format(
                max_images=max_images,
                kind_rule=IMAGE_KIND_RULES.get(settings.image_source, IMAGE_KIND_RULES["auto"]),
            )
        ),
        HumanMessage("Headings:\n" + "\n".join(f"- {h}" for h in headings) + f"\n\nBlog outline:\n{_outline(md)}"),
    ]
    try:
        image_plan: ImagePlan = structured(settings, ImagePlan).invoke(messages)
    except Exception as exc:  # images are optional: never fail the whole blog over them
        log.warning("Image planning failed: %s", str(exc)[:300])
        return {"md_with_placeholders": md, "image_specs": []}

    specs, used = [], set()
    for spec in image_plan.images:
        if len(specs) >= max_images or spec.after_heading in used:
            continue
        n = len(specs) + 1
        placeholder = f"[[IMAGE_{n}]]"
        md, ok = insert_after_heading(md, spec.after_heading, placeholder)
        if ok:
            used.add(spec.after_heading)
            specs.append({**spec.model_dump(), "placeholder": placeholder, "filename": f"img{n}-{slugify(spec.alt, 40)}"})
    return {"md_with_placeholders": md, "image_specs": specs}


# ---------------------------------------------------------------- 3. generate & place
def generate_images_node(state: State, config: RunnableConfig) -> dict:
    settings = get_settings(config)
    md = state["md_with_placeholders"]
    specs = state.get("image_specs", [])
    run_dir = Path(state["run_dir"])

    def work(spec: dict) -> dict:
        try:
            image = images.make_image(settings, spec, used_photo_ids)
            ext = {"image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(image.mime, ".png")
            path = run_dir / "images" / f"{spec['filename']}{ext}"
            path.write_bytes(image.data)
            return {**spec, "path": f"images/{path.name}", "backend": image.backend, "credit": image.credit, "error": None}
        except Exception as exc:
            log.warning("Image generation failed for %s: %s", spec["placeholder"], exc)
            return {**spec, "path": None, "backend": None, "credit": "", "error": str(exc)[:500]}

    used_photo_ids: set[str] = set()  # no duplicate photos within one blog
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(work, specs))

    for r in results:
        caption = f"*{r['caption']}*" + (f"  \n{r['credit']}" if r["credit"] else "")
        replacement = f"![{r['alt']}]({r['path']})\n{caption}" if r["path"] else ""
        md = md.replace(r["placeholder"], replacement)
    md = re.sub(r"\n{3,}", "\n\n", md)

    (run_dir / "blog.md").write_text(md, encoding="utf-8")
    (run_dir / "images.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {"final": md, "image_specs": results}


# ---------------------------------------------------------------- sub-graph
def build_reducer_graph():
    g = StateGraph(State)
    g.add_node("merge", merge_node)
    g.add_node("decide_images", decide_images_node)
    g.add_node("generate_images", generate_images_node)
    g.add_edge(START, "merge")
    g.add_edge("merge", "decide_images")
    g.add_edge("decide_images", "generate_images")
    g.add_edge("generate_images", END)
    return g.compile()


_reducer_graph = None
OUTPUT_KEYS = ("run_dir", "merged_md", "md_with_placeholders", "image_specs", "final")


def reducer_node(state: State, config: RunnableConfig) -> dict:
    # Wrapped in a function so the sub-graph doesn't re-emit `sections`
    # (which would be appended twice by the operator.add reducer).
    global _reducer_graph
    if _reducer_graph is None:
        _reducer_graph = build_reducer_graph()
    out = _reducer_graph.invoke(state, config)
    return {k: out[k] for k in OUTPUT_KEYS if k in out}
