from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "blog"


def new_run_dir(output_dir: str, title: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(output_dir) / f"{stamp}-{slugify(title)}"
    (path / "images").mkdir(parents=True, exist_ok=True)
    return path


def format_evidence(evidence: list[dict], snippet_chars: int = 500) -> str:
    if not evidence:
        return "(no evidence available)"
    lines = []
    for i, e in enumerate(evidence, 1):
        date = f" ({e['published_at']})" if e.get("published_at") else ""
        lines.append(f"[{i}] {e['title']}{date}\n    URL: {e['url']}\n    {e['snippet'][:snippet_chars]}")
    return "\n".join(lines)


def _norm_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lstrip("#").strip().lower())


def insert_after_heading(md: str, heading: str, block: str) -> tuple[str, bool]:
    """Insert `block` after the first paragraph that follows `heading`."""
    lines = md.split("\n")
    target = _norm_heading(heading)
    for i, line in enumerate(lines):
        if line.startswith("#") and _norm_heading(line) == target:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            # Walk past one plain paragraph (not a code fence / heading / table).
            if j < len(lines) and not lines[j].lstrip().startswith(("```", "#", "|")):
                while j < len(lines) and lines[j].strip():
                    j += 1
            else:
                j = i + 1
            lines[j:j] = ["", block, ""]
            return "\n".join(lines), True
    return md, False
