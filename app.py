"""Streamlit frontend for the Blog Writing Agent."""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from pathlib import Path

import streamlit as st

from blog_agent import Settings, build_graph, initial_state, run_config
from blog_agent.config import PROVIDERS
from blog_agent.llm import list_image_models, list_text_models

st.set_page_config(page_title="AI Blog Writing Agent", page_icon="✍️", layout="wide")

STEP_LABELS = {
    "router": "🧭 Router — deciding if research is needed",
    "research": "🔎 Research — Tavily search + evidence pack",
    "orchestrator": "🗺️ Orchestrator — planning sections",
    "worker": "✍️ Worker — drafted a section",
    "merge": "🧩 Reducer — merged sections",
    "decide_images": "🖼️ Reducer — planned images",
    "generate_images": "🎨 Reducer — generated & placed images",
    "reducer": "✅ Reducer finished",
}
MODE_BADGE = {"closed_book": "📕 Closed book", "hybrid": "📗 Hybrid", "open_book": "🌐 Open book"}
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


# ---------------------------------------------------------------- helpers
def secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml locally
        pass
    return os.getenv(name, "")


@st.cache_resource
def get_graph():
    return build_graph()


def render_blog(md: str, base_dir: Path) -> None:
    """st.markdown can't show local image files, so split them out and use st.image."""
    pos = 0
    for m in IMAGE_RE.finditer(md):
        if text := md[pos : m.start()].strip():
            st.markdown(text)
        img = base_dir / m.group(2)
        if img.exists():
            st.image(str(img), width="stretch")
        else:
            st.markdown(m.group(0))
        pos = m.end()
    if text := md[pos:].strip():
        st.markdown(text)


def zip_run(run_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in run_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(run_dir))
    return buf.getvalue()


def load_run(run_dir: Path) -> dict:
    def read_json(name, default):
        p = run_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    return {
        "run_dir": str(run_dir),
        "final": (run_dir / "blog.md").read_text(encoding="utf-8"),
        "plan": read_json("plan.json", None),
        "evidence": read_json("evidence.json", []),
        **read_json("meta.json", {}),
    }


def past_runs(output_dir: str) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if (p / "blog.md").exists()), reverse=True)


# ---------------------------------------------------------------- sidebar
@st.cache_data(ttl=3600, show_spinner="Fetching available models…")
def text_models(provider: str, api_key: str) -> list[str]:
    return list_text_models(provider, api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def image_models(api_key: str) -> list[str]:
    return list_image_models(api_key)


def pick(label: str, options: list[str], preferred: str, **kwargs) -> str:
    index = options.index(preferred) if preferred in options else 0
    return st.selectbox(label, options, index=index, **kwargs)


def key_input(label: str, name: str) -> str:
    """Server keys (secrets / .env) stay on the server: never put them into a widget,
    because widget values are sent to the browser where any visitor could read them."""
    if server_key := secret(name):
        st.markdown(f"✅ **{label}** configured")
        return server_key
    return st.text_input(label, type="password", placeholder="Paste your key")


with st.sidebar:
    st.header("🔑 API keys")
    groq_key = key_input("Groq API key", "GROQ_API_KEY")
    google_key = key_input("Gemini API key", "GOOGLE_API_KEY")
    tavily_key = key_input("Tavily API key (research, optional)", "TAVILY_API_KEY")

    st.header("🧠 Writing model")
    keyed = Settings(groq_api_key=groq_key, google_api_key=google_key)
    provider = pick(
        "LLM provider",
        list(PROVIDERS),
        (secret("LLM_PROVIDER") or keyed.provider).lower(),
        format_func=lambda p: PROVIDERS[p]["label"],
    )
    llm_model = pick(
        "Model",
        text_models(provider, keyed.key_for(provider)),
        secret("LLM_MODEL"),
        key=f"model_{provider}",
        help="First options are recommended; the rest are fetched live from your account.",
    )

    st.header("⚙️ Options")
    target_words = st.slider("Blog length (words)", 600, 4000, 1500, step=100)
    enable_research = st.toggle("Web research (Tavily)", value=True, disabled=not tavily_key)
    recency_days = st.slider("News window (days, open-book topics)", 1, 365, 30)
    enable_images = st.toggle("Generate images (Gemini)", value=True, disabled=not google_key)
    max_images = st.slider("Max images", 0, 5, 3, disabled=not (enable_images and google_key))
    image_model = pick(
        "Gemini image model",
        image_models(google_key),
        secret("GEMINI_IMAGE_MODEL"),
        disabled=not (enable_images and google_key),
    )

settings = Settings.from_env(
    llm_provider=provider,
    llm_model=llm_model,
    groq_api_key=groq_key,
    tavily_api_key=tavily_key,
    google_api_key=google_key,
    image_model=image_model,
)

with st.sidebar:
    runs = past_runs(settings.output_dir)
    if runs:
        st.header("📚 Past blogs")
        choice = st.selectbox("Open a previous blog", ["—"] + [p.name for p in runs])
        if choice != "—" and st.button("Load"):
            st.session_state.result = load_run(Path(settings.output_dir) / choice)


# ---------------------------------------------------------------- main
st.title("✍️ AI Blog Writing Agent")
st.caption("Router → Research (Tavily) → Orchestrator → Parallel Workers → Reducer (merge + Gemini images) · built with LangGraph")

with st.form("topic_form"):
    topic = st.text_area(
        "What should the blog be about?",
        placeholder="e.g. How LangGraph's Send API enables parallel agents  •  Top open-source LLM releases this month",
        height=90,
    )
    submitted = st.form_submit_button("🚀 Generate blog", type="primary", width="stretch")

if submitted:
    if not topic.strip():
        st.warning("Please enter a topic.")
    elif not settings.llm_api_key:
        st.error(f"Add an API key for {PROVIDERS[settings.provider]['label']} in the sidebar (or Streamlit secrets).")
    else:
        state = initial_state(
            topic,
            target_words=target_words,
            recency_days=recency_days,
            enable_research=enable_research and settings.can_research,
            enable_images=enable_images and settings.can_make_images,
            max_images=max_images,
        )
        result: dict = {}
        with st.status("Agent is working…", expanded=True) as status:
            try:
                stream = get_graph().stream(
                    state, run_config(settings), stream_mode="updates", subgraphs=True
                )
                for namespace, update in stream:
                    for node, data in update.items():
                        data = data or {}
                        label = STEP_LABELS.get(node, node)
                        if node == "router":
                            label += f" → **{MODE_BADGE.get(data.get('mode'), data.get('mode'))}**"
                        elif node == "research":
                            label += f" → {len(data.get('evidence', []))} sources"
                        elif node == "orchestrator":
                            label += f" → {len(data['plan'].tasks)} sections"
                        elif node == "worker":
                            sid, md = data["sections"][0]
                            label += f" #{sid}: {md.splitlines()[0].lstrip('# ')}"
                        st.write(label)
                        if not namespace:
                            result.update(data)
                status.update(label="Blog ready!", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Something went wrong", state="error")
                st.exception(exc)
                result = {}
        failed = [i for i in result.get("image_specs") or [] if i.get("error")]
        if failed:
            reason = failed[0]["error"]
            hint = " Gemini image models need billing enabled on your Google AI Studio project." if "429" in reason or "RESOURCE_EXHAUSTED" in reason else ""
            st.warning(
                f"{len(failed)} image(s) could not be generated, so the blog has no images there.{hint}"
                f"\n\nError: {reason[:200]}"
            )
        if result.get("final"):
            result["topic"] = topic
            if result.get("plan") is not None:
                result["plan"] = result["plan"].model_dump()
            st.session_state.result = result

res = st.session_state.get("result")
if res:
    run_dir = Path(res["run_dir"])
    cols = st.columns(4)
    cols[0].metric("Mode", MODE_BADGE.get(res.get("mode"), res.get("mode") or "—"))
    cols[1].metric("Sections", len((res.get("plan") or {}).get("tasks", [])))
    cols[2].metric("Sources", len(res.get("evidence") or []))
    cols[3].metric("Words", len(res["final"].split()))

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download Markdown", res["final"], file_name="blog.md", width="stretch")
    d2.download_button(
        "⬇️ Download ZIP (md + images)",
        zip_run(run_dir),
        file_name=f"{run_dir.name}.zip",
        mime="application/zip",
        width="stretch",
    )

    tab_blog, tab_plan, tab_research, tab_md = st.tabs(["📰 Blog", "🗺️ Plan", "🔎 Research", "📝 Markdown"])
    with tab_blog:
        render_blog(res["final"], run_dir)
    with tab_plan:
        plan = res.get("plan")
        if plan:
            st.subheader(plan["blog_title"])
            st.write(f"**Audience:** {plan['audience']}  ·  **Tone:** {plan['tone']}  ·  **Kind:** {plan['blog_kind']}")
            if plan.get("constraints"):
                st.markdown("**Constraints:**\n" + "\n".join(f"- {c}" for c in plan["constraints"]))
            for t in plan["tasks"]:
                flags = " ".join(f for f, on in (("💻 code", t["requires_code"]), ("🔗 citations", t["requires_citations"])) if on)
                with st.expander(f"{t['id']}. {t['title']}  ·  ~{t['target_words']} words  {flags}"):
                    st.write(f"**Goal:** {t['goal']}")
                    st.markdown("\n".join(f"- {b}" for b in t["bullets"]))
        else:
            st.info("No plan saved for this run.")
    with tab_research:
        if res.get("router_reason"):
            st.info(f"**Router:** {res['router_reason']}")
        if res.get("queries"):
            st.markdown("**Search queries:**\n" + "\n".join(f"- `{q}`" for q in res["queries"]))
        for e in res.get("evidence") or []:
            date = f" · {e['published_at']}" if e.get("published_at") else ""
            st.markdown(f"**[{e['title']}]({e['url']})**{date}  \n{e['snippet']}")
        if not res.get("evidence"):
            st.caption("No web research was used for this blog.")
    with tab_md:
        st.code(res["final"], language="markdown")
else:
    st.info("Enter a topic and hit **Generate blog**. Add your keys in the sidebar (or in Streamlit secrets).")
