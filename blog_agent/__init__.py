"""Planning agent that researches and writes technical blogs.

Pipeline (LangGraph):
    router -> [research] -> orchestrator -> worker x N (fan-out) -> reducer
                                                                 (merge -> decide_images -> generate_images)
"""

from blog_agent.config import Settings
from blog_agent.graph import build_graph, initial_state, run, run_config

__all__ = ["Settings", "build_graph", "initial_state", "run", "run_config"]
