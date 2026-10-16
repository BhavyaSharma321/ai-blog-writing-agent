"""CLI: python main.py "Your blog topic" [--words 1500] [--no-research] [--no-images]"""

from __future__ import annotations

import argparse
import logging

from blog_agent import Settings, build_graph, initial_state, run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="AI planning agent that researches and writes technical blogs.")
    parser.add_argument("topic")
    parser.add_argument("--words", type=int, default=1500, help="Approximate total length.")
    parser.add_argument("--recency-days", type=int, default=30, help="News window for open-book topics.")
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--no-research", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    state = initial_state(
        args.topic,
        target_words=args.words,
        recency_days=args.recency_days,
        enable_research=not args.no_research,
        enable_images=not args.no_images,
        max_images=args.max_images,
    )

    final: dict = {}
    for namespace, update in build_graph().stream(
        state, run_config(settings), stream_mode="updates", subgraphs=True
    ):
        for node, data in update.items():
            prefix = "  reducer > " if namespace else ""
            print(f"[done] {prefix}{node}")
            if not namespace and data:
                final.update(data)

    print(f"\nMode: {final.get('mode')} - {final.get('router_reason')}")
    print(f"Saved to: {final.get('run_dir')}/blog.md")


if __name__ == "__main__":
    main()
