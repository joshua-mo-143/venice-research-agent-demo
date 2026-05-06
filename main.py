from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from research_agent import (
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_CHUNKS_PER_SOURCE,
    DEFAULT_MAX_SOURCES,
    DEFAULT_QUERY_COUNT,
    DEFAULT_REPORT_STYLE,
    DEFAULT_RESULTS_PER_QUERY,
    ResearchAgent,
)
from research_agent.artifacts import ArtifactWriter
from research_agent.venice import VeniceClient, VeniceError
from research_agent.web import WebSearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal deep research agent powered by Venice AI.",
    )
    parser.add_argument("topic", nargs="+", help="Research topic, wrapped in quotes for best results.")
    parser.add_argument("--model", help="Venice model name. Defaults to VENICE_MODEL or venice-uncensored.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Research passes to run. Default: {DEFAULT_ITERATIONS}.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=DEFAULT_QUERY_COUNT,
        help=f"Search queries per pass. Default: {DEFAULT_QUERY_COUNT}.",
    )
    parser.add_argument(
        "--results",
        type=int,
        default=DEFAULT_RESULTS_PER_QUERY,
        help=f"Search results to read per provider per query. Default: {DEFAULT_RESULTS_PER_QUERY}.",
    )
    parser.add_argument(
        "--output",
        "--markdown-output",
        dest="output",
        type=Path,
        help="Optional path to write the Markdown report.",
    )
    parser.add_argument("--artifacts", type=Path, help="Optional directory for JSONL research artifacts.")
    parser.add_argument(
        "--providers",
        default="duckduckgo",
        help="Comma-separated source providers. Available: duckduckgo, arxiv. Default: duckduckgo.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=DEFAULT_MAX_SOURCES,
        help=f"Cap on usable sources collected. Default: {DEFAULT_MAX_SOURCES}.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=3000,
        help="Characters per extracted evidence chunk. Default: 3000.",
    )
    parser.add_argument(
        "--max-chunks-per-source",
        type=int,
        default=DEFAULT_MAX_CHUNKS_PER_SOURCE,
        help=f"Maximum chunks summarized per source. Default: {DEFAULT_MAX_CHUNKS_PER_SOURCE}.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Minimum seconds between requests to the same host. Default: 0.",
    )
    parser.add_argument("--web-retries", type=int, default=2, help="Retries for web search/fetch. Default: 2.")
    parser.add_argument("--venice-retries", type=int, default=2, help="Retries for Venice calls. Default: 2.")
    parser.add_argument(
        "--report-style",
        choices=["brief", "standard", "deep"],
        default=DEFAULT_REPORT_STYLE,
        help=f"Final report depth. Default: {DEFAULT_REPORT_STYLE}.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide progress messages.")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    topic = " ".join(args.topic)

    try:
        venice = VeniceClient.from_env(model=args.model, max_retries=args.venice_retries)
        progress = None if args.quiet else lambda message: print(f"[agent] {message}")
        provider_names = [name.strip() for name in args.providers.split(",") if name.strip()]
        with WebSearch.from_provider_names(
            provider_names,
            max_retries=args.web_retries,
            request_delay_seconds=args.request_delay,
            chunk_chars=args.chunk_chars,
            scraper=venice.scrape,
        ) as web:
            agent = ResearchAgent(
                venice=venice,
                web=web,
                artifacts=ArtifactWriter(args.artifacts),
                progress=progress,
                max_sources=args.max_sources,
                max_chunks_per_source=args.max_chunks_per_source,
                report_style=args.report_style,
            )
            report = agent.run(
                topic,
                iterations=args.iterations,
                query_count=args.queries,
                results_per_query=args.results,
            )
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except VeniceError as exc:
        print(f"Venice API error: {exc}")
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.markdown, encoding="utf-8")
        print(f"\nSaved report to {args.output}")
    else:
        print()
        print(report.markdown)

    if report.artifacts_dir:
        print(f"Saved research artifacts to {report.artifacts_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
