"""Minimal Venice-powered deep research agent demo."""

from .agent import (
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_CHUNKS_PER_SOURCE,
    DEFAULT_MAX_SOURCES,
    DEFAULT_QUERY_COUNT,
    DEFAULT_REPORT_STYLE,
    DEFAULT_RESULTS_PER_QUERY,
    ResearchAgent,
)
from .models import ResearchReport, SourceNote

__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_MAX_CHUNKS_PER_SOURCE",
    "DEFAULT_MAX_SOURCES",
    "DEFAULT_QUERY_COUNT",
    "DEFAULT_REPORT_STYLE",
    "DEFAULT_RESULTS_PER_QUERY",
    "ResearchAgent",
    "ResearchReport",
    "SourceNote",
]
