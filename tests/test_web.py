from __future__ import annotations

import unittest

from research_agent.models import ScrapeResult, SearchResult
from research_agent.web import SearchProvider, WebSearch


class StaticProvider(SearchProvider):
    name = "static"

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, web: WebSearch, query: str, limit: int) -> list[SearchResult]:
        return self.results[:limit]


class WebSearchTest(unittest.TestCase):
    def test_search_dedupes_provider_results_by_canonical_url(self) -> None:
        duplicate_a = SearchResult(title="A", url="https://example.com/post?utm_source=x", snippet="")
        duplicate_b = SearchResult(title="B", url="https://example.com/post#frag", snippet="")
        unique = SearchResult(title="C", url="https://example.org/other", snippet="")
        web = WebSearch(providers=[StaticProvider([duplicate_a]), StaticProvider([duplicate_b, unique])])

        try:
            results = web.search("topic", limit=5)
        finally:
            web.close()

        self.assertEqual([result.title for result in results], ["A", "C"])

    def test_fetch_uses_scraper_result_as_markdown_source(self) -> None:
        def scrape(url: str) -> ScrapeResult:
            return ScrapeResult(
                url=url,
                final_url="https://example.com/final",
                title="Scraped title",
                content="# Heading\n\nScraped markdown content.",
            )

        web = WebSearch(scraper=scrape, chunk_chars=100)
        try:
            page = web.fetch(SearchResult(title="Search title", url="https://example.com", snippet="Snippet"))
        finally:
            web.close()

        self.assertEqual(page.title, "Scraped title")
        self.assertEqual(page.final_url, "https://example.com/final")
        self.assertEqual(page.content_type, "text/markdown")
        self.assertEqual(page.text, "# Heading\n\nScraped markdown content.")
        self.assertEqual(len(page.chunks), 1)


if __name__ == "__main__":
    unittest.main()
