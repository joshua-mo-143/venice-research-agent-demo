from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_agent.agent import ResearchAgent
from research_agent.artifacts import ArtifactWriter
from research_agent.models import SearchResult, WebPage, chunk_text


class FakeVenice:
    def __init__(self) -> None:
        self.report_prompts: list[str] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1600,
    ) -> str:
        prompt = messages[-1]["content"]
        if prompt.startswith("Create"):
            return json.dumps({"queries": ["agent research primary source"]})
        if "Extract only evidence" in prompt:
            return json.dumps({"summary": "The chunk contains relevant evidence.", "quotes": ["relevant evidence"]})
        if prompt.startswith("Topic:") and "Synthesize a source note" in prompt:
            return "Source note with relevant evidence [S1]."
        if "Report style:" in prompt:
            self.report_prompts.append(prompt)
            return "# Report\n\nFinding with citation [S1].\n\n## Sources\n\n[S1] Source"
        return json.dumps({"queries": ["fallback"]})


class FakeWeb:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                title="First source",
                url="https://example.com/a",
                snippet="snippet",
                query=query,
                rank=1,
            ),
            SearchResult(
                title="Duplicate content",
                url="https://example.com/b",
                snippet="snippet",
                query=query,
                rank=2,
            ),
        ][:limit]

    def fetch(self, result: SearchResult) -> WebPage:
        text = "This page contains relevant evidence for the research topic. " * 4
        return WebPage(
            title=result.title,
            url=result.url,
            final_url=result.url,
            text=text,
            content_type="text/html",
            content_hash="same-content",
            chunks=chunk_text(text, chunk_chars=80, overlap=10),
        )


class ResearchAgentTest(unittest.TestCase):
    def test_run_persists_artifacts_and_skips_duplicate_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = ResearchAgent(
                venice=FakeVenice(),  # type: ignore[arg-type]
                web=FakeWeb(),  # type: ignore[arg-type]
                artifacts=ArtifactWriter(Path(tmp)),
                max_chunks_per_source=2,
            )

            report = agent.run("agent research", iterations=1, query_count=1, results_per_query=2)

            dedupe_records = (Path(tmp) / "dedupe.jsonl").read_text(encoding="utf-8")
            chunk_records = (Path(tmp) / "source_chunks.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(report.sources), 1)
        self.assertIn("Finding with citation [S1]", report.markdown)
        self.assertIn("content_hash", dedupe_records)
        self.assertEqual(len(chunk_records), 2)
        self.assertEqual(report.sources[0].chunks[0].quotes, ("relevant evidence",))

    def test_deep_report_style_uses_expanded_report_prompt(self) -> None:
        venice = FakeVenice()
        agent = ResearchAgent(
            venice=venice,  # type: ignore[arg-type]
            web=FakeWeb(),  # type: ignore[arg-type]
            max_chunks_per_source=1,
            report_style="deep",
        )

        agent.run("agent research", iterations=1, query_count=1, results_per_query=1)

        self.assertIn("Report style: deep", venice.report_prompts[-1])
        self.assertIn("source-by-source notes", venice.report_prompts[-1])


if __name__ == "__main__":
    unittest.main()
