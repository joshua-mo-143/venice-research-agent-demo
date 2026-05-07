from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_agent.agent import ResearchAgent, _report_section_prompt
from research_agent.artifacts import ArtifactWriter
from research_agent.models import SearchResult, SourceNote, WebPage, chunk_text


class FakeVenice:
    def __init__(self) -> None:
        self.report_prompts: list[str] = []
        self.outline_prompts: list[str] = []
        self.section_prompts: list[str] = []
        self.editor_prompts: list[str] = []
        self.gap_prompts: list[str] = []
        self.stream_prompts: list[str] = []

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1600,
    ) -> str:
        self.stream_prompts.append(messages[-1]["content"])
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

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
        if "Identify coverage gaps before the next research pass" in prompt:
            self.gap_prompts.append(prompt)
            return json.dumps(
                {
                    "gaps": [
                        {
                            "missing": "Modern agent design patterns",
                            "why_it_matters": "The report needs concrete architecture coverage.",
                            "query": "agent research ReAct Reflexion LATS architecture patterns",
                        }
                    ],
                    "queries": ["agent research ReAct Reflexion LATS architecture patterns"],
                }
            )
        if "Plan a staged deep research report" in prompt:
            self.outline_prompts.append(prompt)
            return json.dumps(
                {
                    "title": "Agent Research Deep Dive",
                    "thesis": "Agent research needs concrete synthesis.",
                    "sections": [
                        {
                            "heading": "Concrete Findings",
                            "purpose": "Explain the concrete findings.",
                            "questions": ["What did the sources show?"],
                            "source_ids": ["S1"],
                            "expected_tables": ["Findings table"],
                        },
                        {
                            "heading": "What This Means in Practice",
                            "purpose": "Translate findings into practical takeaways.",
                            "questions": ["What should readers do next?"],
                            "source_ids": ["S1"],
                            "expected_tables": [],
                        },
                    ],
                }
            )
        if "Draft one deep report section" in prompt:
            self.section_prompts.append(prompt)
            return "## Concrete Findings\n\nUseful staged section synthesis with citation [S1]."
        if "Assemble the final deep research report" in prompt:
            self.editor_prompts.append(prompt)
            return "# Final Deep Report\n\n## Overview\n\nUseful staged synthesis [^1].\n\n## References\n\n1. [First source](https://example.com/a) - Useful evidence."
        if "Report style:" in prompt:
            self.report_prompts.append(prompt)
            return "# Report\n\nFinding with citation [S1].\n\n## Sources\n\n[S1] Source"
        return json.dumps({"queries": ["fallback"]})


class InvalidOutlineVenice(FakeVenice):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1600,
    ) -> str:
        prompt = messages[-1]["content"]
        if "Plan a staged deep research report" in prompt:
            self.outline_prompts.append(prompt)
            return "not json"
        return super().chat(messages, temperature=temperature, max_tokens=max_tokens)


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
            outline_records = (Path(tmp) / "report_outline.jsonl").read_text(encoding="utf-8")
            section_records = (Path(tmp) / "report_sections.jsonl").read_text(encoding="utf-8")
            editor_records = (Path(tmp) / "report_editor.jsonl").read_text(encoding="utf-8")

        self.assertEqual(len(report.sources), 1)
        self.assertIn("Useful staged synthesis", report.markdown)
        self.assertIn("content_hash", dedupe_records)
        self.assertEqual(len(chunk_records), 2)
        self.assertIn("Agent Research Deep Dive", outline_records)
        self.assertIn("Concrete Findings", section_records)
        self.assertIn("Final Deep Report", editor_records)
        self.assertEqual(report.sources[0].chunks[0].quotes, ("relevant evidence",))

    def test_follow_up_pass_uses_gap_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            venice = FakeVenice()
            agent = ResearchAgent(
                venice=venice,  # type: ignore[arg-type]
                web=FakeWeb(),  # type: ignore[arg-type]
                artifacts=ArtifactWriter(Path(tmp)),
                max_chunks_per_source=1,
                report_style="standard",
            )

            agent.run("agent research", iterations=2, query_count=1, results_per_query=1)

            gap_records = (Path(tmp) / "research_gaps.jsonl").read_text(
                encoding="utf-8"
            )
            query_records = (Path(tmp) / "queries.jsonl").read_text(encoding="utf-8")

        self.assertEqual(len(venice.gap_prompts), 1)
        self.assertIn("Source balance", venice.gap_prompts[0])
        self.assertIn("overrepresented source domains", venice.gap_prompts[0])
        self.assertIn("deliberately broaden beyond it", venice.gap_prompts[0])
        self.assertIn("Modern agent design patterns", gap_records)
        self.assertIn("source_balance", gap_records)
        self.assertIn("ReAct Reflexion LATS", query_records)

    def test_deep_report_style_uses_staged_report_writer(self) -> None:
        venice = FakeVenice()
        agent = ResearchAgent(
            venice=venice,  # type: ignore[arg-type]
            web=FakeWeb(),  # type: ignore[arg-type]
            max_chunks_per_source=1,
            report_style="deep",
        )

        agent.run("agent research", iterations=1, query_count=1, results_per_query=1)

        self.assertEqual(venice.report_prompts, [])
        self.assertEqual(len(venice.stream_prompts), 4)
        self.assertIn("Report style: deep", venice.outline_prompts[-1])
        self.assertIn("concrete synthesis", venice.outline_prompts[-1])
        self.assertIn("broad, source-backed research survey", venice.outline_prompts[-1])
        self.assertIn("Do not let one vendor", venice.outline_prompts[-1])
        self.assertIn("Draft one deep report section", venice.section_prompts[-1])
        self.assertIn("decision criteria", venice.section_prompts[-1])
        self.assertIn("Preserve substantive source-backed coverage", venice.section_prompts[-1])
        self.assertIn("overrepresents one vendor", venice.section_prompts[-1])
        self.assertIn("Assemble the final deep research report", venice.editor_prompts[-1])
        self.assertIn("Do not compress", venice.editor_prompts[-1])
        self.assertIn("source base is skewed", venice.editor_prompts[-1])
        self.assertIn("[^1]", venice.editor_prompts[-1])

    def test_standard_report_style_still_uses_single_report_prompt(self) -> None:
        venice = FakeVenice()
        agent = ResearchAgent(
            venice=venice,  # type: ignore[arg-type]
            web=FakeWeb(),  # type: ignore[arg-type]
            max_chunks_per_source=1,
            report_style="standard",
        )

        agent.run("agent research", iterations=1, query_count=1, results_per_query=1)

        self.assertEqual(venice.outline_prompts, [])
        self.assertEqual(venice.section_prompts, [])
        self.assertEqual(venice.editor_prompts, [])
        self.assertEqual(len(venice.stream_prompts), 1)
        self.assertIn("Report style: standard", venice.report_prompts[-1])
        self.assertIn("source-backed research survey", venice.report_prompts[-1])
        self.assertIn("Avoid source-cluster capture", venice.report_prompts[-1])

    def test_deep_report_falls_back_when_outline_json_is_invalid(self) -> None:
        venice = InvalidOutlineVenice()
        agent = ResearchAgent(
            venice=venice,  # type: ignore[arg-type]
            web=FakeWeb(),  # type: ignore[arg-type]
            max_chunks_per_source=1,
            report_style="deep",
        )

        report = agent.run("agent research", iterations=1, query_count=1, results_per_query=1)

        self.assertIn("Final Deep Report", report.markdown)
        self.assertEqual(len(venice.outline_prompts), 1)
        self.assertGreaterEqual(len(venice.section_prompts), 1)
        self.assertIn("Core Concepts and Historical Context", venice.section_prompts[0])

    def test_deep_report_prompt_preserves_late_sources(self) -> None:
        notes = [
            SourceNote(
                source_id=f"S{index}",
                title=f"Source {index}",
                url=f"https://example.com/{index}",
                final_url=f"https://example.com/{index}",
                canonical_url=f"https://example.com/{index}",
                query="agent research",
                summary=f"Detailed source {index} evidence. " * 50,
            )
            for index in range(1, 46)
        ]

        outline = {
            "title": "Agent Research",
            "thesis": "Agent research needs useful synthesis.",
            "sections": [
                {
                    "heading": "Late Source Coverage",
                    "purpose": "Use late sources in staged section prompts.",
                    "questions": ["Are late sources preserved?"],
                    "source_ids": ["S45"],
                    "expected_tables": [],
                }
            ],
        }
        prompt = _report_section_prompt(
            "agent research",
            notes,
            [notes[-1]],
            outline,
            outline["sections"][0],  # type: ignore[index]
        )

        self.assertIn("[S45] Source 45", prompt)
        self.assertIn("Detailed source 45 evidence", prompt)


if __name__ == "__main__":
    unittest.main()
