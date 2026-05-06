from __future__ import annotations

import json
from collections.abc import Callable
from textwrap import dedent
from typing import Literal

from .artifacts import ArtifactWriter
from .models import (
    CollectionError,
    EvidenceChunk,
    ResearchReport,
    SearchResult,
    SourceNote,
    WebPage,
    utc_now,
)
from .venice import VeniceClient, VeniceError
from .web import WebSearch

SYSTEM_PROMPT = """You are a careful research assistant.
Use the supplied source material only when making factual claims.
Flag uncertainty, contradictions, and missing context instead of filling gaps."""


ProgressCallback = Callable[[str], None]
ReportStyle = Literal["brief", "standard", "deep"]
DEFAULT_ITERATIONS = 3
DEFAULT_QUERY_COUNT = 6
DEFAULT_RESULTS_PER_QUERY = 4
DEFAULT_MAX_SOURCES = 25
DEFAULT_MAX_CHUNKS_PER_SOURCE = 6
DEFAULT_REPORT_STYLE: ReportStyle = "deep"
REPORT_TOKEN_BUDGETS: dict[ReportStyle, int] = {
    "brief": 2400,
    "standard": 7000,
    "deep": 16000,
}
REPORT_SOURCE_DIGEST_CHAR_LIMITS: dict[ReportStyle, int] = {
    "brief": 18000,
    "standard": 45000,
    "deep": 90000,
}
REPORT_CHUNK_DIGEST_CHAR_LIMITS: dict[ReportStyle, int] = {
    "brief": 500,
    "standard": 700,
    "deep": 900,
}
REPORT_OUTLINE_TOKEN_BUDGET = 2200
REPORT_SECTION_TOKEN_BUDGET = 4200
REPORT_EDITOR_TOKEN_BUDGET = 16000
MAX_DEEP_REPORT_SECTIONS = 8


class ResearchAgent:
    def __init__(
        self,
        venice: VeniceClient,
        web: WebSearch | None = None,
        artifacts: ArtifactWriter | None = None,
        progress: ProgressCallback | None = None,
        max_sources: int | None = DEFAULT_MAX_SOURCES,
        max_chunks_per_source: int = DEFAULT_MAX_CHUNKS_PER_SOURCE,
        report_style: ReportStyle = DEFAULT_REPORT_STYLE,
    ) -> None:
        if report_style not in REPORT_TOKEN_BUDGETS:
            raise ValueError(f"Unknown report style: {report_style}")

        self.venice = venice
        self.web = web or WebSearch(scraper=venice.scrape)
        self.artifacts = artifacts or ArtifactWriter()
        self.progress = progress or (lambda _: None)
        self.max_sources = max_sources
        self.max_chunks_per_source = max_chunks_per_source
        self.report_style = report_style

    def run(
        self,
        topic: str,
        *,
        iterations: int = DEFAULT_ITERATIONS,
        query_count: int = DEFAULT_QUERY_COUNT,
        results_per_query: int = DEFAULT_RESULTS_PER_QUERY,
    ) -> ResearchReport:
        notes: list[SourceNote] = []
        seen_source_keys: set[str] = set()
        seen_content_hashes: set[str] = set()
        queries = self._initial_queries(topic, query_count)
        self.artifacts.write(
            "queries", {"stage": "initial", "topic": topic, "queries": queries}
        )

        for iteration in range(1, iterations + 1):
            self.progress(
                f"Research pass {iteration}/{iterations}: {', '.join(queries)}"
            )
            self._collect_notes(
                topic,
                queries,
                results_per_query,
                seen_source_keys,
                seen_content_hashes,
                notes,
                iteration,
            )

            if iteration < iterations:
                queries = self._follow_up_queries(topic, notes, query_count)
                self.artifacts.write(
                    "queries",
                    {
                        "stage": "follow_up",
                        "topic": topic,
                        "iteration": iteration + 1,
                        "queries": queries,
                    },
                )

        report = self._write_report(topic, notes)
        self.artifacts.write(
            "reports",
            {
                "topic": topic,
                "source_count": len(notes),
                "report_style": self.report_style,
                "generated_at": utc_now(),
                "markdown": report,
            },
        )
        artifacts_dir = (
            str(self.artifacts.root) if self.artifacts.root is not None else None
        )
        return ResearchReport(
            topic=topic, markdown=report, sources=notes, artifacts_dir=artifacts_dir
        )

    def _collect_notes(
        self,
        topic: str,
        queries: list[str],
        results_per_query: int,
        seen_source_keys: set[str],
        seen_content_hashes: set[str],
        notes: list[SourceNote],
        iteration: int,
    ) -> None:
        for query in queries:
            if self._source_budget_reached(notes):
                return

            self.progress(f"Searching: {query}")
            try:
                results = self.web.search(query, limit=results_per_query)
            except Exception as exc:  # noqa: BLE001 - demo should keep researching if one search fails.
                self._record_error("search", exc, query=query)
                continue

            self.artifacts.write(
                "search_results",
                {"iteration": iteration, "query": query, "results": results},
            )

            for result in results:
                if self._source_budget_reached(notes):
                    return

                source_key = result.canonical_url or result.url
                if source_key in seen_source_keys:
                    self.artifacts.write(
                        "dedupe",
                        {
                            "reason": "canonical_url",
                            "query": query,
                            "url": result.url,
                            "canonical_url": source_key,
                            "provider": result.provider,
                        },
                    )
                    continue

                seen_source_keys.add(source_key)
                source_id = f"S{len(notes) + 1}"
                note = self._read_source(
                    topic,
                    query,
                    source_id,
                    result,
                    seen_source_keys,
                    seen_content_hashes,
                )
                if note is not None:
                    notes.append(note)

    def _read_source(
        self,
        topic: str,
        query: str,
        source_id: str,
        result: SearchResult,
        seen_source_keys: set[str],
        seen_content_hashes: set[str],
    ) -> SourceNote | None:
        self.progress(f"Reading {source_id}: {result.title}")
        try:
            page = self.web.fetch(result)
        except Exception as exc:  # noqa: BLE001 - inaccessible pages are common during web research.
            self._record_error(
                "fetch",
                exc,
                query=query,
                url=result.url,
                source_id=source_id,
                provider=result.provider,
            )
            return None

        if (
            page.canonical_url in seen_source_keys
            and page.canonical_url != result.canonical_url
        ):
            self.artifacts.write(
                "dedupe",
                {
                    "reason": "redirect_canonical_url",
                    "source_id": source_id,
                    "url": result.url,
                    "final_url": page.final_url,
                    "canonical_url": page.canonical_url,
                    "provider": result.provider,
                },
            )
            return None
        seen_source_keys.add(page.canonical_url)

        if not page.text:
            self._record_error(
                "extract",
                ValueError("no usable source text"),
                query=query,
                url=result.url,
                source_id=source_id,
                provider=result.provider,
            )
            return None

        if page.content_hash in seen_content_hashes:
            self.artifacts.write(
                "dedupe",
                {
                    "reason": "content_hash",
                    "source_id": source_id,
                    "url": result.url,
                    "final_url": page.final_url,
                    "content_hash": page.content_hash,
                    "provider": result.provider,
                },
            )
            return None
        seen_content_hashes.add(page.content_hash)

        self.artifacts.write(
            "fetches",
            {
                "source_id": source_id,
                "title": page.title,
                "url": result.url,
                "final_url": page.final_url,
                "canonical_url": page.canonical_url,
                "query": query,
                "rank": result.rank,
                "provider": result.provider,
                "content_type": page.content_type,
                "retrieved_at": page.retrieved_at,
                "content_hash": page.content_hash,
                "text_chars": len(page.text),
                "chunk_count": len(page.chunks),
            },
        )

        chunks = self._summarize_chunks(topic, query, source_id, page)
        if not chunks:
            self._record_error(
                "summarize_chunk",
                VeniceError("no chunks could be summarized"),
                query=query,
                url=result.url,
                source_id=source_id,
                provider=result.provider,
            )
            return None

        try:
            summary = self._summarize_source(topic, query, source_id, page, chunks)
        except Exception as exc:  # noqa: BLE001 - keep collection moving when one summary fails.
            self._record_error(
                "summarize_source",
                exc,
                query=query,
                url=result.url,
                source_id=source_id,
                provider=result.provider,
            )
            return None

        note = SourceNote(
            source_id=source_id,
            title=page.title,
            url=result.url,
            canonical_url=page.canonical_url,
            final_url=page.final_url,
            query=query,
            rank=result.rank,
            snippet=result.snippet,
            provider=result.provider,
            retrieved_at=page.retrieved_at,
            content_type=page.content_type,
            content_hash=page.content_hash,
            chunks=chunks,
            summary=summary,
        )
        self.artifacts.write("source_notes", note)
        return note

    def _summarize_chunks(
        self,
        topic: str,
        query: str,
        source_id: str,
        page: WebPage,
    ) -> tuple[EvidenceChunk, ...]:
        evidence: list[EvidenceChunk] = []
        for chunk in page.chunks[: self.max_chunks_per_source]:
            self.artifacts.write(
                "source_chunks",
                {
                    "source_id": source_id,
                    "chunk_id": chunk.chunk_id,
                    "url": page.final_url,
                    "content_hash": chunk.content_hash,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                },
            )

            prompt = dedent(
                f"""
                Topic: {topic}
                Search query: {query}
                Source ID: {source_id}
                Chunk ID: {chunk.chunk_id}
                Source title: {page.title}
                Source URL: {page.final_url}

                Source chunk:
                {chunk.text}

                Extract only evidence relevant to the topic.
                Return JSON only in this shape:
                {{"summary": "...", "quotes": ["short exact quote", "..."]}}
                """
            ).strip()

            try:
                response = self.venice.chat(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=600,
                )
                evidence_chunk = _parse_evidence_chunk(
                    chunk.chunk_id, chunk.text, response
                )
            except Exception as exc:  # noqa: BLE001 - one bad chunk should not kill the source.
                self._record_error(
                    "summarize_chunk",
                    exc,
                    query=query,
                    url=page.final_url,
                    source_id=source_id,
                )
                continue

            self.artifacts.write(
                "chunk_summaries",
                {
                    "source_id": source_id,
                    "chunk_id": evidence_chunk.chunk_id,
                    "summary": evidence_chunk.summary,
                    "quotes": evidence_chunk.quotes,
                },
            )
            evidence.append(evidence_chunk)

        return tuple(evidence)

    def _summarize_source(
        self,
        topic: str,
        query: str,
        source_id: str,
        page: WebPage,
        chunks: tuple[EvidenceChunk, ...],
    ) -> str:
        chunk_digest = _chunk_digest(chunks, max_chars=9000)
        prompt = dedent(
            f"""
            Topic: {topic}
            Search query: {query}
            Source ID: {source_id}
            Source title: {page.title}
            Source URL: {page.final_url}
            Retrieved at: {page.retrieved_at}
            Content hash: {page.content_hash}

            Chunk evidence:
            {chunk_digest}

            Synthesize a source note using only the chunk evidence. Include:
            - key facts with dates/numbers where present
            - any limitations or bias in the source
            - useful exact wording from quotes if it is short

            Keep the note under 180 words and refer to the source as [{source_id}].
            """
        ).strip()
        return self.venice.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )

    def _initial_queries(self, topic: str, count: int) -> list[str]:
        prompt = dedent(
            f"""
            Create {count} diverse web search queries for researching this topic:
            {topic}

            Cover background, recent developments, primary sources, criticism, and data.
            Include at least one query likely to find primary sources or datasets.
            Return JSON only in this shape: {{"queries": ["..."]}}
            """
        ).strip()
        return self._query_list(prompt, count, fallback=[topic])

    def _follow_up_queries(
        self, topic: str, notes: list[SourceNote], count: int
    ) -> list[str]:
        digest = _source_digest(notes, max_chars=9000)
        prompt = dedent(
            f"""
            We are researching: {topic}

            Current notes:
            {digest}

            Create {count} follow-up web search queries that fill gaps, verify important claims,
            find primary evidence, and look for dissenting evidence.
            Return JSON only in this shape: {{"queries": ["..."]}}
            """
        ).strip()
        return self._query_list(prompt, count, fallback=[topic])

    def _write_report(self, topic: str, notes: list[SourceNote]) -> str:
        if not notes:
            return (
                f"# Research report: {topic}\n\n"
                "No usable web sources were collected. Check your network connection or try a narrower topic."
            )

        if self.report_style == "deep":
            return self._write_staged_deep_report(topic, notes)

        prompt = _report_prompt(topic, notes, self.report_style)
        return self._chat_report(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=REPORT_TOKEN_BUDGETS[self.report_style],
        )

    def _write_staged_deep_report(
        self, topic: str, notes: list[SourceNote]
    ) -> str:
        self.progress("Planning deep report outline")
        outline = self._create_report_outline(topic, notes)
        self.artifacts.write("report_outline", {"topic": topic, "outline": outline})

        section_drafts: list[dict[str, object]] = []
        sections = outline["sections"]
        for index, section in enumerate(sections, start=1):
            heading = str(section["heading"])
            self.progress(f"Drafting report section {index}/{len(sections)}: {heading}")
            draft = self._write_report_section(topic, notes, outline, section)
            record = {
                "index": index,
                "heading": heading,
                "source_ids": section["source_ids"],
                "markdown": draft,
            }
            self.artifacts.write("report_sections", record)
            section_drafts.append(record)

        self.progress("Editing staged deep report")
        report = self._edit_staged_deep_report(topic, notes, outline, section_drafts)
        self.artifacts.write(
            "report_editor",
            {
                "topic": topic,
                "title": outline["title"],
                "section_count": len(section_drafts),
                "markdown": report,
            },
        )
        return report

    def _create_report_outline(
        self, topic: str, notes: list[SourceNote]
    ) -> dict[str, object]:
        prompt = _report_outline_prompt(topic, notes)
        response = self._chat_report(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=REPORT_OUTLINE_TOKEN_BUDGET,
        )
        try:
            data = _loads_json(response)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        return _normalize_report_outline(data, topic, notes)

    def _write_report_section(
        self,
        topic: str,
        notes: list[SourceNote],
        outline: dict[str, object],
        section: dict[str, object],
    ) -> str:
        source_ids = [
            source_id
            for source_id in section.get("source_ids", [])
            if isinstance(source_id, str)
        ]
        section_notes = _notes_for_source_ids(notes, source_ids) or notes
        prompt = _report_section_prompt(topic, notes, section_notes, outline, section)
        return self._chat_report(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            max_tokens=REPORT_SECTION_TOKEN_BUDGET,
        ).strip()

    def _edit_staged_deep_report(
        self,
        topic: str,
        notes: list[SourceNote],
        outline: dict[str, object],
        section_drafts: list[dict[str, object]],
    ) -> str:
        prompt = _report_editor_prompt(topic, notes, outline, section_drafts)
        return self._chat_report(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=REPORT_EDITOR_TOKEN_BUDGET,
        ).strip()

    def _chat_report(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        chat_stream = getattr(self.venice, "chat_stream", None)
        if callable(chat_stream):
            return chat_stream(
                messages, temperature=temperature, max_tokens=max_tokens
            )
        return self.venice.chat(
            messages, temperature=temperature, max_tokens=max_tokens
        )

    def _query_list(self, prompt: str, count: int, fallback: list[str]) -> list[str]:
        response = self.venice.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=500,
        )
        try:
            data = _loads_json(response)
            queries = data.get("queries", [])
        except (json.JSONDecodeError, AttributeError):
            queries = []

        clean_queries = [
            query.strip()
            for query in queries
            if isinstance(query, str) and query.strip()
        ]
        return (clean_queries or fallback)[:count]

    def _record_error(
        self,
        stage: str,
        exc: Exception,
        *,
        query: str = "",
        url: str = "",
        source_id: str = "",
        provider: str = "",
    ) -> None:
        message = str(exc)
        self.progress(
            f"{stage.replace('_', ' ').title()} failed{f' for {url}' if url else ''}: {message}"
        )
        self.artifacts.write(
            "errors",
            CollectionError(
                stage=stage,
                message=message,
                query=query,
                url=url,
                source_id=source_id,
                provider=provider,
            ),
        )

    def _source_budget_reached(self, notes: list[SourceNote]) -> bool:
        return self.max_sources is not None and len(notes) >= self.max_sources


def _source_digest(
    notes: list[SourceNote], max_chars: int, chunk_chars: int = 2000
) -> str:
    chunks = [
        "\n".join(
            [
                f"[{note.source_id}] {note.title}",
                f"URL: {note.final_url or note.url}",
                f"Canonical URL: {note.canonical_url}",
                f"Found via: {note.query}",
                f"Provider/rank: {note.provider}/{note.rank}",
                f"Retrieved: {note.retrieved_at}",
                f"Content hash: {note.content_hash}",
                f"Note: {note.summary}",
                f"Chunk evidence: {_chunk_digest(note.chunks, max_chars=chunk_chars)}",
            ]
        )
        for note in notes
    ]
    digest = "\n\n".join(chunks)
    return digest[:max_chars]


def _source_index(notes: list[SourceNote]) -> str:
    return "\n".join(
        f"[{note.source_id}] {note.title} - {note.final_url or note.url}"
        for note in notes
    )


def _notes_for_source_ids(
    notes: list[SourceNote], source_ids: list[str]
) -> list[SourceNote]:
    wanted = set(source_ids)
    return [note for note in notes if note.source_id in wanted]


def _report_outline_prompt(topic: str, notes: list[SourceNote]) -> str:
    source_digest = _source_digest(
        notes,
        max_chars=REPORT_SOURCE_DIGEST_CHAR_LIMITS["deep"],
        chunk_chars=REPORT_CHUNK_DIGEST_CHAR_LIMITS["deep"],
    )
    return dedent(
        f"""
        Plan a staged deep research report.

        Research topic:
        {topic}

        Source notes:
        {source_digest}

        Report style: deep

        Requirements:
        - Return JSON only.
        - Plan a useful long-form report, not a compressed answer.
        - Choose 5-8 body sections. Do not include Overview, Finishing Up, References, or methodology sections in the section list.
        - Each section must have a distinct synthesis job so section drafts do not overlap.
        - Assign relevant source IDs to each section. Use source IDs exactly as supplied, such as S1.
        - Identify likely tables where comparison, timelines, frameworks, decision criteria, or tradeoffs would make the report more useful.
        - Prioritize concrete synthesis: mechanisms, examples, named entities, patterns, tradeoffs, and practical implications.

        JSON shape:
        {{
          "title": "specific report title",
          "thesis": "central synthesis in 1-2 sentences",
          "sections": [
            {{
              "heading": "topic-specific section heading",
              "purpose": "what this section should explain or decide",
              "questions": ["question this section answers"],
              "source_ids": ["S1", "S2"],
              "expected_tables": ["table idea if useful"]
            }}
          ]
        }}
        """
    ).strip()


def _report_section_prompt(
    topic: str,
    all_notes: list[SourceNote],
    section_notes: list[SourceNote],
    outline: dict[str, object],
    section: dict[str, object],
) -> str:
    section_digest = _source_digest(section_notes, max_chars=30000, chunk_chars=1100)
    return dedent(
        f"""
        Draft one deep report section.

        Research topic:
        {topic}

        Full source index:
        {_source_index(all_notes)}

        Overall report outline:
        {json.dumps(outline, ensure_ascii=False)}

        Section plan:
        {json.dumps(section, ensure_ascii=False)}

        Detailed source notes for this section:
        {section_digest}

        Requirements:
        - Write only this section, starting with the planned "##" heading.
        - Use a thoughtful long-form technical blog voice: clear, practical, and explanatory.
        - Develop the section with multiple paragraphs before using bullets or tables.
        - Make the section useful: explain mechanisms, compare alternatives, name concrete examples, include decision criteria where relevant, and spell out practical implications.
        - Include a Markdown table if the section plan calls for one or if comparison would make the synthesis clearer.
        - Use internal source citations like [S1] and [S2] for factual claims. The final editor will convert them to footnote-style citations.
        - Do not write the report overview, Finishing Up, References, or a source-by-source note list.
        - Aim for 700-1,300 words when the evidence supports it.
        """
    ).strip()


def _report_editor_prompt(
    topic: str,
    notes: list[SourceNote],
    outline: dict[str, object],
    section_drafts: list[dict[str, object]],
) -> str:
    drafts = "\n\n".join(
        dedent(
            f"""
            Section {draft["index"]}: {draft["heading"]}
            {draft["markdown"]}
            """
        ).strip()
        for draft in section_drafts
    )
    source_reference_digest = _source_digest(notes, max_chars=60000, chunk_chars=250)
    return dedent(
        f"""
        Assemble the final deep research report from staged section drafts.

        Research topic:
        {topic}

        Report outline:
        {json.dumps(outline, ensure_ascii=False)}

        Source index:
        {_source_index(notes)}

        Source reference details:
        {source_reference_digest}

        Section drafts:
        {drafts}

        Requirements:
        - Produce one coherent Markdown report.
        - Start with the outline title as an H1.
        - Add a developed "## Overview" before the staged sections. It should synthesize the thesis, explain why the topic matters, and preview the concrete takeaways.
        - Preserve the useful detail from the section drafts. Do not compress the report into a short summary.
        - Remove duplicated introductions, repeated claims, and section-to-section seams.
        - Keep or improve useful tables from the drafts.
        - Convert internal source citations like [S1] into Perplexity-like footnote markers like [^1]. Reuse the same marker every time the same source is cited.
        - Do not leave source-ID citations like [S1] in the final report body.
        - Do not include uncited factual claims.
        - Add "## Finishing Up" before references with practical takeaways and what the reader should do with the findings.
        - End with "## References" as a numbered Markdown list ordered by first citation. Each reference must include title, URL, and a short description.
        - Do not include placeholder images.
        """
    ).strip()


def _normalize_report_outline(
    data: dict[str, object], topic: str, notes: list[SourceNote]
) -> dict[str, object]:
    fallback = _fallback_report_outline(topic, notes)
    valid_source_ids = {note.source_id for note in notes}
    all_source_ids = [note.source_id for note in notes]

    title = _clean_string(data.get("title")) or str(fallback["title"])
    thesis = _clean_string(data.get("thesis")) or str(fallback["thesis"])
    raw_sections = data.get("sections")
    sections: list[dict[str, object]] = []

    if isinstance(raw_sections, list):
        for raw_section in raw_sections:
            if not isinstance(raw_section, dict):
                continue
            heading = _clean_string(raw_section.get("heading"))
            if not heading or _is_boilerplate_report_section(heading):
                continue
            source_ids = [
                source_id
                for source_id in _clean_string_list(raw_section.get("source_ids"))
                if source_id in valid_source_ids
            ]
            sections.append(
                {
                    "heading": heading,
                    "purpose": _clean_string(raw_section.get("purpose")),
                    "questions": _clean_string_list(raw_section.get("questions")),
                    "source_ids": source_ids or all_source_ids,
                    "expected_tables": _clean_string_list(
                        raw_section.get("expected_tables")
                    )
                    or _clean_string_list(raw_section.get("tables")),
                }
            )
            if len(sections) >= MAX_DEEP_REPORT_SECTIONS:
                break

    if not sections:
        sections = fallback["sections"]  # type: ignore[assignment]

    return {"title": title, "thesis": thesis, "sections": sections}


def _fallback_report_outline(
    topic: str, notes: list[SourceNote]
) -> dict[str, object]:
    source_ids = [note.source_id for note in notes]
    return {
        "title": topic.strip().title() if topic.strip() else "Deep Research Report",
        "thesis": (
            "The report should synthesize the collected source evidence into concrete "
            "patterns, tradeoffs, examples, and practical takeaways."
        ),
        "sections": [
            {
                "heading": "Core Concepts and Current Landscape",
                "purpose": "Define the topic and explain the current state using concrete source evidence.",
                "questions": ["What is this topic, and what has changed recently?"],
                "source_ids": source_ids,
                "expected_tables": [],
            },
            {
                "heading": "Major Patterns and Tradeoffs",
                "purpose": "Compare the main approaches, categories, or design choices in the evidence.",
                "questions": ["Which patterns matter most, and how do they compare?"],
                "source_ids": source_ids,
                "expected_tables": ["Comparison of major patterns, options, or tradeoffs"],
            },
            {
                "heading": "Concrete Examples and Use Cases",
                "purpose": "Ground the synthesis in named examples, implementations, or real-world applications.",
                "questions": ["Where does this show up in practice?"],
                "source_ids": source_ids,
                "expected_tables": [],
            },
            {
                "heading": "What This Means in Practice",
                "purpose": "Translate the research into decision criteria, adoption guidance, and risk considerations.",
                "questions": ["What should a reader do with these findings?"],
                "source_ids": source_ids,
                "expected_tables": ["Decision criteria and practical implications"],
            },
            {
                "heading": "Limitations and Open Questions",
                "purpose": "Explain source limitations, unresolved disagreements, and areas needing further evidence.",
                "questions": ["Where is the evidence uncertain or incomplete?"],
                "source_ids": source_ids,
                "expected_tables": [],
            },
        ],
    }


def _clean_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clean_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _is_boilerplate_report_section(heading: str) -> bool:
    normalized = heading.lower().strip("# ")
    return normalized in {
        "overview",
        "executive summary",
        "research method",
        "research method and limits",
        "source landscape",
        "source landscape and quality assessment",
        "source-by-source notes",
        "sources",
        "references",
        "finishing up",
    }


def _report_prompt(topic: str, notes: list[SourceNote], style: ReportStyle) -> str:
    style_instructions = {
        "brief": dedent(
            """
            Write a concise Perplexity-style research report.
            Layout:
            - Start with a precise H1 title derived from the topic, not "Research report".
            - Open with "## Overview": 1-2 dense paragraphs that directly answer the topic.
            - Add 3-5 topic-specific "##" sections with short, descriptive headings.
            - Use bullets only for compact lists of components, tradeoffs, or steps.
            - End with "## References".
            """
        ).strip(),
        "standard": dedent(
            """
            Write a detailed Perplexity-style research report with the readability of a strong technical blog post.
            Layout:
            - Start with a precise H1 title derived from the topic, not "Research report".
            - Open with "## Overview": 3-4 polished paragraphs that set context, explain why the topic matters, and synthesize the answer.
            - Insert "***" between major sections.
            - Use topic-specific "##" sections and occasional "###" subsections rather than generic report scaffolding.
            - Lead each major section with a short narrative paragraph before using bullets or tables.
            - Include concrete examples, named entities, decision criteria, and tradeoffs wherever the source material supports them.
            - Include a comparison table when the evidence supports comparing tools, options, categories, timelines, or tradeoffs.
            - Include a practical "how to think about this" section when the topic involves choices, adoption, implementation, or strategy.
            - Weave uncertainty and disagreement into the relevant topical section, or add a short "## Limitations and Open Questions" section if needed.
            - Close with "## Finishing Up": 2-3 paragraphs that summarize the practical takeaway without adding new claims.
            - End with "## References".
            """
        ).strip(),
        "deep": dedent(
            """
            Write a comprehensive Perplexity-style deep research report that reads like a thoughtful long-form technical blog post.
            Layout:
            - Start with a precise H1 title derived from the topic, not "Research report".
            - Open with "## Overview": 4-6 substantial paragraphs that define the subject, explain why it matters now, state the main conclusion, and preview the most important dimensions.
            - Insert "***" between major sections.
            - Build the body around topic-specific "##" sections and "###" subsections, as if writing an explanatory reference article.
            - Make the report feel written for an interested human reader: use natural transitions, define terms before leaning on them, and explain why each section matters.
            - Prefer synthesized analysis over source-by-source narration. Do not include default sections named "Executive Summary", "Research Method", "Source Landscape", or "Source-by-Source Notes".
            - Do not over-compress. For important sections, use multiple developed paragraphs before switching to bullets or tables.
            - Make each major section useful: explain the concrete mechanism, give examples or named cases, compare alternatives, and spell out the practical implication.
            - Use comparison tables where they clarify patterns, categories, timelines, frameworks, benchmarks, or tradeoffs. A deep report should usually include multiple tables when the source material supports comparison.
            - Use bullets and numbered lists for concrete components, workflows, decision rules, or ranked considerations; keep most analysis in paragraphs.
            - Include a practical synthesis section, such as "## What This Means in Practice", "## Choosing an Approach", or another topic-specific equivalent.
            - Include decision criteria, adoption guidance, implementation considerations, or risk tradeoffs when the topic has practical consequences.
            - Add a short limitations/open questions section only when the evidence has meaningful gaps, disagreement, or uncertainty.
            - Close with "## Finishing Up": 3-5 paragraphs that bring the argument together, explain what a reader should do with the findings, and avoid introducing unsupported new facts.
            - End with "## References".

            Synthesize across the full source set. When 40+ usable sources are available,
            cite broadly instead of relying on a small subset.
            Aim for a generous long-form treatment, roughly 4,000-6,000 words when the
            source material supports it.
            """
        ).strip(),
    }[style]
    source_digest = _source_digest(
        notes,
        max_chars=REPORT_SOURCE_DIGEST_CHAR_LIMITS[style],
        chunk_chars=REPORT_CHUNK_DIGEST_CHAR_LIMITS[style],
    )

    return dedent(
        f"""
        Research topic:
        {topic}

        Source notes:
        {source_digest}

        Report style: {style}

        Requirements:
        - Use Markdown.
        - {style_instructions}
        - Use only the supplied source notes and chunk evidence for factual claims.
        - Use Perplexity-like citation markers in prose: [^1], [^2], and so on. Reuse the same marker every time you cite the same source.
        - Do not cite with source IDs like [S1] in the report body; treat source IDs as internal provenance only.
        - Do not include uncited factual claims.
        - Keep citation markers tight to the claim they support, often at the end of the sentence or paragraph.
        - Write in a clear, conversational expert voice. Prefer plain language, useful context, and smooth transitions over terse bullet summaries.
        - Paragraphs should usually be 3-6 sentences. Avoid one-sentence sections unless the section is intentionally short.
        - Optimize for usefulness and concrete synthesis. Do not merely summarize sources; combine them into takeaways, distinctions, patterns, and tradeoffs.
        - Avoid generic filler phrases such as "is critical", "is important", or "has significant implications" unless you immediately explain the specific reason.
        - For every major finding, answer at least two of: what changed, why it matters, who it affects, what tradeoff it creates, how it compares to alternatives, or what a reader should do next.
        - Flag uncertainty, source limitations, and contradictions in the relevant section instead of creating boilerplate methodology sections.
        - Do not include placeholder images.
        - End with a "References" section formatted as a numbered Markdown list. Each item must include the source title, URL, and a short description. Order references by first citation.
        """
    ).strip()


def _chunk_digest(chunks: tuple[EvidenceChunk, ...], max_chars: int) -> str:
    digest = "\n\n".join(
        "\n".join(
            [
                f"{chunk.chunk_id}: {chunk.summary}",
                f"Quotes: {' | '.join(chunk.quotes) if chunk.quotes else 'None'}",
            ]
        )
        for chunk in chunks
    )
    return digest[:max_chars]


def _parse_evidence_chunk(chunk_id: str, text: str, response: str) -> EvidenceChunk:
    try:
        data = _loads_json(response)
    except json.JSONDecodeError:
        return EvidenceChunk(
            chunk_id=chunk_id, text=text, summary=response.strip(), quotes=()
        )

    summary = data.get("summary", "")
    quotes = data.get("quotes", [])
    clean_quotes = tuple(
        quote.strip() for quote in quotes if isinstance(quote, str) and quote.strip()
    )
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text,
        summary=summary.strip()
        if isinstance(summary, str) and summary.strip()
        else response.strip(),
        quotes=clean_quotes,
    )


def _loads_json(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]

    return json.loads(cleaned)
