from __future__ import annotations

import json
from collections.abc import Callable
from textwrap import dedent
from typing import Literal

from .artifacts import ArtifactWriter
from .models import CollectionError, EvidenceChunk, ResearchReport, SearchResult, SourceNote, WebPage, utc_now
from .venice import VeniceClient, VeniceError
from .web import WebSearch


SYSTEM_PROMPT = """You are a careful research assistant.
Use the supplied source material only when making factual claims.
Flag uncertainty, contradictions, and missing context instead of filling gaps."""


ProgressCallback = Callable[[str], None]
ReportStyle = Literal["brief", "standard", "deep"]
REPORT_TOKEN_BUDGETS: dict[ReportStyle, int] = {
    "brief": 2400,
    "standard": 4000,
    "deep": 6500,
}


class ResearchAgent:
    def __init__(
        self,
        venice: VeniceClient,
        web: WebSearch | None = None,
        artifacts: ArtifactWriter | None = None,
        progress: ProgressCallback | None = None,
        max_sources: int | None = None,
        max_chunks_per_source: int = 4,
        report_style: ReportStyle = "brief",
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
        iterations: int = 2,
        query_count: int = 4,
        results_per_query: int = 3,
    ) -> ResearchReport:
        notes: list[SourceNote] = []
        seen_source_keys: set[str] = set()
        seen_content_hashes: set[str] = set()
        queries = self._initial_queries(topic, query_count)
        self.artifacts.write("queries", {"stage": "initial", "topic": topic, "queries": queries})

        for iteration in range(1, iterations + 1):
            self.progress(f"Research pass {iteration}/{iterations}: {', '.join(queries)}")
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
                    {"stage": "follow_up", "topic": topic, "iteration": iteration + 1, "queries": queries},
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
        artifacts_dir = str(self.artifacts.root) if self.artifacts.root is not None else None
        return ResearchReport(topic=topic, markdown=report, sources=notes, artifacts_dir=artifacts_dir)

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

        if page.canonical_url in seen_source_keys and page.canonical_url != result.canonical_url:
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
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=600,
                )
                evidence_chunk = _parse_evidence_chunk(chunk.chunk_id, chunk.text, response)
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
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
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

    def _follow_up_queries(self, topic: str, notes: list[SourceNote], count: int) -> list[str]:
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

        prompt = _report_prompt(topic, notes, self.report_style)
        return self.venice.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=REPORT_TOKEN_BUDGETS[self.report_style],
        )

    def _query_list(self, prompt: str, count: int, fallback: list[str]) -> list[str]:
        response = self.venice.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500,
        )
        try:
            data = _loads_json(response)
            queries = data.get("queries", [])
        except (json.JSONDecodeError, AttributeError):
            queries = []

        clean_queries = [query.strip() for query in queries if isinstance(query, str) and query.strip()]
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
        self.progress(f"{stage.replace('_', ' ').title()} failed{f' for {url}' if url else ''}: {message}")
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


def _source_digest(notes: list[SourceNote], max_chars: int) -> str:
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
                f"Chunk evidence: {_chunk_digest(note.chunks, max_chars=2000)}",
            ]
        )
        for note in notes
    ]
    digest = "\n\n".join(chunks)
    return digest[:max_chars]


def _report_prompt(topic: str, notes: list[SourceNote], style: ReportStyle) -> str:
    style_instructions = {
        "brief": dedent(
            """
            Write a concise deep research briefing.
            Structure:
            - short executive summary
            - important findings
            - disagreements/uncertainties
            - practical implications
            - Sources
            """
        ).strip(),
        "standard": dedent(
            """
            Write a detailed research report.
            Structure:
            - executive summary
            - research method
            - source landscape
            - key findings
            - evidence and analysis
            - disagreements and uncertainties
            - practical implications
            - Sources
            """
        ).strip(),
        "deep": dedent(
            """
            Write a comprehensive deep research report.
            Structure:
            - executive summary
            - research method and limits
            - source landscape and quality assessment
            - detailed findings with evidence
            - timeline or current-state analysis where useful
            - disagreements, uncertainties, and missing evidence
            - practical implications
            - open questions for further research
            - source-by-source notes
            - Sources
            """
        ).strip(),
    }[style]

    return dedent(
        f"""
        Research topic:
        {topic}

        Source notes:
        {_source_digest(notes, max_chars=24000 if style == "deep" else 18000)}

        Report style: {style}

        Requirements:
        - Use Markdown.
        - {style_instructions}
        - Use only the supplied source notes and chunk evidence for factual claims.
        - Cite claims with source IDs like [S1]. Do not include uncited factual claims.
        - Flag uncertainty, source limitations, and contradictions.
        - End with a "Sources" section listing each source ID, title, and URL.
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
        return EvidenceChunk(chunk_id=chunk_id, text=text, summary=response.strip(), quotes=())

    summary = data.get("summary", "")
    quotes = data.get("quotes", [])
    clean_quotes = tuple(quote.strip() for quote in quotes if isinstance(quote, str) and quote.strip())
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text,
        summary=summary.strip() if isinstance(summary, str) and summary.strip() else response.strip(),
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
