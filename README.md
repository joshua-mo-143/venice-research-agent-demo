# Venice Research Agent Demo

A small Python research agent that uses Venice AI to turn a topic into a cited Markdown briefing.

The project is intentionally compact, but it models the main stages of a deep research workflow: query planning, web discovery, source reading, evidence extraction, gap analysis, and final synthesis. Venice does the language-model work through its OpenAI-compatible chat endpoint and reads web pages through its `/augment/scrape` endpoint. The local Python code coordinates the loop, tracks provenance, deduplicates sources, chunks long pages, and writes optional JSONL artifacts so a run can be inspected later.

## How the Agent Works

At a high level, `main.py` wires together three pieces:

- `VeniceClient` in `research_agent/venice.py` handles chat completions, streamed report generation, retries, and page scraping through Venice.
- `WebSearch` in `research_agent/web.py` finds candidate sources through search providers, currently DuckDuckGo and arXiv.
- `ResearchAgent` in `research_agent/agent.py` runs the research loop and turns collected evidence into a report.

For each topic, the agent follows this flow:

1. **Plan searches**: Venice receives the topic and returns diverse search queries covering background, recent developments, primary sources, criticism, and data.
2. **Discover sources**: each query runs against the configured providers. DuckDuckGo is the default; arXiv can be added for papers.
3. **Read pages**: each result is passed to Venice's scrape endpoint, which returns Markdown content plus page metadata.
4. **Normalize and deduplicate**: URLs are canonicalized, tracking parameters are removed, redirects are checked, and repeated content hashes are skipped.
5. **Chunk long sources**: scraped Markdown is split into overlapping chunks so useful evidence is not lost when a source is too long for one model call.
6. **Extract evidence**: Venice summarizes each chunk and returns short supporting quotes when available.
7. **Create source notes**: chunk evidence is compressed into concise notes with stable source IDs such as `[S1]`.
8. **Find gaps**: between research passes, Venice reviews the source notes, source balance, and missing coverage, then proposes targeted follow-up queries.
9. **Write the report**: the final source notes are synthesized into Markdown with footnote-style citations and a numbered `References` section.

The default `deep` report style uses a staged writer: Venice first plans an outline, drafts each body section against relevant source notes, then edits the sections into one coherent report. The `brief` and `standard` styles use a single report prompt with smaller token budgets.

## What Gets Tracked

Each accepted source keeps enough provenance to audit the final report:

- search query, provider, rank, title, snippet, and retrieval time
- original URL, final redirected URL, and canonical URL
- content type, content hash, chunk ranges, chunk hashes, chunk summaries, and supporting quotes
- source-level summary and source ID used during synthesis

When `--artifacts` is set, those records are written as JSONL files, including queries, research gaps, search results, fetch metadata, extracted chunks, chunk summaries, source notes, dedupe decisions, errors, report outlines, report sections, and the final report.

The agent is designed to keep moving when individual searches, fetches, or chunk summaries fail. Web research often hits blocked pages, redirects, duplicate articles, or malformed responses, so failures are recorded as artifacts instead of stopping the whole run.

## Setup

This project targets Python 3.13.

```bash
uv sync
cp .env.example .env
```

Add your Venice API key to `.env`:

```bash
VENICE_API_KEY=your_venice_api_key_here
VENICE_MODEL=openai-gpt-55
```

Venice exposes an OpenAI-compatible chat completions endpoint at `https://api.venice.ai/api/v1/chat/completions`. You can change `VENICE_MODEL` to any chat model available to your Venice account.

## Run

Run the agent with `uv` by passing a research topic to `main.py`:

```bash
uv run python main.py "How are AI agents changing software engineering workflows?"
```

The topic should usually be wrapped in quotes so your shell passes it as one argument.

Write the report to a file:

```bash
uv run python main.py "state of open source LLM inference in 2026" --markdown-output reports/inference.md
```

`--output` is also supported as a shorter alias for `--markdown-output`.

Tune the research depth:

```bash
uv run python main.py "Venice AI API developer ecosystem" --iterations 3 --queries 5 --results 4
```

These options mean:

- `--iterations`: how many research passes to run.
- `--queries`: how many search queries Venice should generate per pass.
- `--results`: how many results to collect per provider for each query.

Choose how detailed the final report should be:

```bash
uv run python main.py "AI agents in software engineering" --report-style deep
```

Report styles:

- `brief`: concise source-backed briefing with an overview, a few topical sections, and references.
- `standard`: fuller research survey with context, evidence, disagreements, implications, tables when useful, and a closing synthesis.
- `deep`: staged long-form report with an outline, section drafts, final editing pass, broader source coverage, and practical takeaways.

Save auditable research artifacts:

```bash
uv run python main.py "open source LLM evaluation" --artifacts runs/llm-eval
```

Use multiple source providers and collection controls:

```bash
uv run python main.py "agentic coding research" \
  --providers duckduckgo,arxiv \
  --max-sources 12 \
  --chunk-chars 3000 \
  --max-chunks-per-source 4 \
  --request-delay 1
```

If you install the project, the script entry point is also available:

```bash
uv run venice-research "privacy tradeoffs in hosted LLM APIs"
```

## Source Providers

Available providers:

- `duckduckgo`: general web search via DuckDuckGo's HTML endpoint.
- `arxiv`: paper discovery via arXiv's Atom API.

## Test

```bash
uv run python -m unittest discover -s tests
```

## Notes

- The web search layer is intentionally lightweight for demo purposes.
- Source IDs like `[S1]` are internal provenance markers. Final reports are prompted to use footnote-style citations like `[^1]`.
- The agent continues past individual search or fetch failures because web pages are often blocked, moved, or non-HTML.
