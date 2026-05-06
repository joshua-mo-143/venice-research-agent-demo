# Venice Research Agent Demo

A minimal Python demo of a deep research agent that uses Venice AI's OpenAI-compatible API for LLM planning, source note-taking, follow-up query generation, and final synthesis.

The agent:

1. asks Venice to plan diverse search queries for a topic,
2. searches the web with one or more source providers,
3. uses Venice's scrape endpoint to turn source pages into Markdown,
4. chunks the scraped content and asks Venice to extract evidence,
5. asks Venice for follow-up searches to fill gaps,
6. writes a cited Markdown research briefing.

## Setup

This project targets Python 3.13.

```bash
uv sync
cp .env.example .env
```

Add your Venice API key to `.env`:

```bash
VENICE_API_KEY=your_venice_api_key_here
VENICE_MODEL=venice-uncensored
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

- `brief`: concise briefing with summary, findings, uncertainties, implications, and sources.
- `standard`: fuller report with method, source landscape, evidence, disagreements, and implications.
- `deep`: comprehensive report with detailed findings, source quality assessment, open questions, and source-by-source notes.

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

## How research works

The research process is coordinated by `ResearchAgent` in `research_agent/agent.py`.

1. **Plan initial searches**: Venice receives the topic and returns a JSON list of diverse search queries. The prompt asks for background, recent developments, primary sources, criticism, and data.
2. **Search providers**: `WebSearch` sends each query to configured providers. The default is DuckDuckGo; `arxiv` can be added for papers with `--providers duckduckgo,arxiv`.
3. **Deduplicate results**: URLs are canonicalized before reading. Tracking parameters and fragments are removed, redirects are checked after scraping, and duplicate content hashes are skipped.
4. **Scrape pages with Venice**: for each new result, `VeniceClient.scrape()` calls Venice's `POST /augment/scrape` endpoint with the source URL. Venice returns the page content as Markdown.
5. **Chunk source text**: scraped Markdown is split into overlapping chunks so long sources are not reduced to a single truncated excerpt.
6. **Extract chunk evidence**: Venice summarizes each chunk and returns short supporting quotes where useful. These become chunk-level evidence records.
7. **Summarize each source**: Venice combines the chunk evidence into a concise source note with a source ID like `[S1]`.
8. **Generate follow-up queries**: after each pass except the last, Venice reviews the current source notes and proposes new searches to fill gaps, verify claims, and find dissenting evidence.
9. **Write the report**: Venice receives the source notes and writes a Markdown briefing. The report is instructed to cite factual claims with source IDs and include a `Sources` section.

Use `--report-style standard` or `--report-style deep` when you want the final synthesis to spend more tokens on methodology, evidence, uncertainty, and source-by-source analysis.

## Data collection

The collection layer now keeps richer provenance for each source: canonical URL, final redirected URL, search query, provider, rank, snippet, retrieval time, content type, content hash, extracted chunks, chunk summaries, and supporting quotes. Page fetching uses Venice's `/augment/scrape` endpoint, which returns source content as Markdown before chunking.

When `--artifacts` is set, the agent writes JSONL files for queries, search results, fetch metadata, extracted chunks, chunk summaries, source notes, dedupe decisions, errors, and the final report.

Available providers:

- `duckduckgo`: general web search via DuckDuckGo's HTML endpoint.
- `arxiv`: paper discovery via arXiv's Atom API.

## Test

```bash
uv run python -m unittest discover -s tests
```

## Notes

- The web search layer is intentionally lightweight for demo purposes.
- The model is instructed to cite source IDs like `[S1]` and to avoid uncited factual claims.
- The agent continues past individual search or fetch failures because web pages are often blocked, moved, or non-HTML.
