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

The topic should usually be wrapped in quotes so your shell passes it as one argument. By default,
the agent now runs a deep report with 3 research passes, 6 queries per pass, 4 results per query,
up to 48 usable sources, and up to 6 summarized chunks per source.

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

Choose a different report depth:

```bash
uv run python main.py "AI agents in software engineering" --report-style standard
```

Report styles:

- `brief`: concise Perplexity-style briefing with an overview, topical sections, and references.
- `standard`: fuller Perplexity-style report with a natural technical-blog voice, concrete synthesis, useful tables, uncertainty, and practical takeaways.
- `deep`: comprehensive long-form report with technical-blog pacing, broad citations, developed sections, concrete comparisons, decision criteria, tables where useful, and limitations or open questions when warranted.

Save auditable research artifacts:

```bash
uv run python main.py "open source LLM evaluation" --artifacts runs/llm-eval
```

Use multiple source providers and collection controls:

```bash
uv run python main.py "agentic coding research" \
  --providers duckduckgo,arxiv \
  --max-sources 48 \
  --chunk-chars 3000 \
  --max-chunks-per-source 6 \
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
9. **Write the report**: Venice receives the source notes and writes a Perplexity-style Markdown report. Deep reports use a staged writer that plans an outline, drafts major sections separately, and runs a final editor pass. Report-writing calls use streaming responses and aggregate the chunks into the final Markdown, which helps avoid long read timeouts. The report is instructed to cite factual claims with footnote-style markers and include a `References` section.

Use `--report-style brief` or `--report-style standard` when you want a shorter final synthesis. The
default `deep` style spends more tokens on long-form topical analysis, evidence, uncertainty, tables,
practical takeaways, tradeoffs, decision guidance, and broad references.

## Data collection

The collection layer now keeps richer provenance for each source: canonical URL, final redirected URL, search query, provider, rank, snippet, retrieval time, content type, content hash, extracted chunks, chunk summaries, and supporting quotes. Page fetching uses Venice's `/augment/scrape` endpoint, which returns source content as Markdown before chunking.

When `--artifacts` is set, the agent writes JSONL files for queries, search results, fetch metadata, extracted chunks, chunk summaries, source notes, dedupe decisions, errors, staged report outlines and section drafts, editor output, and the final report.

Available providers:

- `duckduckgo`: general web search via DuckDuckGo's HTML endpoint.
- `arxiv`: paper discovery via arXiv's Atom API.

## Test

```bash
uv run python -m unittest discover -s tests
```

## Notes

- The web search layer is intentionally lightweight for demo purposes.
- The model is instructed to cite factual claims with Perplexity-like markers such as `[^1]` and to avoid uncited factual claims.
- The agent continues past individual search or fetch failures because web pages are often blocked, moved, or non-HTML.
