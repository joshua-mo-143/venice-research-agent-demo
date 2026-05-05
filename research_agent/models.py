from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "msclkid",
    "ref",
    "ref_src",
}


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    query: str = ""
    rank: int = 0
    provider: str = "duckduckgo"
    canonical_url: str = ""

    def __post_init__(self) -> None:
        if not self.canonical_url:
            object.__setattr__(self, "canonical_url", canonicalize_url(self.url))


@dataclass(frozen=True)
class ScrapeResult:
    url: str
    content: str
    title: str = ""
    final_url: str = ""
    content_type: str = "text/markdown"


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    start: int
    end: int
    content_hash: str


@dataclass(frozen=True)
class WebPage:
    title: str
    url: str
    text: str
    final_url: str = ""
    canonical_url: str = ""
    content_type: str = ""
    retrieved_at: str = ""
    content_hash: str = ""
    chunks: tuple[TextChunk, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        final_url = self.final_url or self.url
        object.__setattr__(self, "final_url", final_url)
        if not self.canonical_url:
            object.__setattr__(self, "canonical_url", canonicalize_url(final_url))
        if not self.retrieved_at:
            object.__setattr__(self, "retrieved_at", utc_now())
        if not self.content_hash:
            object.__setattr__(self, "content_hash", content_hash(self.text))


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    text: str
    summary: str
    quotes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SourceNote:
    source_id: str
    title: str
    url: str
    query: str
    summary: str
    canonical_url: str = ""
    final_url: str = ""
    rank: int = 0
    snippet: str = ""
    provider: str = "duckduckgo"
    retrieved_at: str = ""
    content_type: str = ""
    content_hash: str = ""
    chunks: tuple[EvidenceChunk, ...] = field(default_factory=tuple)
    status: str = "summarized"


@dataclass(frozen=True)
class CollectionError:
    stage: str
    message: str
    query: str = ""
    url: str = ""
    source_id: str = ""
    provider: str = ""


@dataclass(frozen=True)
class ResearchReport:
    topic: str
    markdown: str
    sources: list[SourceNote]
    artifacts_dir: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_url(raw_url: str) -> str:
    if not raw_url:
        return ""

    parsed = urlparse(raw_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_tracking_param(key)
    ]
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def chunk_text(text: str, *, chunk_chars: int = 3000, overlap: int = 250) -> tuple[TextChunk, ...]:
    clean = text.strip()
    if not clean:
        return ()
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be greater than 0")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap must be at least 0 and smaller than chunk_chars")

    chunks: list[TextChunk] = []
    start = 0
    index = 1
    while start < len(clean):
        end = min(len(clean), start + chunk_chars)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(
                TextChunk(
                    chunk_id=f"C{index}",
                    text=chunk,
                    start=start,
                    end=end,
                    content_hash=content_hash(chunk),
                )
            )
            index += 1
        if end == len(clean):
            break
        start = end - overlap

    return tuple(chunks)


def _is_tracking_param(key: str) -> bool:
    lowered = key.lower()
    return lowered.startswith("utm_") or lowered in TRACKING_PARAMS
