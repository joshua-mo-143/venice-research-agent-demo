from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .models import ScrapeResult


DEFAULT_BASE_URL = "https://api.venice.ai/api/v1"
DEFAULT_MODEL = "venice-uncensored"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class VeniceError(RuntimeError):
    """Raised when the Venice API returns an unusable response."""


@dataclass(frozen=True)
class VeniceClient:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 60.0
    max_retries: int = 2
    backoff_seconds: float = 1.0

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        *,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
    ) -> "VeniceClient":
        api_key = os.getenv("VENICE_API_KEY")
        if not api_key:
            raise VeniceError("VENICE_API_KEY is required. Add it to your environment or .env file.")

        return cls(
            api_key=api_key,
            model=model or os.getenv("VENICE_MODEL", DEFAULT_MODEL),
            base_url=os.getenv("VENICE_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1600,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data = self._post_json("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise VeniceError(f"Unexpected Venice API response: {data}") from exc

    def scrape(self, url: str) -> ScrapeResult:
        data = self._post_json("/augment/scrape", {"url": url})
        content = _first_string(data, "content", "markdown", "text")
        if not content:
            raise VeniceError(f"Unexpected Venice scrape response: {data}")

        return ScrapeResult(
            url=url,
            final_url=_first_string(data, "final_url", "url", "source_url") or url,
            title=_first_string(data, "title"),
            content=content,
            content_type="text/markdown",
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}{path}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    _sleep_before_retry(attempt, self.backoff_seconds, response=response)
                    continue
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    _sleep_before_retry(attempt, self.backoff_seconds, response=exc.response)
                    continue
                body = exc.response.text[:1000]
                raise VeniceError(f"Venice API returned {exc.response.status_code}: {body}") from exc
            except httpx.HTTPError as exc:
                if attempt < self.max_retries:
                    _sleep_before_retry(attempt, self.backoff_seconds)
                    continue
                raise VeniceError(f"Could not reach Venice API: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise VeniceError(f"Unexpected Venice API response: {response.text[:1000]}") from exc

        if not isinstance(data, dict):
            raise VeniceError(f"Unexpected Venice API response: {data}")
        return data


def _sleep_before_retry(
    attempt: int,
    backoff_seconds: float,
    response: httpx.Response | None = None,
) -> None:
    retry_after = response.headers.get("retry-after") if response is not None else None
    if retry_after:
        try:
            time.sleep(float(retry_after))
            return
        except ValueError:
            pass

    time.sleep(backoff_seconds * (2**attempt))


def _first_string(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for nested_key in ("data", "result", "scrape"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            value = _first_string(nested, *keys)
            if value:
                return value

    return ""
