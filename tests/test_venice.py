from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from research_agent.venice import VeniceClient


class VeniceClientTest(unittest.TestCase):
    def test_chat_stream_aggregates_streamed_chunks(self) -> None:
        class StreamResponse:
            status_code = 200
            headers: dict[str, str] = {}

            def __enter__(self) -> "StreamResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def raise_for_status(self) -> None:
                return None

            def iter_lines(self) -> list[str]:
                return [
                    'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                    'data: {"choices":[{"delta":{"content":" world"}}]}',
                    "data: [DONE]",
                ]

        with patch("research_agent.venice.httpx.stream", return_value=StreamResponse()) as stream:
            result = VeniceClient(api_key="test").chat_stream(
                [{"role": "user", "content": "Say hello"}],
                max_tokens=20,
            )

        self.assertEqual(result, "Hello world")
        stream.assert_called_once()
        self.assertEqual(str(stream.call_args.args[1]), "https://api.venice.ai/api/v1/chat/completions")
        self.assertTrue(stream.call_args.kwargs["json"]["stream"])

    def test_scrape_posts_to_augment_endpoint(self) -> None:
        response = httpx.Response(
            200,
            json={
                "data": {
                    "markdown": "# Page\n\nContent",
                    "title": "Page",
                    "url": "https://example.com/final",
                }
            },
            request=httpx.Request("POST", "https://api.venice.ai/api/v1/augment/scrape"),
        )

        with patch("research_agent.venice.httpx.post", return_value=response) as post:
            result = VeniceClient(api_key="test").scrape("https://example.com")

        self.assertEqual(result.content, "# Page\n\nContent")
        self.assertEqual(result.title, "Page")
        self.assertEqual(result.final_url, "https://example.com/final")
        post.assert_called_once()
        self.assertEqual(str(post.call_args.args[0]), "https://api.venice.ai/api/v1/augment/scrape")
        self.assertEqual(post.call_args.kwargs["json"], {"url": "https://example.com"})


if __name__ == "__main__":
    unittest.main()
