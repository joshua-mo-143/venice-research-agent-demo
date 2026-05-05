from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from research_agent.venice import VeniceClient


class VeniceClientTest(unittest.TestCase):
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
