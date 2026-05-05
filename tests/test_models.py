from __future__ import annotations

import unittest

from research_agent.models import canonicalize_url, chunk_text


class ModelHelpersTest(unittest.TestCase):
    def test_canonicalize_url_strips_tracking_and_fragments(self) -> None:
        url = "HTTPS://Example.com:443/research/?b=2&utm_source=x&a=1#section"

        self.assertEqual(canonicalize_url(url), "https://example.com/research?a=1&b=2")

    def test_chunk_text_adds_overlap_and_hashes(self) -> None:
        chunks = chunk_text("abcdefghi", chunk_chars=4, overlap=1)

        self.assertEqual([chunk.text for chunk in chunks], ["abcd", "defg", "ghi"])
        self.assertEqual(chunks[0].chunk_id, "C1")
        self.assertEqual(len(chunks[0].content_hash), 64)


if __name__ == "__main__":
    unittest.main()
