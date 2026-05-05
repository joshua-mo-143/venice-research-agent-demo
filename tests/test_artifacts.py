from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_agent.artifacts import ArtifactWriter
from research_agent.models import SearchResult


class ArtifactWriterTest(unittest.TestCase):
    def test_writes_dataclasses_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = ArtifactWriter(Path(tmp))
            writer.write("search_results", SearchResult(title="Title", url="https://example.com", snippet="Text"))

            records = (Path(tmp) / "search_results.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0])["canonical_url"], "https://example.com/")


if __name__ == "__main__":
    unittest.main()
