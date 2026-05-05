from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class ArtifactWriter:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def write(self, kind: str, record: object) -> None:
        if self.root is None:
            return

        path = self.root / f"{kind}.jsonl"
        payload = json.dumps(_to_jsonable(record), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(f"{payload}\n")


def _to_jsonable(value: object) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
