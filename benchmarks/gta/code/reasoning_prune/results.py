"""Comparable baseline/pruned result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ComparisonWriter:
    def __init__(self, output_dir: str | Path, benchmark: str, variant: str):
        self.root = Path(output_dir) / benchmark / variant
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows_path = self.root / "predictions.jsonl"

    def append(self, row: dict[str, Any]) -> None:
        with self.rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> Path:
        path = self.root / "summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path
