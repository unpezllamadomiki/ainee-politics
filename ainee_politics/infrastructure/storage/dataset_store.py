"""Persistence helpers for JSONL and CSV outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Iterable
from uuid import uuid4

from ainee_politics.domain.models import ArticleRow, SCHEMA_COLUMNS


def ensure_output_dir(output_dir: Path) -> Path:
    """Create the output directory if it does not exist."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _temporary_output_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _replace_with_retry(temp_path: Path, target_path: Path, retries: int = 3, delay_seconds: float = 0.2) -> None:
    last_error: OSError | None = None
    for attempt in range(retries + 1):
        try:
            temp_path.replace(target_path)
            return
        except OSError as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(delay_seconds * (attempt + 1))

    if temp_path.exists():
        temp_path.unlink()
    if last_error is not None:
        raise last_error


def write_jsonl(path: Path, rows: Iterable[ArticleRow]) -> None:
    """Write rows as newline-delimited JSON using UTF-8."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_output_path(path)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        _replace_with_retry(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def read_jsonl(path: Path) -> list[ArticleRow]:
    """Read a newline-delimited JSON file into memory."""

    if not path.exists():
        return []

    rows: list[ArticleRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[ArticleRow], fieldnames: list[str] | None = None) -> None:
    """Write rows using a stable and documented CSV schema."""

    selected_fieldnames = fieldnames or SCHEMA_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_output_path(path)
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=selected_fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in selected_fieldnames})
        _replace_with_retry(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()