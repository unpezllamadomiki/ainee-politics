"""Use case that prepares the clean dataset for downstream modeling."""

from __future__ import annotations

import json
from pathlib import Path

from ainee_politics.application.summaries import build_prepared_summary
from ainee_politics.domain.catalog import DEFAULT_POLITICIANS
from ainee_politics.domain.models import ArticleRow, CLEAN_SCHEMA_COLUMNS, PrepareDatasetSettings
from ainee_politics.infrastructure.storage import ensure_output_dir, read_jsonl, write_csv, write_jsonl
from ainee_politics.infrastructure.text import normalize_url


def build_alias_index() -> dict[str, tuple[str, ...]]:
    """Build a politician-to-alias mapping from the default catalog."""

    return {politician.name: politician.aliases for politician in DEFAULT_POLITICIANS}


def build_model_text(title: str, content: str) -> str:
    """Create the final modeling text by concatenating title and content."""

    parts = [part.strip() for part in (title, content) if part and part.strip()]
    return "\n\n".join(parts)


def find_mentioned_aliases(title: str, content: str, aliases: tuple[str, ...]) -> list[str]:
    """Return the aliases explicitly present in the title or content."""

    haystack = f"{title}\n{content}".lower()
    found = []
    for alias in aliases:
        if alias.lower() in haystack:
            found.append(alias)
    return sorted(set(found))


def prepare_rows(rows: list[ArticleRow], settings: PrepareDatasetSettings) -> list[ArticleRow]:
    """Apply cleaning, deduplication and relevance filtering to raw rows."""

    alias_index = build_alias_index()
    prepared_rows: list[ArticleRow] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        title = str(row.get("title", "")).strip()
        content = str(row.get("content", "")).strip()
        tone_label = str(row.get("gdelt_tone_label", "")).strip().lower()
        content_status = str(row.get("content_fetch_status", "")).strip().lower()
        normalized_url = normalize_url(str(row.get("url", "")))

        if not settings.allow_empty_content:
            if content_status != "ok":
                continue
            if len(content) < settings.min_content_chars:
                continue

        if not settings.keep_neutral and tone_label == "neutral":
            continue

        dedupe_key = (str(row.get("politician", "")), normalized_url)
        if dedupe_key in seen:
            continue

        aliases = alias_index.get(str(row.get("politician", "")), tuple())
        mentioned_aliases = find_mentioned_aliases(title, content, aliases)
        if settings.use_alias_filter and not mentioned_aliases:
            continue

        prepared_row = dict(row)
        prepared_row["url"] = normalized_url
        prepared_row["normalized_url"] = normalized_url
        prepared_row["text"] = build_model_text(title, content)
        prepared_row["mentioned_aliases"] = "|".join(mentioned_aliases)
        prepared_row["relevance_status"] = "alias-match" if mentioned_aliases else "unchecked"

        prepared_rows.append(prepared_row)
        seen.add(dedupe_key)

    return prepared_rows


def prepare_dataset(settings: PrepareDatasetSettings) -> tuple[Path, Path, Path, int]:
    """Run the preparation pipeline and persist the clean dataset outputs."""

    rows = read_jsonl(settings.input_path)
    prepared_rows = prepare_rows(rows, settings)

    output_dir = ensure_output_dir(settings.output_dir)
    output_jsonl = output_dir / "corpus_politicos_clean.jsonl"
    output_csv = output_dir / "corpus_politicos_clean.csv"
    summary_path = output_dir / "resumen_preparacion.json"

    write_jsonl(output_jsonl, prepared_rows)
    write_csv(output_csv, prepared_rows, fieldnames=CLEAN_SCHEMA_COLUMNS)
    summary_path.write_text(json.dumps(build_prepared_summary(prepared_rows), ensure_ascii=False, indent=2), encoding="utf-8")

    return output_jsonl, output_csv, summary_path, len(prepared_rows)