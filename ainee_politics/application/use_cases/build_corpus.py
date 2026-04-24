"""Use case that builds the raw political corpus from GDELT."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ainee_politics.application.summaries import build_raw_summary
from ainee_politics.domain.catalog import DEFAULT_POLITICIANS
from ainee_politics.domain.models import ArticleRow, BuildCorpusSettings, Politician
from ainee_politics.infrastructure.gdelt import GdeltClient, build_query, enrich_row_with_gdelt_tone
from ainee_politics.infrastructure.storage import ensure_output_dir, write_csv, write_jsonl
from ainee_politics.infrastructure.text import deduplicate_rows, extract_article_payload, normalize_article


def resolve_politicians(max_politicians: int) -> tuple[Politician, ...]:
    """Select all politicians or just the first N for quick test runs."""

    if max_politicians > 0:
        return DEFAULT_POLITICIANS[:max_politicians]
    return DEFAULT_POLITICIANS


def enrich_rows(
    rows: list[ArticleRow],
    settings: BuildCorpusSettings,
    gdelt_client: GdeltClient,
    output_jsonl: Path,
    output_csv: Path,
    summary_path: Path,
) -> list[ArticleRow]:
    """Fetch content and GDELT tone, writing incremental checkpoints."""

    enriched_rows: list[ArticleRow] = []
    total_rows = len(rows)

    for index, row in enumerate(rows, start=1):
        payload = extract_article_payload(
            str(row.get("url", "")),
            timeout=settings.request_timeout,
            retries=settings.retries,
            session=gdelt_client.session,
        )
        enriched_row = dict(row)
        enriched_row.update(payload)
        enriched_row.update(enrich_row_with_gdelt_tone(enriched_row, gdelt_client))
        enriched_rows.append(enriched_row)

        should_checkpoint = index % max(1, settings.checkpoint_every) == 0 or index == total_rows
        if should_checkpoint:
            write_jsonl(output_jsonl, enriched_rows)
            write_csv(output_csv, enriched_rows)
            summary_path.write_text(json.dumps(build_raw_summary(enriched_rows), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[INFO] Checkpoint guardado: {index}/{total_rows} articulos procesados")

        if settings.sleep_seconds > 0:
            time.sleep(settings.sleep_seconds)

    return enriched_rows


def build_corpus(settings: BuildCorpusSettings) -> tuple[Path, Path, Path]:
    """Run the full corpus-building pipeline and return the output paths."""

    output_dir = ensure_output_dir(settings.output_dir)
    gdelt_client = GdeltClient(
        timeout=settings.request_timeout,
        retries=settings.retries,
        min_interval_seconds=settings.gdelt_min_interval,
    )

    rows: list[ArticleRow] = []
    politicians = resolve_politicians(settings.max_politicians)

    for politician in politicians:
        query = build_query(politician)
        print(f"[INFO] Descargando articulos para {politician.name} en en")
        try:
            articles = gdelt_client.fetch_articles(query, settings.timespan, settings.max_records)
        except Exception as error:
            print(f"[WARN] No se pudieron descargar articulos para {politician.name} en en: {error}")
            continue

        for article in articles:
            rows.append(normalize_article(politician, query, article))

    rows = deduplicate_rows(rows)

    output_jsonl = output_dir / "corpus_politicos_en.jsonl"
    output_csv = output_dir / "corpus_politicos_en.csv"
    summary_path = output_dir / "resumen_politicos_api.json"

    write_jsonl(output_jsonl, [])
    write_csv(output_csv, [])
    summary_path.write_text(json.dumps(build_raw_summary([]), ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Extrayendo texto para el dataset en ingles")
    enrich_rows(rows, settings, gdelt_client, output_jsonl, output_csv, summary_path)

    return output_jsonl, output_csv, summary_path