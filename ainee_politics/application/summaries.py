"""Summary builders for application outputs."""

from __future__ import annotations

from ainee_politics.domain.models import ArticleRow, CLEAN_SCHEMA_COLUMNS, SCHEMA_COLUMNS


def build_raw_summary(rows: list[ArticleRow]) -> dict[str, object]:
    """Build metadata for the raw corpus files."""

    counts: dict[str, int] = {}
    for row in rows:
        politician = str(row["politician"])
        counts[politician] = counts.get(politician, 0) + 1

    return {
        "dataset_en": len(rows),
        "en_by_politician": dict(sorted(counts.items())),
        "schema_note": (
            "Cada fila representa una noticia recuperada desde la DOC API de GDELT. "
            "El campo content se extrae automaticamente desde la URL y el tono por articulo "
            "se enlaza desde GDELT GKG usando la URL y el seendate de la noticia. "
            "La deduplicacion normaliza URLs para reducir variantes equivalentes."
        ),
        "recommended_columns": SCHEMA_COLUMNS,
    }


def build_prepared_summary(rows: list[ArticleRow]) -> dict[str, object]:
    """Build metadata for the prepared clean dataset."""

    counts_by_politician: dict[str, int] = {}
    counts_by_tone: dict[str, int] = {}
    for row in rows:
        politician = str(row.get("politician", ""))
        tone_label = str(row.get("gdelt_tone_label", ""))
        counts_by_politician[politician] = counts_by_politician.get(politician, 0) + 1
        counts_by_tone[tone_label] = counts_by_tone.get(tone_label, 0) + 1

    return {
        "clean_rows": len(rows),
        "rows_by_politician": dict(sorted(counts_by_politician.items())),
        "rows_by_tone": dict(sorted(counts_by_tone.items())),
        "schema_note": (
            "El dataset limpio elimina duplicados por URL normalizada, puede filtrar por aliases del politico "
            "y añade un campo text que concatena title y content para modelado."
        ),
        "recommended_columns": CLEAN_SCHEMA_COLUMNS,
    }