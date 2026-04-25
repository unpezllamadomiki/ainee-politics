"""label-corpus use case: enrich the clean JSONL with spaCy NLP annotations."""

from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from ainee_politics.domain.models import LABELED_SCHEMA_COLUMNS, LabelSettings
from ainee_politics.infrastructure.nlp.spacy_processor import enrich_rows, load_spacy_model
from ainee_politics.infrastructure.storage.dataset_store import (
    ensure_output_dir,
    read_jsonl,
    write_csv,
    write_jsonl,
)


def label_corpus(settings: LabelSettings) -> tuple[Path, Path]:
    """Enrich the clean corpus with spaCy NER, politician modifiers and sentence stats.

    Reads ``settings.input_path`` (clean JSONL), processes every row through
    ``en_core_web_lg`` (or the configured model), and writes:

    - ``corpus_labeled.jsonl``
    - ``corpus_labeled.csv``
    """
    ensure_output_dir(settings.output_dir)

    rows = read_jsonl(settings.input_path)
    if not rows:
        raise ValueError(f"No se encontraron filas en {settings.input_path}")

    print(f"[INFO] Cargando modelo spaCy '{settings.spacy_model}'...")
    nlp = load_spacy_model(settings.spacy_model)

    print(f"[INFO] Enriqueciendo {len(rows)} artículos con spaCy NLP...")
    enriched = enrich_rows(rows, nlp, batch_size=settings.batch_size)

    out_jsonl = settings.output_dir / "corpus_labeled.jsonl"
    out_csv = settings.output_dir / "corpus_labeled.csv"
    write_jsonl(out_jsonl, enriched)
    write_csv(out_csv, enriched, fieldnames=LABELED_SCHEMA_COLUMNS)

    print(f"[INFO] {len(enriched)} artículos enriquecidos.")
    return out_jsonl, out_csv
