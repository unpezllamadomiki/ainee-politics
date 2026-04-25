"""Domain models and shared application types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from ainee_politics.config import API_RETRIES, GDELT_MIN_INTERVAL_SECONDS, REQUEST_TIMEOUT

ArticleRow: TypeAlias = dict[str, Any]

SCHEMA_COLUMNS = [
    "dataset_language",
    "politician",
    "title",
    "url",
    "domain",
    "seendate",
    "sourcecountry",
    "language",
    "socialimage",
    "content",
    "content_fetch_status",
    "content_length_chars",
    "content_length_words",
    "gdelt_v2tone_raw",
    "gdelt_tone_score",
    "gdelt_tone_label",
    "gdelt_positive_score",
    "gdelt_negative_score",
    "gdelt_polarity",
    "gdelt_activity_reference_density",
    "gdelt_self_group_reference_density",
    "gdelt_word_count",
    "gdelt_tone_source",
    "query",
]

CLEAN_SCHEMA_COLUMNS = SCHEMA_COLUMNS + [
    "normalized_url",
    "text",
    "mentioned_aliases",
    "relevance_status",
]


@dataclass(frozen=True)
class Politician:
    """Represents a politician and the aliases used to query GDELT."""

    name: str
    aliases: tuple[str, ...]
    role_hint_en: str


@dataclass(frozen=True)
class BuildCorpusSettings:
    """Runtime configuration for the corpus-building pipeline."""

    output_dir: Path
    timespan: str = "30d"
    max_records: int = 75
    sleep_seconds: float = 1.0
    request_timeout: float = REQUEST_TIMEOUT
    retries: int = API_RETRIES
    gdelt_min_interval: float = GDELT_MIN_INTERVAL_SECONDS
    max_politicians: int = 0
    checkpoint_every: int = 10


@dataclass(frozen=True)
class PrepareDatasetSettings:
    """Runtime configuration for the clean dataset preparation layer."""

    input_path: Path
    output_dir: Path
    keep_neutral: bool = False
    allow_empty_content: bool = False
    min_content_chars: int = 200
    use_alias_filter: bool = True


LABELED_SCHEMA_COLUMNS = CLEAN_SCHEMA_COLUMNS + [
    "spacy_entities",
    "politician_adjectives",
    "sentence_count",
    "avg_sentence_length",
]


@dataclass(frozen=True)
class LabelSettings:
    """Runtime configuration for the spaCy NLP enrichment step."""

    input_path: Path
    output_dir: Path
    spacy_model: str = "en_core_web_lg"
    batch_size: int = 32


@dataclass(frozen=True)
class TrainingSettings:
    """Runtime configuration for the tone-classification training pipeline."""

    input_path: Path
    output_dir: Path
    cv_folds: int = 5
    max_tfidf_features: int = 10_000
    transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    text_max_chars: int = 1500