"""Tone parsing and enrichment using GDELT GKG V2Tone."""

from __future__ import annotations

from ainee_politics.domain.models import ArticleRow
from .client import GdeltClient


def seendate_to_gkg_bucket(seendate: str) -> str:
    """Convert a GDELT `seendate` to the corresponding GKG bucket string."""

    return seendate.replace("T", "").replace("Z", "")


def parse_gdelt_v2tone(v2tone_raw: str, tone_source: str) -> dict[str, str | float | int]:
    """Parse the seven-part GDELT V2Tone field into typed columns."""

    default_payload: dict[str, str | float | int] = {
        "gdelt_v2tone_raw": v2tone_raw,
        "gdelt_tone_score": 0.0,
        "gdelt_tone_label": "unknown",
        "gdelt_positive_score": 0.0,
        "gdelt_negative_score": 0.0,
        "gdelt_polarity": 0.0,
        "gdelt_activity_reference_density": 0.0,
        "gdelt_self_group_reference_density": 0.0,
        "gdelt_word_count": 0,
        "gdelt_tone_source": tone_source,
    }
    if not v2tone_raw:
        return default_payload

    parts = v2tone_raw.split(",")
    if len(parts) != 7:
        return default_payload

    tone_score = float(parts[0])
    tone_label = "neutral"
    if tone_score > 0:
        tone_label = "positive"
    elif tone_score < 0:
        tone_label = "negative"

    return {
        "gdelt_v2tone_raw": v2tone_raw,
        "gdelt_tone_score": round(tone_score, 4),
        "gdelt_tone_label": tone_label,
        "gdelt_positive_score": round(float(parts[1]), 4),
        "gdelt_negative_score": round(float(parts[2]), 4),
        "gdelt_polarity": round(float(parts[3]), 4),
        "gdelt_activity_reference_density": round(float(parts[4]), 4),
        "gdelt_self_group_reference_density": round(float(parts[5]), 4),
        "gdelt_word_count": int(float(parts[6])),
        "gdelt_tone_source": tone_source,
    }


def enrich_row_with_gdelt_tone(row: ArticleRow, gdelt_client: GdeltClient) -> dict[str, str | float | int]:
    """Enrich a normalized article row with article-level GDELT tone."""

    seendate = str(row.get("seendate", ""))
    article_url = str(row.get("url", ""))
    if not seendate or not article_url:
        return parse_gdelt_v2tone("", "missing-seendate-or-url")

    bucket = seendate_to_gkg_bucket(seendate)
    bucket_map = gdelt_client.fetch_gkg_bucket_map(bucket)
    v2tone_raw = bucket_map.get(article_url, "")
    source = f"gdelt-gkg:{bucket}" if v2tone_raw else f"gdelt-gkg-missing:{bucket}"
    return parse_gdelt_v2tone(v2tone_raw, source)