"""Row normalization and deduplication utilities."""

from __future__ import annotations

from urllib.parse import SplitResult, urlsplit, urlunsplit

from ainee_politics.domain.models import ArticleRow, Politician


def normalize_url(url: str) -> str:
    """Normalize URLs to reduce obvious duplicates."""

    if not url:
        return ""

    parsed = urlsplit(url)
    hostname = parsed.hostname.lower() if parsed.hostname else ""
    netloc = hostname
    if parsed.port and not (
        (parsed.scheme.lower() == "https" and parsed.port == 443)
        or (parsed.scheme.lower() == "http" and parsed.port == 80)
    ):
        netloc = f"{hostname}:{parsed.port}"

    path = parsed.path.rstrip("/") or "/"
    normalized = SplitResult(parsed.scheme.lower(), netloc, path, parsed.query, "")
    return urlunsplit(normalized)


def normalize_article(politician: Politician, query: str, article: dict[str, object]) -> ArticleRow:
    """Map the raw GDELT article payload to the stable output schema."""

    return {
        "dataset_language": "en",
        "politician": politician.name,
        "query": query,
        "title": article.get("title", ""),
        "url": normalize_url(str(article.get("url", ""))),
        "domain": article.get("domain", ""),
        "seendate": article.get("seendate", ""),
        "sourcecountry": article.get("sourcecountry", ""),
        "language": article.get("language", ""),
        "socialimage": article.get("socialimage", ""),
        "content": "",
        "content_fetch_status": "pending",
        "content_length_chars": 0,
        "content_length_words": 0,
        "gdelt_v2tone_raw": "",
        "gdelt_tone_score": 0.0,
        "gdelt_tone_label": "pending",
        "gdelt_positive_score": 0.0,
        "gdelt_negative_score": 0.0,
        "gdelt_polarity": 0.0,
        "gdelt_activity_reference_density": 0.0,
        "gdelt_self_group_reference_density": 0.0,
        "gdelt_word_count": 0,
        "gdelt_tone_source": "pending",
    }


def deduplicate_rows(rows: list[ArticleRow]) -> list[ArticleRow]:
    """Remove duplicated politician and URL pairs after URL normalization."""

    deduped: list[ArticleRow] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("politician", "")), normalize_url(str(row.get("url", ""))))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped