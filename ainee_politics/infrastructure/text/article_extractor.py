"""HTML content extraction for article bodies."""

from __future__ import annotations

import re
import time

import requests

from ainee_politics.config import RETRY_BACKOFF_SECONDS, USER_AGENT


def extract_article_payload(
    url: str,
    timeout: float,
    retries: int,
    session: requests.Session | None = None,
) -> dict[str, str | int]:
    """Download an article URL and extract paragraph text."""

    http = session or requests.Session()
    last_error: requests.RequestException | None = None
    response: requests.Response | None = None

    for attempt in range(1, retries + 1):
        try:
            response = http.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            break
        except requests.RequestException as error:
            last_error = error
            if attempt == retries:
                response = None
                break
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    if response is None:
        error_label = type(last_error).__name__ if last_error else "error"
        return {
            "content": "",
            "content_fetch_status": error_label,
            "content_length_chars": 0,
            "content_length_words": 0,
        }

    html = response.text
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)

    clean_parts: list[str] = []
    for paragraph in paragraphs:
        text = re.sub(r"<[^>]+>", " ", paragraph)
        text = re.sub(r"&nbsp;|&#160;", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"&amp;", "&", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= 40:
            clean_parts.append(text)

    content = "\n\n".join(clean_parts[:20])
    return {
        "content": content,
        "content_fetch_status": "ok" if content else "empty",
        "content_length_chars": len(content),
        "content_length_words": len(content.split()),
    }