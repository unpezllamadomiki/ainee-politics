"""HTTP client encapsulating the interaction with GDELT DOC and GKG endpoints."""

from __future__ import annotations

import io
import time
import zipfile
from typing import Any

import requests

from ainee_politics.config import DOC_API_URL, GDELT_RATE_LIMIT_MESSAGE, GKG_URL_TEMPLATE, RETRY_BACKOFF_SECONDS, USER_AGENT


class GdeltClient:
    """Thin client with rate limiting, retries and bucket caching."""

    def __init__(self, timeout: float, retries: int, min_interval_seconds: float) -> None:
        self.timeout = timeout
        self.retries = retries
        self.min_interval_seconds = min_interval_seconds
        self.last_gdelt_request_ts = 0.0
        self.gkg_bucket_cache: dict[str, dict[str, str]] = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def wait_for_gdelt_slot(self) -> None:
        """Sleep if needed to respect the documented GDELT request interval."""

        elapsed = time.monotonic() - self.last_gdelt_request_ts
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            print(f"[INFO] Esperando {remaining:.1f}s para respetar el limite de GDELT")
            time.sleep(remaining)

    def request_json(self, params: dict[str, Any]) -> dict[str, Any]:
        """Request JSON from the DOC API using retries and rate limiting."""

        last_error: Exception | None = None
        response: requests.Response | None = None

        for attempt in range(1, self.retries + 1):
            try:
                self.wait_for_gdelt_slot()
                response = self.session.get(DOC_API_URL, params=params, timeout=self.timeout)
                self.last_gdelt_request_ts = time.monotonic()
                response.raise_for_status()
                if GDELT_RATE_LIMIT_MESSAGE.lower() in response.text.lower():
                    raise requests.HTTPError("GDELT rate limit exceeded: one request every 5 seconds")
                return response.json()
            except ValueError as error:
                last_error = error
                if attempt == self.retries:
                    break
                wait_seconds = max(RETRY_BACKOFF_SECONDS * attempt, self.min_interval_seconds)
                print(
                    f"[WARN] Respuesta no JSON desde GDELT, reintento {attempt}/{self.retries} en {wait_seconds:.1f}s"
                )
                if response is not None:
                    body_preview = response.text[:400].replace("\n", " ")
                    if body_preview:
                        print(f"[WARN] Cuerpo recibido desde GDELT: {body_preview}")
                time.sleep(wait_seconds)
            except requests.RequestException as error:
                last_error = error
                if attempt == self.retries:
                    break
                wait_seconds = max(RETRY_BACKOFF_SECONDS * attempt, self.min_interval_seconds)
                print(f"[WARN] Error consultando GDELT, reintento {attempt}/{self.retries} en {wait_seconds:.1f}s")
                time.sleep(wait_seconds)

        if last_error is None:
            raise RuntimeError("Fallo inesperado al consultar GDELT sin excepcion capturada")
        raise RuntimeError(str(last_error)) from last_error

    def fetch_articles(self, query: str, timespan: str, max_records: int) -> list[dict[str, Any]]:
        """Fetch articles from GDELT DOC API."""

        payload = {
            "query": query,
            "mode": "artlist",
            "maxrecords": str(max_records),
            "format": "json",
            "sort": "datedesc",
            "timespan": timespan,
        }
        data = self.request_json(payload)
        return data.get("articles", [])

    def fetch_gkg_bucket_map(self, bucket: str) -> dict[str, str]:
        """Download a GKG bucket and build a URL to V2Tone map."""

        if bucket in self.gkg_bucket_cache:
            return self.gkg_bucket_cache[bucket]

        gkg_map: dict[str, str] = {}
        last_error: requests.RequestException | None = None
        gkg_url = GKG_URL_TEMPLATE.format(bucket=bucket)

        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.get(gkg_url, timeout=self.timeout)
                response.raise_for_status()
                archive = zipfile.ZipFile(io.BytesIO(response.content))
                entry_name = archive.namelist()[0]
                with archive.open(entry_name) as handle:
                    for raw_line in handle:
                        line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                        fields = line.split("\t")
                        if len(fields) < 16:
                            continue
                        document_url = fields[4]
                        v2tone_raw = fields[15]
                        if document_url and v2tone_raw:
                            gkg_map[document_url] = v2tone_raw

                self.gkg_bucket_cache[bucket] = gkg_map
                return gkg_map
            except requests.RequestException as error:
                last_error = error
                if attempt == self.retries:
                    break
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        if last_error is not None:
            print(f"[WARN] No se pudo descargar GKG para {bucket}: {last_error}")
        self.gkg_bucket_cache[bucket] = gkg_map
        return gkg_map