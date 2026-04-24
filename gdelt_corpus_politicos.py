"""Construye un corpus politico en ingles usando GDELT y extrae texto automaticamente.

- GDELT se usa para descubrir las noticias.
- El corpus objetivo se genera en ingles.
- Se intenta descargar automaticamente parte del texto del articulo en `content`.

Limitacion importante:
La DOC API no devuelve el texto completo del articulo. Por eso este script hace
una segunda fase automatica: descarga la URL de cada noticia y extrae parrafos
del HTML para guardarlos en `content`, es decir, que no garantiza un texto perfecto.

Uso rapido:

    python gdelt_corpus_politicos.py

Salida:

    data/
      corpus_politicos_en.jsonl
      corpus_politicos_en.csv
      resumen_politicos_api.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GKG_URL_TEMPLATE = "http://data.gdeltproject.org/gdeltv2/{bucket}.gkg.csv.zip"
REQUEST_TIMEOUT = 30
API_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
GDELT_MIN_INTERVAL_SECONDS = 5.2
GDELT_RATE_LIMIT_MESSAGE = "Please limit requests to one every 5 seconds"
USER_AGENT = (
    "Mozilla/5.0 (compatible; TrabajoAineePoliticalCorpusAPI/3.0; +https://gdeltproject.org)"
)

GDELT_SOURCE_LANGUAGE = "english"

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

LAST_GDELT_REQUEST_TS = 0.0
GKG_BUCKET_CACHE: dict[str, dict[str, str]] = {}


@dataclass(frozen=True)
class Politician:
    name: str
    aliases: tuple[str, ...]
    role_hint_en: str


DEFAULT_POLITICIANS: tuple[Politician, ...] = (
    Politician("Donald Trump", ("Donald Trump", "Trump"), "president OR election OR campaign"),
    Politician("Joe Biden", ("Joe Biden", "Biden"), "president OR white house OR election"),
    Politician("Kamala Harris", ("Kamala Harris", "Harris"), "vice president OR election OR campaign"),
    Politician("Pedro Sanchez", ("Pedro Sanchez",), "prime minister OR government OR parliament"),
    Politician("Alberto Nunez Feijoo", ("Alberto Nunez Feijoo", "Feijoo"), "opposition OR parliament OR election"),
    Politician("Ursula von der Leyen", ("Ursula von der Leyen",), "european commission OR eu OR parliament"),
    Politician("Emmanuel Macron", ("Emmanuel Macron", "Macron"), "president OR government OR parliament"),
    Politician("Giorgia Meloni", ("Giorgia Meloni", "Meloni"), "prime minister OR government OR parliament"),
    Politician("Olaf Scholz", ("Olaf Scholz", "Scholz"), "chancellor OR government OR parliament"),
    Politician("Keir Starmer", ("Keir Starmer", "Starmer"), "prime minister OR labour OR parliament"),
    Politician("Volodymyr Zelenskyy", ("Volodymyr Zelenskyy", "Zelenskyy", "Zelensky"), "president OR war OR government"),
    Politician("Vladimir Putin", ("Vladimir Putin", "Putin"), "president OR kremlin OR government"),
    Politician("Benjamin Netanyahu", ("Benjamin Netanyahu", "Netanyahu"), "prime minister OR government OR war"),
    Politician("Narendra Modi", ("Narendra Modi", "Modi"), "prime minister OR government OR election"),
    Politician("Xi Jinping", ("Xi Jinping", "Xi Jinping"), "president OR communist party OR government"),
    Politician("Javier Milei", ("Javier Milei", "Milei"), "president OR government OR congress"),
    Politician("Gustavo Petro", ("Gustavo Petro", "Petro"), "president OR government OR congress"),
    Politician("Claudia Sheinbaum", ("Claudia Sheinbaum", "Sheinbaum"), "president OR government OR election"),
    Politician("Luiz Inacio Lula da Silva", ("Luiz Inacio Lula da Silva", "Lula da Silva", "Lula"), "president OR government OR congress"),
)


def build_alias_query(aliases: tuple[str, ...]) -> str:
    return " OR ".join(f'"{alias}"' for alias in aliases)


def build_query(politician: Politician, timespan: str) -> str:
    alias_query = build_alias_query(politician.aliases)
    return " ".join(
        [
            f"({alias_query})",
            f"sourcelang:{GDELT_SOURCE_LANGUAGE}",
        ]
    )


def wait_for_gdelt_slot(min_interval_seconds: float) -> None:
    global LAST_GDELT_REQUEST_TS
    elapsed = time.monotonic() - LAST_GDELT_REQUEST_TS
    remaining = min_interval_seconds - elapsed
    if remaining > 0:
        print(f"[INFO] Esperando {remaining:.1f}s para respetar el limite de GDELT")
        time.sleep(remaining)


def request_json(
    params: dict[str, Any],
    timeout: float,
    retries: int,
    min_interval_seconds: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            wait_for_gdelt_slot(min_interval_seconds)
            response = requests.get(
                DOC_API_URL,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            global LAST_GDELT_REQUEST_TS
            LAST_GDELT_REQUEST_TS = time.monotonic()
            response.raise_for_status()
            if GDELT_RATE_LIMIT_MESSAGE.lower() in response.text.lower():
                raise requests.HTTPError("GDELT rate limit exceeded: one request every 5 seconds")
            return response.json()
        except ValueError as error:
            last_error = error
            body_preview = response.text[:400].replace("\n", " ") if "response" in locals() else ""
            if attempt == retries:
                break
            wait_seconds = max(RETRY_BACKOFF_SECONDS * attempt, min_interval_seconds)
            print(f"[WARN] Respuesta no JSON desde GDELT, reintento {attempt}/{retries} en {wait_seconds:.1f}s")
            if body_preview:
                print(f"[WARN] Cuerpo recibido desde GDELT: {body_preview}")
            time.sleep(wait_seconds)
        except requests.RequestException as error:
            last_error = error
            if attempt == retries:
                break
            wait_seconds = max(RETRY_BACKOFF_SECONDS * attempt, min_interval_seconds)
            print(f"[WARN] Error consultando GDELT, reintento {attempt}/{retries} en {wait_seconds:.1f}s")
            time.sleep(wait_seconds)

    if last_error is None:
        raise RuntimeError("Fallo inesperado al consultar GDELT sin excepcion capturada")
    raise RuntimeError(str(last_error)) from last_error


def fetch_articles(
    query: str,
    timespan: str,
    max_records: int,
    timeout: float,
    retries: int,
    min_interval_seconds: float,
) -> list[dict[str, Any]]:
    payload = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_records),
        "format": "json",
        "sort": "datedesc",
        "timespan": timespan,
    }
    data = request_json(payload, timeout, retries, min_interval_seconds)
    return data.get("articles", [])


def extract_article_payload(url: str, timeout: float, retries: int) -> dict[str, str | int]:
    last_error: requests.RequestException | None = None
    response: requests.Response | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
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


def seendate_to_gkg_bucket(seendate: str) -> str:
    return seendate.replace("T", "").replace("Z", "")


def fetch_gkg_bucket_map(bucket: str, timeout: float, retries: int) -> dict[str, str]:
    if bucket in GKG_BUCKET_CACHE:
        return GKG_BUCKET_CACHE[bucket]

    last_error: requests.RequestException | None = None
    gkg_map: dict[str, str] = {}
    gkg_url = GKG_URL_TEMPLATE.format(bucket=bucket)
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(gkg_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
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
            GKG_BUCKET_CACHE[bucket] = gkg_map
            return gkg_map
        except requests.RequestException as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    if last_error is not None:
        print(f"[WARN] No se pudo descargar GKG para {bucket}: {last_error}")
    GKG_BUCKET_CACHE[bucket] = gkg_map
    return gkg_map


def parse_gdelt_v2tone(v2tone_raw: str, tone_source: str) -> dict[str, str | float | int]:
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
    if tone_score > 0:
        tone_label = "positive"
    elif tone_score < 0:
        tone_label = "negative"
    else:
        tone_label = "neutral"

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


def enrich_row_with_gdelt_tone(row: dict[str, Any], timeout: float, retries: int) -> dict[str, str | float | int]:
    seendate = row.get("seendate", "")
    article_url = row.get("url", "")
    if not seendate or not article_url:
        return parse_gdelt_v2tone("", "missing-seendate-or-url")

    bucket = seendate_to_gkg_bucket(seendate)
    bucket_map = fetch_gkg_bucket_map(bucket, timeout, retries)
    v2tone_raw = bucket_map.get(article_url, "")
    source = f"gdelt-gkg:{bucket}" if v2tone_raw else f"gdelt-gkg-missing:{bucket}"
    return parse_gdelt_v2tone(v2tone_raw, source)


def normalize_article(
    politician: Politician,
    query: str,
    article: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_language": "en",
        "politician": politician.name,
        "query": query,
        "title": article.get("title", ""),
        "url": article.get("url", ""),
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


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.get("politician", ""), row.get("url", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def enrich_rows_with_content(
    rows: list[dict[str, Any]],
    sleep_seconds: float,
    timeout: float,
    retries: int,
    output_jsonl: Path,
    output_csv: Path,
    summary_path: Path,
    checkpoint_every: int,
) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    total_rows = len(rows)
    for index, row in enumerate(rows, start=1):
        payload = extract_article_payload(row.get("url", ""), timeout, retries)
        enriched_row = dict(row)
        enriched_row.update(payload)
        enriched_row.update(enrich_row_with_gdelt_tone(enriched_row, timeout, retries))
        enriched_rows.append(enriched_row)

        should_checkpoint = index % checkpoint_every == 0 or index == total_rows
        if should_checkpoint:
            write_jsonl(output_jsonl, enriched_rows)
            write_csv(output_csv, enriched_rows)
            summary = build_summary(enriched_rows)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[INFO] Checkpoint guardado: {index}/{total_rows} articulos procesados")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return enriched_rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else SCHEMA_COLUMNS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def build_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def count_by_politician(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            politician = row["politician"]
            counts[politician] = counts.get(politician, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "dataset_en": len(rows),
        "en_by_politician": count_by_politician(rows),
        "schema_note": (
            "Cada fila representa una noticia recuperada desde la DOC API de GDELT. "
            "El campo content se extrae automaticamente desde la URL y el tono por articulo "
            "se enlaza desde GDELT GKG usando la URL y el seendate de la noticia."
        ),
        "recommended_columns": SCHEMA_COLUMNS,
    }


def ensure_dirs(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye un corpus politico usando la API DOC de GDELT."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directorio donde se guardan los ficheros de salida.",
    )
    parser.add_argument(
        "--timespan",
        default="30d",
        help="Ventana temporal GDELT, por ejemplo 7d, 30d o 3m.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=75,
        help="Numero maximo de articulos por politico y por idioma.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Pausa entre peticiones para no saturar la API.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=REQUEST_TIMEOUT,
        help="Timeout en segundos para peticiones HTTP a GDELT y a las noticias.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=API_RETRIES,
        help="Numero de reintentos ante timeout o fallo de red.",
    )
    parser.add_argument(
        "--gdelt-min-interval",
        type=float,
        default=GDELT_MIN_INTERVAL_SECONDS,
        help="Segundos minimos entre peticiones a la API de GDELT.",
    )
    parser.add_argument(
        "--max-politicians",
        type=int,
        default=0,
        help="Si es mayor que 0, procesa solo los primeros N politicos para pruebas rapidas.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Numero de articulos procesados entre guardados incrementales del CSV y JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dirs(Path(args.output_dir))
    politicians = DEFAULT_POLITICIANS[: args.max_politicians] if args.max_politicians > 0 else DEFAULT_POLITICIANS
    rows: list[dict[str, Any]] = []

    for politician in politicians:
        query = build_query(politician, args.timespan)
        print(f"[INFO] Descargando articulos para {politician.name} en en")
        try:
            articles = fetch_articles(
                query,
                args.timespan,
                args.max_records,
                args.request_timeout,
                args.retries,
                args.gdelt_min_interval,
            )
        except Exception as error:
            print(f"[WARN] No se pudieron descargar articulos para {politician.name} en en: {error}")
            continue
        for article in articles:
            rows.append(normalize_article(politician, query, article))

    rows = deduplicate_rows(rows)

    en_jsonl = output_dir / "corpus_politicos_en.jsonl"
    en_csv = output_dir / "corpus_politicos_en.csv"
    summary_path = output_dir / "resumen_politicos_api.json"

    write_jsonl(en_jsonl, [])
    write_csv(en_csv, [])
    summary_path.write_text(json.dumps(build_summary([]), ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Extrayendo texto para el dataset en ingles")
    rows = enrich_rows_with_content(
        rows,
        args.sleep_seconds,
        args.request_timeout,
        args.retries,
        en_jsonl,
        en_csv,
        summary_path,
        max(1, args.checkpoint_every),
    )

    print(f"[OK] EN JSONL: {en_jsonl}")
    print(f"[OK] EN CSV:   {en_csv}")
    print(f"[OK] META:     {summary_path}")


if __name__ == "__main__":
    main()