"""CLI entrypoints for the project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ainee_politics.application import build_corpus, prepare_dataset
from ainee_politics.config import API_RETRIES, GDELT_MIN_INTERVAL_SECONDS, REQUEST_TIMEOUT
from ainee_politics.domain.models import BuildCorpusSettings, PrepareDatasetSettings


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the arguments shared by the corpus-building entrypoints."""

    parser.add_argument("--output-dir", default="data", help="Directorio donde se guardan los ficheros de salida.")
    parser.add_argument("--timespan", default="30d", help="Ventana temporal GDELT, por ejemplo 7d, 30d o 3m.")
    parser.add_argument("--max-records", type=int, default=75, help="Numero maximo de articulos por politico.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Pausa entre descargas de articulos para no saturar los medios origen.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=REQUEST_TIMEOUT,
        help="Timeout en segundos para peticiones HTTP a GDELT y a las noticias.",
    )
    parser.add_argument("--retries", type=int, default=API_RETRIES, help="Numero de reintentos ante timeout o fallo de red.")
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


def add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the arguments used by the dataset preparation layer."""

    parser.add_argument("--input", default="data/corpus_politicos_en.jsonl", help="Ruta del corpus crudo en JSONL.")
    parser.add_argument("--output-dir", default="data", help="Directorio donde se guardan los ficheros del dataset limpio.")
    parser.add_argument("--keep-neutral", action="store_true", help="Conserva filas con etiqueta neutral en el dataset limpio.")
    parser.add_argument(
        "--allow-empty-content",
        action="store_true",
        help="No descarta filas con content vacio o con fallo de extraccion.",
    )
    parser.add_argument(
        "--min-content-chars",
        type=int,
        default=200,
        help="Numero minimo de caracteres en content para conservar una fila.",
    )
    parser.add_argument(
        "--disable-alias-filter",
        action="store_true",
        help="No exige que aparezca un alias del politico en title o content.",
    )


def namespace_to_build_settings(namespace: argparse.Namespace) -> BuildCorpusSettings:
    """Map parsed CLI arguments to the corpus-build settings object."""

    return BuildCorpusSettings(
        output_dir=Path(namespace.output_dir),
        timespan=namespace.timespan,
        max_records=namespace.max_records,
        sleep_seconds=namespace.sleep_seconds,
        request_timeout=namespace.request_timeout,
        retries=namespace.retries,
        gdelt_min_interval=namespace.gdelt_min_interval,
        max_politicians=namespace.max_politicians,
        checkpoint_every=namespace.checkpoint_every,
    )


def namespace_to_prepare_settings(namespace: argparse.Namespace) -> PrepareDatasetSettings:
    """Map parsed CLI arguments to the dataset preparation settings object."""

    return PrepareDatasetSettings(
        input_path=Path(namespace.input),
        output_dir=Path(namespace.output_dir),
        keep_neutral=namespace.keep_neutral,
        allow_empty_content=namespace.allow_empty_content,
        min_content_chars=namespace.min_content_chars,
        use_alias_filter=not namespace.disable_alias_filter,
    )


def run_build_corpus(namespace: argparse.Namespace) -> int:
    """Execute the corpus builder and print the output file locations."""

    output_jsonl, output_csv, summary_path = build_corpus(namespace_to_build_settings(namespace))
    print(f"[OK] EN JSONL: {output_jsonl}")
    print(f"[OK] EN CSV:   {output_csv}")
    print(f"[OK] META:     {summary_path}")
    return 0


def run_prepare_dataset(namespace: argparse.Namespace) -> int:
    """Execute the dataset preparation layer and print its outputs."""

    output_jsonl, output_csv, summary_path, total_rows = prepare_dataset(namespace_to_prepare_settings(namespace))
    print(f"[OK] CLEAN JSONL: {output_jsonl}")
    print(f"[OK] CLEAN CSV:   {output_csv}")
    print(f"[OK] META:        {summary_path}")
    print(f"[INFO] Filas finales: {total_rows}")
    return 0


def build_main_parser() -> argparse.ArgumentParser:
    """Create the main multi-command parser used by `main.py`."""

    parser = argparse.ArgumentParser(
        description="Herramientas para construir y evolucionar el proyecto de corpus politico.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-corpus",
        help="Descarga noticias desde GDELT, extrae contenido y genera el corpus local.",
    )
    add_build_arguments(build_parser)
    build_parser.set_defaults(handler=run_build_corpus)

    prepare_parser = subparsers.add_parser(
        "prepare-dataset",
        help="Limpia, deduplica y filtra el corpus crudo para dejarlo listo para modelado.",
    )
    add_prepare_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=run_prepare_dataset)
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the modern subcommand-based CLI."""

    parser = build_main_parser()
    namespace = parser.parse_args(argv)
    handler = namespace.handler
    return handler(namespace)