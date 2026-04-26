"""CLI entrypoints for the project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ainee_politics.application import build_corpus, compare_llm, label_corpus, prepare_dataset, train_model
from ainee_politics.config import API_RETRIES, GDELT_MIN_INTERVAL_SECONDS, REQUEST_TIMEOUT
from ainee_politics.domain.models import (
    BuildCorpusSettings,
    LabelSettings,
    LLMCompareSettings,
    PrepareDatasetSettings,
    TrainingSettings,
)


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


def add_label_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default="data/corpus_politicos_clean.jsonl", help="Ruta del corpus limpio en JSONL.")
    parser.add_argument("--output-dir", default="data", help="Directorio de salida.")
    parser.add_argument("--spacy-model", default="en_core_web_lg", help="Nombre del modelo spaCy a usar.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size para nlp.pipe().")


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        default="data/corpus_politicos_clean.jsonl",
        help="JSONL de entrada (corpus limpio o etiquetado).",
    )
    parser.add_argument("--output-dir", default="data", help="Directorio de salida para reportes y gráficas.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Número de folds para validación cruzada estratificada.")
    parser.add_argument(
        "--max-features",
        type=int,
        default=10_000,
        help="Número máximo de features TF-IDF.",
    )
    parser.add_argument(
        "--transformer-model",
        default="distilbert-base-uncased",
        help="Identificador HuggingFace del modelo transformer base para fine-tuning.",
    )
    parser.add_argument(
        "--text-max-chars",
        type=int,
        default=1500,
        help="Caracteres máximos del texto enviados al transformer.",
    )
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        help="Desactiva el fine-tuning y usa el modelo transformer en modo zero-shot.",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=3,
        help="Número de epochs para el fine-tuning del transformer.",
    )
    parser.add_argument(
        "--finetune-batch-size",
        type=int,
        default=16,
        help="Batch size para fine-tuning del transformer.",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=2e-5,
        help="Learning rate para fine-tuning del transformer.",
    )
    parser.add_argument(
        "--finetune-test-size",
        type=float,
        default=0.2,
        help="Fracción de datos reservados para el test set durante el fine-tuning (0.0-1.0).",
    )


def namespace_to_label_settings(namespace: argparse.Namespace) -> LabelSettings:
    return LabelSettings(
        input_path=Path(namespace.input),
        output_dir=Path(namespace.output_dir),
        spacy_model=namespace.spacy_model,
        batch_size=namespace.batch_size,
    )


def namespace_to_train_settings(namespace: argparse.Namespace) -> TrainingSettings:
    return TrainingSettings(
        input_path=Path(namespace.input),
        output_dir=Path(namespace.output_dir),
        cv_folds=namespace.cv_folds,
        max_tfidf_features=namespace.max_features,
        transformer_model=namespace.transformer_model,
        text_max_chars=namespace.text_max_chars,
        finetune=not namespace.no_finetune,
        finetune_epochs=namespace.finetune_epochs,
        finetune_batch_size=namespace.finetune_batch_size,
        finetune_lr=namespace.finetune_lr,
        finetune_test_size=namespace.finetune_test_size,
    )


def run_label_corpus(namespace: argparse.Namespace) -> int:
    out_jsonl, out_csv = label_corpus(namespace_to_label_settings(namespace))
    print(f"[OK] LABELED JSONL: {out_jsonl}")
    print(f"[OK] LABELED CSV:   {out_csv}")
    return 0


def run_train_model(namespace: argparse.Namespace) -> int:
    report_path = train_model(namespace_to_train_settings(namespace))
    print(f"[OK] REPORT: {report_path}")
    return 0


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default="data/corpus_labeled.jsonl", help="JSONL de entrada (corpus etiquetado).")
    parser.add_argument("--output-dir", default="data", help="Directorio de salida.")
    parser.add_argument("--ollama-model", default="llama3.1:8b", help="Nombre del modelo Ollama a usar.")
    parser.add_argument("--text-max-chars", type=int, default=1500, help="Caracteres máximos del texto enviados al LLM.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fracción del dataset usada como test (debe coincidir con train-model).")


def namespace_to_llm_settings(namespace: argparse.Namespace) -> LLMCompareSettings:
    return LLMCompareSettings(
        input_path=Path(namespace.input),
        output_dir=Path(namespace.output_dir),
        ollama_model=namespace.ollama_model,
        text_max_chars=namespace.text_max_chars,
        test_size=namespace.test_size,
    )


def run_compare_llm(namespace: argparse.Namespace) -> int:
    report_path = compare_llm(namespace_to_llm_settings(namespace))
    print(f"[OK] REPORT: {report_path}")
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

    label_parser = subparsers.add_parser(
        "label-corpus",
        help="Enriquece el corpus limpio con anotaciones spaCy: NER, adjetivos por político, estadísticas de frases.",
    )
    add_label_arguments(label_parser)
    label_parser.set_defaults(handler=run_label_corpus)

    train_parser = subparsers.add_parser(
        "train-model",
        help="Entrena y evalúa TF-IDF+LinearSVC vs DistilBERT (zero-shot) para clasificación de tono.",
    )
    add_train_arguments(train_parser)
    train_parser.set_defaults(handler=run_train_model)

    llm_parser = subparsers.add_parser(
        "compare-llm",
        help="Evalúa un LLM local (Ollama) en el mismo test set que train-model y añade resultados al reporte.",
    )
    add_llm_arguments(llm_parser)
    llm_parser.set_defaults(handler=run_compare_llm)

    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the modern subcommand-based CLI."""

    parser = build_main_parser()
    namespace = parser.parse_args(argv)
    handler = namespace.handler
    return handler(namespace)