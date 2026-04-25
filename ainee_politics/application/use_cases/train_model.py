"""train-model use case: train and compare classical vs transformer tone classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ainee_politics.domain.models import TrainingSettings
from ainee_politics.infrastructure.nlp.classifier import (
    evaluate_transformer,
    per_politician_stats,
    save_comparison_plot,
    train_classical,
)
from ainee_politics.infrastructure.storage.dataset_store import ensure_output_dir, read_jsonl

_VALID_LABELS = {"positive", "negative"}
_SEP = "=" * 62


def train_model(settings: TrainingSettings) -> Path:
    """Train TF-IDF+LinearSVC and evaluate DistilBERT (zero-shot) on tone classification.

    Both models predict ``gdelt_tone_label`` (positive | negative) from ``text``.
    Outputs a verbose console report, PNG confusion matrices, a comparison plot,
    and a JSON report to ``settings.output_dir/training_report.json``.
    """
    ensure_output_dir(settings.output_dir)

    # ------------------------------------------------------------------
    # Load and validate data
    # ------------------------------------------------------------------
    rows = read_jsonl(settings.input_path)
    if not rows:
        raise ValueError(f"No se encontraron filas en {settings.input_path}")

    valid = [r for r in rows if r.get("gdelt_tone_label") in _VALID_LABELS]
    if len(valid) < 20:
        raise ValueError(
            f"Solo {len(valid)} artículos con etiqueta positivo/negativo. "
            "Construye un corpus más grande antes de entrenar."
        )

    texts = [r.get("text") or r.get("content") or "" for r in valid]
    labels = [r["gdelt_tone_label"] for r in valid]

    label_counts = {l: labels.count(l) for l in _VALID_LABELS}
    politicians = sorted({r["politician"] for r in valid})
    pol_counts = {p: sum(1 for r in valid if r["politician"] == p) for p in politicians}

    _print_corpus_summary(len(valid), label_counts, pol_counts)

    # ------------------------------------------------------------------
    # Model 1: classical TF-IDF + LinearSVC
    # ------------------------------------------------------------------
    print(f"\n{_SEP}")
    print("  MODELO 1 / 2 — TF-IDF (1-2gram) + LinearSVC")
    print(f"  {settings.cv_folds}-fold stratified cross-validation")
    print(_SEP)

    classical_results, classical_preds = train_classical(
        texts=texts,
        labels=labels,
        cv_folds=settings.cv_folds,
        max_features=settings.max_tfidf_features,
        output_dir=settings.output_dir,
    )
    classical_per_pol = per_politician_stats(valid, classical_preds)
    classical_results["per_politician"] = classical_per_pol
    _print_model_report(classical_results, per_pol=classical_per_pol)

    # ------------------------------------------------------------------
    # Model 2: DistilBERT zero-shot
    # ------------------------------------------------------------------
    print(f"\n{_SEP}")
    print(f"  MODELO 2 / 2 — {settings.transformer_model}")
    print("  Zero-shot inference en CPU (sin fine-tuning)")
    print("  Nota: este modelo fue entrenado sobre SST-2 (reseñas de cine),")
    print("  no sobre noticias políticas — diferencia esperada de dominio.")
    print(_SEP)

    transformer_results, transformer_preds = evaluate_transformer(
        texts=texts,
        labels=labels,
        model_name=settings.transformer_model,
        output_dir=settings.output_dir,
        text_max_chars=settings.text_max_chars,
    )
    transformer_per_pol = per_politician_stats(valid, transformer_preds)
    transformer_results["per_politician"] = transformer_per_pol
    _print_model_report(transformer_results, per_pol=transformer_per_pol)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    comparison = _build_comparison(classical_results, transformer_results)
    _print_comparison(comparison)

    # Comparison plot (overall + per-politician)
    plot_path = save_comparison_plot(
        classical=classical_results,
        transformer=transformer_results,
        classical_per_pol=classical_per_pol,
        transformer_per_pol=transformer_per_pol,
        output_dir=settings.output_dir,
    )

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "corpus_stats": {
            "total_articles": len(valid),
            "class_distribution": label_counts,
            "politicians": pol_counts,
        },
        "classical_model": classical_results,
        "transformer_model": transformer_results,
        "comparison": comparison,
        "plots": {
            "comparison": str(plot_path),
            "confusion_classical": classical_results["confusion_matrix_path"],
            "confusion_transformer": transformer_results["confusion_matrix_path"],
        },
    }

    report_path = settings.output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"\n[OK] Reporte JSON  → {report_path}")
    print(f"[OK] Gráfica comp  → {plot_path}")
    print(f"[OK] CM clásico    → {classical_results['confusion_matrix_path']}")
    print(f"[OK] CM transformer → {transformer_results['confusion_matrix_path']}")
    return report_path


# ---------------------------------------------------------------------------
# Verbose console reporting helpers
# ---------------------------------------------------------------------------

def _print_corpus_summary(
    total: int,
    label_counts: dict[str, int],
    pol_counts: dict[str, int],
) -> None:
    print(f"\n{_SEP}")
    print("  RESUMEN DEL CORPUS")
    print(_SEP)
    print(f"  Total artículos (positive + negative): {total}")
    for label, count in sorted(label_counts.items()):
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"    {label:10s}: {count:5d}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Artículos por político:")
    for pol, n in sorted(pol_counts.items(), key=lambda x: -x[1]):
        print(f"    {pol:<30s}: {n}")


def _print_model_report(results: dict[str, Any], per_pol: dict[str, dict]) -> None:
    mode = results.get("mode", "")
    acc = results.get("accuracy", 0.0)
    f1 = results.get("f1_macro_mean") or results.get("f1_macro", 0.0)
    f1_std = results.get("f1_macro_std")

    print(f"\n  Accuracy  : {acc:.4f}")
    if f1_std is not None:
        print(f"  F1-Macro  : {f1:.4f}  ±  {f1_std:.4f}  (CV medio ± std)")
    else:
        print(f"  F1-Macro  : {f1:.4f}")

    report = results.get("classification_report", {})
    if report:
        print(f"\n  {'Clase':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
        print(f"  {'-'*52}")
        for label in ("negative", "positive"):
            if label in report:
                m = report[label]
                print(
                    f"  {label:<12} {m['precision']:>10.3f} {m['recall']:>8.3f}"
                    f" {m['f1-score']:>8.3f} {int(m['support']):>9}"
                )
        if "macro avg" in report:
            m = report["macro avg"]
            print(f"  {'macro avg':<12} {m['precision']:>10.3f} {m['recall']:>8.3f}"
                  f" {m['f1-score']:>8.3f}")
        if "weighted avg" in report:
            m = report["weighted avg"]
            print(f"  {'weighted avg':<12} {m['precision']:>10.3f} {m['recall']:>8.3f}"
                  f" {m['f1-score']:>8.3f}")

    if per_pol:
        print(f"\n  Accuracy por político:")
        print(f"  {'Político':<30} {'Acc':>6} {'n':>5} {'pos':>5} {'neg':>5}")
        print(f"  {'-'*55}")
        for pol, stats in per_pol.items():
            print(
                f"  {pol:<30} {stats['accuracy']:>6.3f} {stats['n']:>5}"
                f" {stats['positive_articles']:>5} {stats['negative_articles']:>5}"
            )

    print(f"\n  Confusion matrix → {results.get('confusion_matrix_path', '')}")


def _build_comparison(classical: dict, transformer: dict) -> dict[str, Any]:
    c_acc = classical.get("accuracy", 0.0)
    t_acc = transformer.get("accuracy", 0.0)
    c_f1 = classical.get("f1_macro_mean", 0.0)
    t_f1 = transformer.get("f1_macro", 0.0)
    winner = "classical" if c_f1 >= t_f1 else "transformer"
    return {
        "winner_by_f1_macro": winner,
        "classical_accuracy": c_acc,
        "transformer_accuracy": t_acc,
        "classical_f1_macro": c_f1,
        "transformer_f1_macro": t_f1,
        "f1_macro_delta_classical_minus_transformer": round(c_f1 - t_f1, 4),
        "accuracy_delta_classical_minus_transformer": round(c_acc - t_acc, 4),
    }


def _print_comparison(comp: dict) -> None:
    delta_f1 = comp["f1_macro_delta_classical_minus_transformer"]
    delta_acc = comp["accuracy_delta_classical_minus_transformer"]
    winner = comp["winner_by_f1_macro"].upper()

    print(f"\n{_SEP}")
    print("  COMPARACIÓN FINAL")
    print(_SEP)
    print(f"  {'Modelo':<40} {'Accuracy':>10}  {'F1-Macro':>9}")
    print(f"  {'-'*60}")
    print(f"  {'TF-IDF + LinearSVC':<40} {comp['classical_accuracy']:>10.4f}  {comp['classical_f1_macro']:>9.4f}")
    print(f"  {'DistilBERT (zero-shot)':<40} {comp['transformer_accuracy']:>10.4f}  {comp['transformer_f1_macro']:>9.4f}")
    print(f"  {'-'*60}")
    sign = "+" if delta_f1 >= 0 else ""
    print(f"  Delta (clásico - transformer)  F1={sign}{delta_f1:.4f}  Acc={sign}{delta_acc:.4f}")
    print(f"\n  >>> Ganador por F1-Macro: {winner} <<<")
    if winner == "CLASSICAL":
        print("  Interpretar: el modelo entrenado en dominio supera al transformer general.")
    else:
        print("  Interpretar: el transformer generaliza mejor que el modelo entrenado.")
    print(_SEP)
