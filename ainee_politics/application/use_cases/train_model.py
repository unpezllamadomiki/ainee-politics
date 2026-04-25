"""train-model use case: train and compare classical vs transformer tone classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ainee_politics.domain.models import TrainingSettings
from ainee_politics.infrastructure.nlp.classifier import (
    compute_label_agreement,
    evaluate_transformer,
    per_politician_stats,
    save_bias_landscape_plot,
    save_comparison_plot,
    train_classical,
)
from ainee_politics.infrastructure.storage.dataset_store import ensure_output_dir, read_jsonl

_VALID_LABELS = {"positive", "negative"}
_SEP = "=" * 62


def train_model(settings: TrainingSettings) -> Path:
    """Train TF-IDF+LinearSVC and evaluate DistilBERT (zero-shot) on tone classification.

    Auto-detects the best available tone label:
    - ``politician_tone_label`` (preferred): VADER on politician-mentioning sentences,
      produced by the ``label-corpus`` step.  More precise for bias detection.
    - ``gdelt_tone_label`` (fallback): article-level tone from GDELT.

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

    # Auto-detect tone label field
    has_politician_tone = any(
        r.get("politician_tone_label") in _VALID_LABELS for r in rows
    )
    if has_politician_tone:
        tone_field = "politician_tone_label"
        print(
            "[INFO] Usando 'politician_tone_label' (VADER sobre frases con el político).\n"
            "       Esta etiqueta refleja cómo se retrata al político específicamente."
        )
    else:
        tone_field = "gdelt_tone_label"
        print(
            "[AVISO] 'politician_tone_label' no encontrado — usando 'gdelt_tone_label' (tono del artículo completo).\n"
            "        Para mejor precisión, ejecuta primero: python main.py label-corpus"
        )

    valid = [r for r in rows if r.get(tone_field) in _VALID_LABELS]
    if len(valid) < 20:
        raise ValueError(
            f"Solo {len(valid)} artículos con etiqueta positivo/negativo en '{tone_field}'. "
            "Construye un corpus más grande antes de entrenar."
        )

    texts = [r.get("text") or r.get("content") or "" for r in valid]
    labels = [r[tone_field] for r in valid]

    label_counts = {l: labels.count(l) for l in _VALID_LABELS}
    politicians = sorted({r["politician"] for r in valid})
    pol_counts = {p: sum(1 for r in valid if r["politician"] == p) for p in politicians}

    # Full label distribution (all rows, including excluded)
    all_tone_labels = [r.get(tone_field, "no_politician_sentences") for r in rows]
    excluded = len(rows) - len(valid)
    full_label_dist: dict[str, int] = {}
    for lbl in all_tone_labels:
        full_label_dist[lbl] = full_label_dist.get(lbl, 0) + 1

    # Per-politician full distribution (for bias landscape and JSON report)
    full_pol_dist: dict[str, dict[str, int]] = {}
    for row in rows:
        pol = row.get("politician", "unknown")
        lbl = row.get(tone_field, "no_politician_sentences")
        if pol not in full_pol_dist:
            full_pol_dist[pol] = {}
        full_pol_dist[pol][lbl] = full_pol_dist[pol].get(lbl, 0) + 1

    _print_corpus_summary(len(valid), label_counts, pol_counts, tone_field, excluded, full_label_dist)

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
    classical_per_pol = per_politician_stats(valid, labels, classical_preds)
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
    transformer_per_pol = per_politician_stats(valid, labels, transformer_preds)
    transformer_results["per_politician"] = transformer_per_pol
    _print_model_report(transformer_results, per_pol=transformer_per_pol)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    comparison = _build_comparison(classical_results, transformer_results)
    _print_comparison(comparison)

    # Comparison plot (overall metrics + per-politician accuracy)
    plot_path = save_comparison_plot(
        classical=classical_results,
        transformer=transformer_results,
        classical_per_pol=classical_per_pol,
        transformer_per_pol=transformer_per_pol,
        output_dir=settings.output_dir,
    )

    # Bias landscape (full corpus — primary research output)
    landscape_path = save_bias_landscape_plot(
        all_rows=rows,
        tone_field=tone_field,
        output_dir=settings.output_dir,
    )

    # Label agreement between politician_tone and gdelt_tone
    agreement = compute_label_agreement(rows)
    if agreement:
        _print_label_agreement(agreement)

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "corpus_stats": {
            "tone_label_field": tone_field,
            "total_labeled_articles": len(rows),
            "used_for_training": len(valid),
            "excluded_from_training": excluded,
            "full_label_distribution": full_label_dist,
            "training_class_distribution": label_counts,
            "per_politician_full_distribution": full_pol_dist,
            "per_politician_training": pol_counts,
        },
        "label_agreement_gdelt_vs_politician": agreement,
        "classical_model": classical_results,
        "transformer_model": transformer_results,
        "comparison": comparison,
        "plots": {
            "bias_landscape": str(landscape_path),
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

    print(f"\n[OK] Sesgo por político → {landscape_path}")
    print(f"[OK] Comparación modelos → {plot_path}")
    print(f"[OK] CM clásico         → {classical_results['confusion_matrix_path']}")
    print(f"[OK] CM transformer      → {transformer_results['confusion_matrix_path']}")
    print(f"[OK] Reporte JSON        → {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Verbose console reporting helpers
# ---------------------------------------------------------------------------

def _print_corpus_summary(
    total_train: int,
    label_counts: dict[str, int],
    pol_counts: dict[str, int],
    tone_field: str,
    excluded: int,
    full_label_dist: dict[str, int],
) -> None:
    total_all = total_train + excluded
    print(f"\n{_SEP}")
    print("  RESUMEN DEL CORPUS")
    print(_SEP)
    print(f"  Etiqueta de tono        : {tone_field}")
    print(f"  Total artículos en fichero : {total_all}")
    print(f"  Usados en entrenamiento    : {total_train}  ({total_train/total_all*100:.1f}%)")
    print(f"  Excluidos (neutral/sin mención) : {excluded}  ({excluded/total_all*100:.1f}%)")
    print(f"\n  Distribución completa (todos los artículos):")
    for lbl in ("positive", "negative", "neutral", "no_politician_sentences"):
        count = full_label_dist.get(lbl, 0)
        pct = count / total_all * 100 if total_all else 0
        bar = "█" * int(pct / 2)
        print(f"    {lbl:<25s}: {count:5d}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Artículos de entrenamiento por político:")
    for pol, n in sorted(pol_counts.items(), key=lambda x: -x[1]):
        print(f"    {pol:<30s}: {n}")


def _print_label_agreement(agreement: dict) -> None:
    print(f"\n{_SEP}")
    print("  ACUERDO: politician_tone_label vs gdelt_tone_label")
    print(_SEP)
    rate = agreement["agreement_rate"]
    print(f"  Artículos comparables : {agreement['n_comparable']}")
    print(f"  Coinciden             : {agreement['n_agree']}  ({rate*100:.1f}%)")
    print(f"  No coinciden          : {agreement['n_comparable'] - agreement['n_agree']}  ({(1-rate)*100:.1f}%)")
    print(f"  → {agreement['interpretation']}")
    if agreement.get("disagreement_examples"):
        print(f"\n  Ejemplos de desacuerdo:")
        for ex in agreement["disagreement_examples"]:
            print(f"    [{ex['politician']}]  gdelt={ex['gdelt']}  politician_tone={ex['politician_tone']}")
            print(f"      \"{ex['title']}\"")
    print(_SEP)


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
