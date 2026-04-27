"""train-model use case: train and compare classical vs transformer tone classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ainee_politics.domain.models import TrainingSettings
from ainee_politics.infrastructure.nlp.classifier import (
    compute_label_agreement,
    cross_politician_eval,
    evaluate_transformer,
    per_politician_stats,
    save_bias_landscape_plot,
    save_comparison_plot,
    train_classical,
    train_transformer_finetuned,
)
from ainee_politics.infrastructure.storage.dataset_store import ensure_output_dir, read_jsonl

_VALID_LABELS = {"positive", "negative"}
_SEP = "=" * 62


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def train_model(settings: TrainingSettings) -> Path:
    """Train TF-IDF+LinearSVC and evaluate DistilBERT (zero-shot) on tone classification.

    Uses article-level GDELT tone by default:
    - ``gdelt_tone_label`` (preferred): positive/negative tone of the full article.
    - ``politician_tone_label`` (fallback): VADER on politician-mentioning sentences
      when GDELT tone is unavailable.

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

    # Use article-level tone by default; only fall back if GDELT tone is missing.
    has_gdelt_tone = any(
        r.get("gdelt_tone_label") in _VALID_LABELS for r in rows
    )
    if has_gdelt_tone:
        tone_field = "gdelt_tone_label"
        print(
            "[INFO] Usando 'gdelt_tone_label' (tono del artículo completo según GDELT).\n"
            "       Esta etiqueta mide si la noticia tiene tono positivo o negativo en conjunto."
        )
    else:
        tone_field = "politician_tone_label"
        print(
            "[AVISO] 'gdelt_tone_label' no encontrado — usando 'politician_tone_label' como respaldo.\n"
            "        Ejecuta primero el pipeline completo de corpus para entrenar con tono de noticia."
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
    # Global 80/20 split — shared by both models for fair comparison
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split as _global_split

    all_indices = list(range(len(valid)))
    train_idx, test_idx = _global_split(
        all_indices,
        test_size=settings.finetune_test_size,
        stratify=labels,
        random_state=42,
    )
    train_texts  = [texts[i]  for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_texts   = [texts[i]  for i in test_idx]
    test_labels  = [labels[i] for i in test_idx]

    print(f"\n  Split global: {len(train_texts)} train / {len(test_texts)} test (mismo para ambos modelos)")

    # ------------------------------------------------------------------
    # Model 1: classical TF-IDF + LinearSVC
    # ------------------------------------------------------------------
    print(f"\n{_SEP}")
    print("  MODELO 1 / 2 — TF-IDF (1-2gram) + LinearSVC")
    print(f"  {settings.cv_folds}-fold CV en train split + evaluación en test compartido")
    print(_SEP)

    classical_results, classical_preds, classical_test_preds = train_classical(
        texts=train_texts,
        labels=train_labels,
        cv_folds=settings.cv_folds,
        max_features=settings.max_tfidf_features,
        output_dir=settings.output_dir,
        test_texts=test_texts,
        test_labels=test_labels,
    )
    # Reconstruct predictions in original order (train_idx + test_idx ≠ 0..N)
    classical_preds_aligned = [""] * len(valid)
    for orig_i, pred in zip(train_idx, classical_preds):
        classical_preds_aligned[orig_i] = pred
    for orig_i, pred in zip(test_idx, classical_test_preds):
        classical_preds_aligned[orig_i] = pred
    classical_per_pol = per_politician_stats(valid, labels, classical_preds_aligned)
    classical_results["per_politician"] = classical_per_pol
    _print_model_report(classical_results, per_pol=classical_per_pol)

    # ------------------------------------------------------------------
    # Model 2: transformer (fine-tuned o zero-shot)
    # ------------------------------------------------------------------
    print(f"\n{_SEP}")
    print(f"  MODELO 2 / 2 — {settings.transformer_model}")
    if settings.finetune:
        cuda_available = _has_cuda()
        device_str = "GPU (CUDA)" if cuda_available else "CPU"
        print(f"  Modo: fine-tuning  |  Dispositivo: {device_str}")
        print(f"  Epochs: {settings.finetune_epochs}  |  Batch: {settings.finetune_batch_size}  |  LR: {settings.finetune_lr}")
        print(f"  Test split: {int(settings.finetune_test_size * 100)}% (mismo split que TF-IDF)")
    else:
        print("  Modo: zero-shot (sin fine-tuning)")
    print(_SEP)

    if settings.finetune:
        transformer_results, transformer_preds_full, transformer_preds_test = train_transformer_finetuned(
            texts=texts,
            labels=labels,
            model_name=settings.transformer_model,
            output_dir=settings.output_dir,
            epochs=settings.finetune_epochs,
            batch_size=settings.finetune_batch_size,
            lr=settings.finetune_lr,
            test_size=settings.finetune_test_size,
            text_max_chars=settings.text_max_chars,
            provided_train_idx=train_idx,
            provided_test_idx=test_idx,
        )

        # Per-politician stats: prefer out-of-sample (test) for comparison/plotting
        test_rows = [valid[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        transformer_per_pol_test = per_politician_stats(test_rows, test_labels, transformer_preds_test)
        # Keep full-corpus (in-sample) stats for reference but mark them separately
        transformer_per_pol_full = per_politician_stats(valid, labels, transformer_preds_full)
        transformer_results["per_politician"] = transformer_per_pol_test
        transformer_results["per_politician_full"] = transformer_per_pol_full
        transformer_per_pol = transformer_per_pol_test
        _print_model_report(transformer_results, per_pol=transformer_per_pol_test)
    else:
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
    # Comparison (using shared test-set metrics)
    # ------------------------------------------------------------------
    comparison = _build_comparison(classical_results, transformer_results)
    _print_comparison(comparison)

    # Comparison plot
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
    # Cross-politician (leave-one-politician-out) evaluation
    # ------------------------------------------------------------------
    print(f"\n{_SEP}")
    print("  EVALUACIÓN CROSS-POLÍTICO (leave-one-politician-out)")
    print("  Entrena sin el político X, evalúa en X — mide generalización real")
    print(_SEP)
    pol_list = [r["politician"] for r in valid]
    lopo_results = cross_politician_eval(
        texts=texts,
        labels=labels,
        politicians=pol_list,
        max_features=settings.max_tfidf_features,
        cv_per_pol=classical_per_pol,
    )
    _print_lopo_results(lopo_results)

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
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        },
        "label_agreement_gdelt_vs_politician": agreement,
        "classical_model": classical_results,
        "transformer_model": transformer_results,
        "comparison": comparison,
        "cross_politician_eval": lopo_results,
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
    # Prefer shared test-set metrics for fair apples-to-apples comparison
    c_f1  = classical.get("test_set_f1_macro")  or classical.get("f1_macro_mean", 0.0)
    c_acc = classical.get("test_set_accuracy")   or classical.get("accuracy", 0.0)
    t_f1  = transformer.get("f1_macro",  0.0)
    t_acc = transformer.get("accuracy",  0.0)
    winner = "classical" if c_f1 >= t_f1 else "transformer"
    short = transformer.get("model", "Transformer").split("/")[-1]
    mode  = transformer.get("mode", "")
    t_label = f"{short} (fine-tuned)" if "fine-tuned" in mode else f"{short} (zero-shot)"
    return {
        "winner_by_f1_macro": winner,
        "transformer_label": t_label,
        "note": "Métricas comparadas sobre el mismo test set (20% compartido)",
        "classical_test_accuracy":  c_acc,
        "transformer_accuracy":     t_acc,
        "classical_test_f1_macro":  c_f1,
        "transformer_f1_macro":     t_f1,
        "classical_cv_f1_macro":    classical.get("f1_macro_mean", 0.0),
        "f1_macro_delta_classical_minus_transformer": round(c_f1 - t_f1, 4),
        "accuracy_delta_classical_minus_transformer": round(c_acc - t_acc, 4),
    }


def _print_comparison(comp: dict) -> None:
    delta_f1  = comp["f1_macro_delta_classical_minus_transformer"]
    delta_acc = comp["accuracy_delta_classical_minus_transformer"]
    winner    = comp["winner_by_f1_macro"].upper()
    t_label   = comp.get("transformer_label", "Transformer")

    print(f"\n{_SEP}")
    print("  COMPARACIÓN FINAL  (mismo test set 20% — comparación justa)")
    print(_SEP)
    print(f"  {'Modelo':<40} {'Accuracy':>10}  {'F1-Macro':>9}")
    print(f"  {'-'*60}")
    print(f"  {'TF-IDF + LinearSVC (test set)':<40} {comp['classical_test_accuracy']:>10.4f}  {comp['classical_test_f1_macro']:>9.4f}")
    print(f"  {'TF-IDF + LinearSVC (CV en train)':<40} {'':>10}  {comp['classical_cv_f1_macro']:>9.4f}")
    print(f"  {t_label:<40} {comp['transformer_accuracy']:>10.4f}  {comp['transformer_f1_macro']:>9.4f}")
    print(f"  {'-'*60}")
    sign = "+" if delta_f1 >= 0 else ""
    print(f"  Delta test (clásico - transformer)  F1={sign}{delta_f1:.4f}  Acc={sign}{delta_acc:.4f}")
    print(f"\n  >>> Ganador por F1-Macro (test): {winner} <<<")
    print(_SEP)


def _print_lopo_results(lopo: dict) -> None:
    per_pol = lopo.get("per_politician", {})
    mean_f1 = lopo.get("mean_lopo_f1")
    print(f"\n  F1-Macro medio LOPO: {mean_f1:.4f}" if mean_f1 else "")
    print(f"  {lopo.get('interpretation', '')}")
    print(f"\n  {'Político':<30} {'LOPO F1':>8} {'LOPO Acc':>9} {'Within Acc':>11} {'Drop':>6} {'n':>4}")
    print(f"  {'-'*70}")
    for pol, s in sorted(per_pol.items(), key=lambda x: x[1].get("generalization_drop", 0), reverse=True):
        drop = s.get("generalization_drop", "—")
        within = s.get("within_dist_accuracy", "—")
        drop_str  = f"{drop:>+.3f}" if isinstance(drop, float) else f"{'—':>6}"
        within_str = f"{within:.3f}" if isinstance(within, float) else "—"
        print(
            f"  {pol:<30} {s['lopo_f1_macro']:>8.3f} {s['lopo_accuracy']:>9.3f}"
            f" {within_str:>11} {drop_str:>6} {s['n_test']:>4}"
        )
    print(_SEP)
