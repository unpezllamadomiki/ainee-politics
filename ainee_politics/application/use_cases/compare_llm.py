"""compare-llm use case: evaluate a local Ollama LLM on the same test split used by train-model."""

from __future__ import annotations

import json
from pathlib import Path

from ainee_politics.domain.models import LLMCompareSettings
from ainee_politics.infrastructure.nlp.classifier import evaluate_llm, per_politician_stats
from ainee_politics.infrastructure.storage.dataset_store import ensure_output_dir, read_jsonl

_VALID_LABELS = {"positive", "negative"}
_SEP = "=" * 62


def compare_llm(settings: LLMCompareSettings) -> Path:
    """Evaluate a local Ollama LLM on the held-out test split.

    Recreates the exact same 80/20 stratified split used by ``train-model``
    (random_state=42) so the three models — TF-IDF, fine-tuned transformer,
    and LLM — are compared on identical examples.

    Appends results to the existing ``training_report.json``.
    """
    from sklearn.model_selection import train_test_split

    ensure_output_dir(settings.output_dir)

    rows = read_jsonl(settings.input_path)
    if not rows:
        raise ValueError(f"No se encontraron filas en {settings.input_path}")

    has_politician_tone = any(r.get("politician_tone_label") in _VALID_LABELS for r in rows)
    tone_field = "politician_tone_label" if has_politician_tone else "gdelt_tone_label"

    valid      = [r for r in rows if r.get(tone_field) in _VALID_LABELS]
    texts      = [r.get("text") or r.get("content") or "" for r in valid]
    labels     = [r[tone_field] for r in valid]
    politicians = [r["politician"] for r in valid]

    # Recreate the exact same split as train-model
    all_indices = list(range(len(valid)))
    _, test_idx = train_test_split(
        all_indices,
        test_size=settings.test_size,
        stratify=labels,
        random_state=42,
    )

    test_texts      = [texts[i]      for i in test_idx]
    test_labels     = [labels[i]     for i in test_idx]
    test_politicians = [politicians[i] for i in test_idx]
    test_rows       = [valid[i]      for i in test_idx]

    print(f"\n{_SEP}")
    print(f"  LLM — {settings.ollama_model}  (zero-shot via Ollama)")
    print(f"  Evaluando {len(test_texts)} artículos del test set")
    print(_SEP)

    llm_results, llm_preds = evaluate_llm(
        texts=test_texts,
        labels=test_labels,
        politicians=test_politicians,
        model_name=settings.ollama_model,
        output_dir=settings.output_dir,
        text_max_chars=settings.text_max_chars,
    )

    llm_per_pol = per_politician_stats(test_rows, test_labels, llm_preds)
    llm_results["per_politician"] = llm_per_pol

    _print_llm_report(llm_results, llm_per_pol)

    # Load existing report and append LLM section
    report_path = settings.output_dir / "training_report.json"
    report: dict = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    report["llm_model"] = llm_results

    # Update comparison block
    comp = report.get("comparison", {})
    c_f1 = comp.get("classical_test_f1_macro") or report.get("classical_model", {}).get("f1_macro_mean", 0)
    t_f1 = report.get("transformer_model", {}).get("f1_macro", 0)
    l_f1 = llm_results["f1_macro"]
    comp["llm_f1_macro"]              = l_f1
    comp["llm_vs_classical_delta"]    = round(l_f1 - c_f1, 4)
    comp["llm_vs_transformer_delta"]  = round(l_f1 - t_f1, 4)
    report["comparison"] = comp

    if "plots" not in report:
        report["plots"] = {}
    report["plots"]["confusion_llm"] = llm_results["confusion_matrix_path"]

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"\n[OK] Reporte actualizado → {report_path}")
    print(f"[OK] CM LLM             → {llm_results['confusion_matrix_path']}")
    return report_path


def _print_llm_report(results: dict, per_pol: dict) -> None:
    print(f"\n  Accuracy  : {results['accuracy']:.4f}")
    print(f"  F1-Macro  : {results['f1_macro']:.4f}")
    if results.get("n_failed_parse"):
        print(f"  Respuestas no parseadas: {results['n_failed_parse']} / {results['n_evaluated']}")

    cr = results.get("classification_report", {})
    if cr:
        print(f"\n  {'Clase':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
        print(f"  {'-'*52}")
        for lbl in ("negative", "positive", "macro avg"):
            m = cr.get(lbl, {})
            if not m:
                continue
            sup = int(m.get("support", 0)) if lbl != "macro avg" else "—"
            print(
                f"  {lbl:<12} {m['precision']:>10.3f} {m['recall']:>8.3f}"
                f" {m['f1-score']:>8.3f} {str(sup):>9}"
            )

    if per_pol:
        print(f"\n  Accuracy por político:")
        print(f"  {'Político':<30} {'Acc':>6} {'n':>5} {'pos':>5} {'neg':>5}")
        print(f"  {'-'*55}")
        for pol, s in per_pol.items():
            print(
                f"  {pol:<30} {s['accuracy']:>6.3f} {s['n']:>5}"
                f" {s['positive_articles']:>5} {s['negative_articles']:>5}"
            )
