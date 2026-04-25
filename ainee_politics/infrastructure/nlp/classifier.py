"""Classical (TF-IDF + LinearSVC) and transformer (DistilBERT zero-shot) classifiers.

Both classifiers return (metrics_dict, y_pred) so the caller can compute
per-politician breakdowns without re-running inference.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ainee_politics.domain.models import ArticleRow

# Canonical label ordering for consistent confusion matrices
_LABEL_ORDER = ["negative", "positive"]

# HuggingFace label → our label
_HF_LABEL_MAP: dict[str, str] = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "LABEL_1": "positive",
    "LABEL_0": "negative",
}


# ---------------------------------------------------------------------------
# Classical model
# ---------------------------------------------------------------------------

def train_classical(
    texts: list[str],
    labels: list[str],
    cv_folds: int,
    max_features: int,
    output_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Train TF-IDF (1-2gram) + LinearSVC and evaluate with stratified CV.

    Returns (metrics_dict, y_pred_aligned_with_input).
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("clf", LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # F1-macro across folds for headline score
    cv_scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring="f1_macro")

    # Full predictions for per-class and per-politician analysis
    y_pred = list(cross_val_predict(pipeline, texts, labels, cv=cv))

    acc = accuracy_score(labels, y_pred)
    report = classification_report(labels, y_pred, output_dict=True, zero_division=0)

    label_names = [l for l in _LABEL_ORDER if l in set(labels)]
    cm = confusion_matrix(labels, y_pred, labels=label_names, normalize="true")
    cm_path = output_dir / "confusion_matrix_classical.png"
    _save_confusion_matrix(cm, label_names, cm_path, "TF-IDF (1-2gram) + LinearSVC")

    results: dict[str, Any] = {
        "model": "TF-IDF (1-2gram) + LinearSVC (class_weight=balanced)",
        "mode": f"{cv_folds}-fold stratified cross-validation",
        "cv_folds": cv_folds,
        "f1_macro_mean": round(float(cv_scores.mean()), 4),
        "f1_macro_std": round(float(cv_scores.std()), 4),
        "accuracy": round(float(acc), 4),
        "classification_report": report,
        "confusion_matrix_path": str(cm_path),
    }
    return results, y_pred


# ---------------------------------------------------------------------------
# Transformer model (zero-shot, CPU)
# ---------------------------------------------------------------------------

def evaluate_transformer(
    texts: list[str],
    labels: list[str],
    model_name: str,
    output_dir: Path,
    text_max_chars: int = 1500,
) -> tuple[dict[str, Any], list[str]]:
    """Run zero-shot inference with a HuggingFace text-classification pipeline.

    Uses device=-1 (CPU). No fine-tuning — the pre-trained model is used as-is.
    Returns (metrics_dict, y_pred_aligned_with_input).
    """
    from transformers import pipeline as hf_pipeline

    print(f"      Descargando/cargando '{model_name}' en CPU...")
    clf = hf_pipeline(
        "text-classification",
        model=model_name,
        device=-1,
        truncation=True,
        max_length=512,
    )

    # Truncate to limit CPU inference time on long articles
    truncated = [t[:text_max_chars] for t in texts]

    print(f"      Infiriendo {len(truncated)} artículos (batch_size=16)...")
    raw_preds = clf(truncated, batch_size=16)

    y_pred = [
        _HF_LABEL_MAP.get(p["label"].upper(), p["label"].lower())
        for p in raw_preds
    ]

    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average="macro", zero_division=0)
    report = classification_report(labels, y_pred, output_dict=True, zero_division=0)

    label_names = [l for l in _LABEL_ORDER if l in set(labels)]
    cm = confusion_matrix(labels, y_pred, labels=label_names, normalize="true")
    cm_path = output_dir / "confusion_matrix_transformer.png"
    _save_confusion_matrix(cm, label_names, cm_path, f"DistilBERT (zero-shot)")

    results: dict[str, Any] = {
        "model": model_name,
        "mode": "zero-shot inference (no fine-tuning, CPU)",
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1), 4),
        "classification_report": report,
        "confusion_matrix_path": str(cm_path),
    }
    return results, y_pred


# ---------------------------------------------------------------------------
# Per-politician breakdown
# ---------------------------------------------------------------------------

def per_politician_stats(
    valid_rows: list[ArticleRow],
    y_pred: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute per-politician accuracy and label distribution."""
    buckets: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "pos": 0, "neg": 0})
    for row, pred in zip(valid_rows, y_pred):
        pol = row["politician"]
        true = row["gdelt_tone_label"]
        buckets[pol]["total"] += 1
        if true == "positive":
            buckets[pol]["pos"] += 1
        else:
            buckets[pol]["neg"] += 1
        if true == pred:
            buckets[pol]["correct"] += 1

    return {
        pol: {
            "accuracy": round(v["correct"] / v["total"], 3),
            "n": v["total"],
            "positive_articles": v["pos"],
            "negative_articles": v["neg"],
        }
        for pol, v in sorted(buckets.items(), key=lambda x: -x[1]["total"])
    }


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def save_comparison_plot(
    classical: dict[str, Any],
    transformer: dict[str, Any],
    classical_per_pol: dict[str, dict],
    transformer_per_pol: dict[str, dict],
    output_dir: Path,
) -> Path:
    """Save a two-panel comparison figure (overall metrics + per-politician accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: overall accuracy and F1
    metric_labels = ["Accuracy", "F1-Macro"]
    c_vals = [classical["accuracy"], classical.get("f1_macro_mean", 0.0)]
    t_vals = [transformer["accuracy"], transformer.get("f1_macro", 0.0)]

    x = np.arange(len(metric_labels))
    w = 0.35
    bars_c = axes[0].bar(x - w / 2, c_vals, w, label="TF-IDF + LinearSVC", color="steelblue")
    bars_t = axes[0].bar(x + w / 2, t_vals, w, label="DistilBERT (zero-shot)", color="darkorange")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Métricas globales")
    axes[0].legend()
    for bar in (*bars_c, *bars_t):
        h = bar.get_height()
        axes[0].annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            ha="center", va="bottom", fontsize=8,
        )

    # Panel 2: per-politician accuracy (politicians with ≥5 articles)
    common_pols = sorted(
        {p for p, v in classical_per_pol.items() if v["n"] >= 5}
        & {p for p, v in transformer_per_pol.items() if v["n"] >= 5}
    )
    if common_pols:
        c_acc = [classical_per_pol[p]["accuracy"] for p in common_pols]
        t_acc = [transformer_per_pol[p]["accuracy"] for p in common_pols]
        y_idx = np.arange(len(common_pols))
        axes[1].barh(y_idx - w / 2, c_acc, w, label="TF-IDF + LinearSVC", color="steelblue")
        axes[1].barh(y_idx + w / 2, t_acc, w, label="DistilBERT (zero-shot)", color="darkorange")
        axes[1].set_yticks(y_idx)
        axes[1].set_yticklabels(common_pols, fontsize=8)
        axes[1].set_xlim(0, 1.1)
        axes[1].set_xlabel("Accuracy")
        axes[1].set_title("Accuracy por político (n ≥ 5)")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "Insuficientes datos\npor político (n<5)",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Accuracy por político")

    plt.suptitle("Comparación: Modelo Clásico vs Transformer (zero-shot)", fontsize=13)
    plt.tight_layout()
    path = output_dir / "comparison_plot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(ax=ax, xticks_rotation="horizontal", colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
