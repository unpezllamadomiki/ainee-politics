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
    test_texts: list[str] | None = None,
    test_labels: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Train TF-IDF (1-2gram) + LinearSVC and evaluate with stratified CV.

    If test_texts/test_labels are provided (shared global split), also evaluates
    on that held-out set so the comparison with the transformer is apple-to-apple.

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

    cv_scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring="f1_macro")
    y_pred = list(cross_val_predict(pipeline, texts, labels, cv=cv))

    acc = accuracy_score(labels, y_pred)
    report = classification_report(labels, y_pred, output_dict=True, zero_division=0)

    label_names = [l for l in _LABEL_ORDER if l in set(labels)]
    cm = confusion_matrix(labels, y_pred, labels=label_names, normalize="true")
    cm_path = output_dir / "confusion_matrix_classical.png"
    _save_confusion_matrix(cm, label_names, cm_path, "TF-IDF (1-2gram) + LinearSVC")

    # Fit on train split, evaluate on held-out test set if provided
    pipeline.fit(texts, labels)

    test_metrics: dict[str, Any] = {}
    test_preds: list[str] = []
    if test_texts and test_labels:
        test_preds = list(pipeline.predict(test_texts))
        t_acc = accuracy_score(test_labels, test_preds)
        t_f1  = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        t_rep = classification_report(test_labels, test_preds, output_dict=True, zero_division=0)
        test_metrics = {
            "test_set_accuracy":               round(float(t_acc), 4),
            "test_set_f1_macro":               round(float(t_f1),  4),
            "test_set_size":                   len(test_texts),
            "test_set_classification_report":  t_rep,
        }

    # Refit on all data (train + test) for the inference model saved to disk
    import joblib
    all_texts  = texts  + (test_texts  or [])
    all_labels = labels + (test_labels or [])
    pipeline.fit(all_texts, all_labels)
    model_path = output_dir / "classical_model.joblib"
    joblib.dump(pipeline, model_path)

    results: dict[str, Any] = {
        "model": "TF-IDF (1-2gram) + LinearSVC (class_weight=balanced)",
        "mode": f"{cv_folds}-fold stratified cross-validation (train split)",
        "cv_folds": cv_folds,
        "f1_macro_mean": round(float(cv_scores.mean()), 4),
        "f1_macro_std":  round(float(cv_scores.std()),  4),
        "accuracy":      round(float(acc), 4),
        "classification_report": report,
        "confusion_matrix_path": str(cm_path),
        "model_saved_to": str(model_path),
        **test_metrics,
    }
    return results, y_pred, test_preds


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
    short_name = model_name.split("/")[-1]
    _save_confusion_matrix(cm, label_names, cm_path, f"{short_name} (zero-shot)")

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
# Transformer model (fine-tuned, GPU/CPU)
# ---------------------------------------------------------------------------

_LABEL2ID = {"negative": 0, "positive": 1}
_ID2LABEL = {0: "negative", 1: "positive"}


def train_transformer_finetuned(
    texts: list[str],
    labels: list[str],
    model_name: str,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    test_size: float = 0.2,
    text_max_chars: int = 1500,
    provided_train_idx: list[int] | None = None,
    provided_test_idx: list[int] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Fine-tune a HuggingFace sequence classifier on the corpus.

    Auto-detects CUDA. Splits texts/labels into train/test (stratified), trains
    for `epochs` epochs, evaluates on the held-out test split, then runs
    inference on the full corpus so per-politician stats remain aligned.

    Saves the fine-tuned model + tokenizer to output_dir/finetuned_model/.
    Returns (metrics_dict, y_pred_full) where y_pred_full aligns with all inputs.
    """
    import torch
    from sklearn.model_selection import train_test_split as _split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from torch.utils.data import Dataset as _TorchDataset

    class _DS(_TorchDataset):
        def __init__(self, encodings: Any, int_labels: list[int]) -> None:
            self.encodings = encodings
            self.labels = int_labels

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self) -> int:
            return len(self.labels)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    use_cpu = device_name == "cpu"
    print(f"      Dispositivo: {device_name.upper()}")
    print(f"      Cargando tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    truncated = [t[:text_max_chars] for t in texts]
    int_labels = [_LABEL2ID[l] for l in labels]

    if provided_train_idx is not None and provided_test_idx is not None:
        train_idx, test_idx = provided_train_idx, provided_test_idx
    else:
        indices = list(range(len(truncated)))
        train_idx, test_idx = _split(indices, test_size=test_size, stratify=int_labels, random_state=42)

    X_train = [truncated[i] for i in train_idx]
    X_test  = [truncated[i] for i in test_idx]
    y_train = [int_labels[i] for i in train_idx]
    y_test  = [int_labels[i] for i in test_idx]
    y_test_str = [_ID2LABEL[l] for l in y_test]

    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"      Tokenizando...")

    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=512)
    test_enc  = tokenizer(X_test,  truncation=True, padding=True, max_length=512)
    train_ds  = _DS(train_enc, y_train)
    test_ds   = _DS(test_enc,  y_test)

    print(f"      Cargando modelo '{model_name}'...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=_ID2LABEL, label2id=_LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    model_save_path = output_dir / "finetuned_model"
    checkpoints_dir = output_dir / "finetune_checkpoints"

    def _compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, lbls = eval_pred
        preds = np.argmax(logits, axis=-1)
        pred_strs = [_ID2LABEL[int(p)] for p in preds]
        true_strs = [_ID2LABEL[int(l)] for l in lbls]
        return {
            "accuracy": float(accuracy_score(true_strs, pred_strs)),
            "f1": float(f1_score(true_strs, pred_strs, average="macro", zero_division=0)),
        }

    common_args: dict[str, Any] = dict(
        output_dir=str(checkpoints_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=lr,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        seed=42,
        fp16=(not use_cpu),
        report_to="none",
    )
    # Handle transformers API rename: evaluation_strategy → eval_strategy (>=4.46)
    # Also handle fp16 not supported on CPU in some versions
    def _make_training_args(extra: dict[str, Any]) -> TrainingArguments:
        merged = {**common_args, **extra}
        try:
            return TrainingArguments(**merged)
        except (TypeError, ValueError):
            merged.pop("fp16", None)
            return TrainingArguments(**merged)

    try:
        training_args = _make_training_args({"eval_strategy": "epoch"})
    except TypeError:
        training_args = _make_training_args({"evaluation_strategy": "epoch"})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=_compute_metrics,
    )

    print(f"      Fine-tuning ({epochs} epochs, lr={lr})...")
    trainer.train()

    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))
    print(f"      Modelo guardado en {model_save_path}")

    # Evaluate on held-out test split
    test_out     = trainer.predict(test_ds)
    test_pred_ids = np.argmax(test_out.predictions, axis=-1)
    y_pred_test  = [_ID2LABEL[int(p)] for p in test_pred_ids]

    acc    = accuracy_score(y_test_str, y_pred_test)
    f1     = f1_score(y_test_str, y_pred_test, average="macro", zero_division=0)
    report = classification_report(y_test_str, y_pred_test, output_dict=True, zero_division=0)

    label_names = [l for l in _LABEL_ORDER if l in set(y_test_str)]
    cm      = confusion_matrix(y_test_str, y_pred_test, labels=label_names, normalize="true")
    cm_path = output_dir / "confusion_matrix_transformer.png"
    short_name = model_name.split("/")[-1]
    _save_confusion_matrix(cm, label_names, cm_path, f"{short_name} (fine-tuned)")

    # Full-corpus predictions so per-politician stats align with all rows
    print(f"      Infiriendo corpus completo para estadísticas por político...")
    full_enc = tokenizer(truncated, truncation=True, padding=True, max_length=512)
    full_ds  = _DS(full_enc, int_labels)
    full_out = trainer.predict(full_ds)
    y_pred_full = [_ID2LABEL[int(p)] for p in np.argmax(full_out.predictions, axis=-1)]

    results: dict[str, Any] = {
        "model": model_name,
        "mode": f"fine-tuned ({epochs} epochs, {device_name.upper()}, test={int(test_size * 100)}%)",
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "classification_report": report,
        "confusion_matrix_path": str(cm_path),
        "model_saved_to": str(model_save_path),
    }
    return results, y_pred_full


# ---------------------------------------------------------------------------
# Cross-politician (leave-one-politician-out) evaluation
# ---------------------------------------------------------------------------

def cross_politician_eval(
    texts: list[str],
    labels: list[str],
    politicians: list[str],
    max_features: int,
    cv_per_pol: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Leave-one-politician-out evaluation using TF-IDF + LinearSVC.

    For each politician, trains on all other politicians and tests on this one.
    Compares the out-of-distribution accuracy against the within-distribution
    accuracy (cv_per_pol) to reveal how much the model relies on politician
    identity as a shortcut rather than genuine tone signals.
    """
    unique_pols = sorted(set(politicians))
    results: dict[str, Any] = {}
    f1_scores: list[float] = []

    pipeline_proto = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("clf", LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)),
    ])

    for test_pol in unique_pols:
        mask_test  = [p == test_pol for p in politicians]
        mask_train = [not m for m in mask_test]

        X_train = [t for t, m in zip(texts,  mask_train) if m]
        y_train = [l for l, m in zip(labels, mask_train) if m]
        X_test  = [t for t, m in zip(texts,  mask_test)  if m]
        y_test  = [l for l, m in zip(labels, mask_test)  if m]

        if len(X_test) < 5 or len(set(y_train)) < 2:
            continue

        import copy
        clf = copy.deepcopy(pipeline_proto)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_scores.append(f1)

        entry: dict[str, Any] = {
            "lopo_accuracy": round(float(acc), 3),
            "lopo_f1_macro": round(float(f1),  3),
            "n_test":        len(X_test),
        }
        if cv_per_pol and test_pol in cv_per_pol:
            within = cv_per_pol[test_pol]["accuracy"]
            entry["within_dist_accuracy"] = within
            entry["generalization_drop"]  = round(within - float(acc), 3)

        results[test_pol] = entry

    return {
        "per_politician": results,
        "mean_lopo_f1":   round(float(np.mean(f1_scores)), 4) if f1_scores else None,
        "interpretation": (
            "El modelo generaliza bien a políticos no vistos."
            if f1_scores and np.mean(f1_scores) >= 0.65
            else "El modelo depende de señales específicas por político — generalización limitada."
        ),
    }


# ---------------------------------------------------------------------------
# Per-politician breakdown
# ---------------------------------------------------------------------------

def per_politician_stats(
    valid_rows: list[ArticleRow],
    true_labels: list[str],
    y_pred: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute per-politician accuracy and label distribution using the supplied true labels."""
    buckets: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "pos": 0, "neg": 0})
    for row, true, pred in zip(valid_rows, true_labels, y_pred):
        pol = row["politician"]
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
    transformer_label: str = "",
) -> Path:
    """Save a two-panel comparison figure (overall metrics + per-politician accuracy)."""
    if not transformer_label:
        short = transformer.get("model", "Transformer").split("/")[-1]
        mode = transformer.get("mode", "")
        transformer_label = f"{short} (fine-tuned)" if "fine-tuned" in mode else f"{short} (zero-shot)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: overall accuracy and F1
    metric_labels = ["Accuracy", "F1-Macro"]
    c_vals = [classical["accuracy"], classical.get("f1_macro_mean", 0.0)]
    t_vals = [transformer["accuracy"], transformer.get("f1_macro", 0.0)]

    x = np.arange(len(metric_labels))
    w = 0.35
    bars_c = axes[0].bar(x - w / 2, c_vals, w, label="TF-IDF + LinearSVC", color="steelblue")
    bars_t = axes[0].bar(x + w / 2, t_vals, w, label=transformer_label, color="darkorange")
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
        axes[1].barh(y_idx + w / 2, t_acc, w, label=transformer_label, color="darkorange")
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

    plt.suptitle(f"Comparación: TF-IDF + LinearSVC  vs  {transformer_label}", fontsize=13)
    plt.tight_layout()
    path = output_dir / "comparison_plot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Bias landscape plot  (primary research output)
# ---------------------------------------------------------------------------

_TONE_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
    "no_politician_sentences": "#dfe6e9",
}
_TONE_ORDER = ["positive", "negative", "neutral", "no_politician_sentences"]


def save_bias_landscape_plot(
    all_rows: list[ArticleRow],
    tone_field: str,
    output_dir: Path,
) -> Path:
    """Stacked horizontal bar chart: tone distribution per politician.

    Uses ALL rows in the labeled corpus (including neutral / no_politician_sentences),
    not just the training subset.  Sorted by negative percentage descending so the
    most negatively covered politicians appear at the top.

    This is the primary bias-detection output of the pipeline.
    """
    # Aggregate counts
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {t: 0 for t in _TONE_ORDER})
    for row in all_rows:
        pol = row.get("politician")
        label = row.get(tone_field) or "no_politician_sentences"
        if pol and label in _TONE_ORDER:
            buckets[pol][label] += 1

    if not buckets:
        return output_dir / "bias_landscape.png"

    # Sort by negative ratio (descending)
    def _neg_ratio(pol: str) -> float:
        total = sum(buckets[pol].values())
        return buckets[pol]["negative"] / total if total else 0.0

    politicians_sorted = sorted(buckets.keys(), key=_neg_ratio, reverse=True)
    totals = [sum(buckets[p].values()) for p in politicians_sorted]
    n_pols = len(politicians_sorted)

    fig, ax = plt.subplots(figsize=(11, max(5, n_pols * 0.55)))
    y = np.arange(n_pols)
    left = np.zeros(n_pols)

    for tone in _TONE_ORDER:
        vals = np.array([buckets[p][tone] for p in politicians_sorted], dtype=float)
        pcts = np.where(np.array(totals) > 0, vals / np.array(totals) * 100, 0.0)
        ax.barh(y, pcts, left=left, color=_TONE_COLORS[tone], label=tone, edgecolor="white", linewidth=0.4)
        for i, (pct, lft) in enumerate(zip(pcts, left)):
            if pct >= 8:
                ax.text(
                    lft + pct / 2, i, f"{pct:.0f}%",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if tone in ("positive", "negative") else "#555",
                )
        left = left + pcts

    # Annotate total n on the right
    for i, (pol, tot) in enumerate(zip(politicians_sorted, totals)):
        ax.text(101, i, f"n={tot}", va="center", fontsize=7.5, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels(politicians_sorted, fontsize=9)
    ax.set_xlim(0, 113)
    ax.set_xlabel("% de artículos")
    ax.set_title(
        f"Distribución de tono por político  ({tone_field})\n"
        "Ordenado por % negativo ↓  —  incluye artículos excluidos del entrenamiento",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax.axvline(50, color="#bdc3c7", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    path = output_dir / "bias_landscape.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Label agreement
# ---------------------------------------------------------------------------

def compute_label_agreement(all_rows: list[ArticleRow]) -> dict[str, Any] | None:
    """Compare politician_tone_label vs gdelt_tone_label on rows that have both.

    Returns None if fewer than 5 comparable rows exist.
    A low agreement rate means GDELT article-level tone is a poor proxy for
    how the politician is specifically framed — which validates the labeling step.
    """
    _BINARY = {"positive", "negative"}
    comparable = [
        r for r in all_rows
        if r.get("politician_tone_label") in _BINARY
        and r.get("gdelt_tone_label") in _BINARY
    ]
    if len(comparable) < 5:
        return None

    agree = sum(
        1 for r in comparable
        if r["politician_tone_label"] == r["gdelt_tone_label"]
    )
    disagree_examples = [
        {
            "politician": r["politician"],
            "gdelt": r["gdelt_tone_label"],
            "politician_tone": r["politician_tone_label"],
            "title": r.get("title", "")[:80],
        }
        for r in comparable
        if r["politician_tone_label"] != r["gdelt_tone_label"]
    ][:5]

    return {
        "n_comparable": len(comparable),
        "n_agree": agree,
        "agreement_rate": round(agree / len(comparable), 3),
        "interpretation": (
            "Alta coincidencia: GDELT es buen proxy del tono hacia el político."
            if agree / len(comparable) >= 0.70
            else "Baja coincidencia: la etiqueta por político aporta información distinta al tono GDELT."
        ),
        "disagreement_examples": disagree_examples,
    }


# ---------------------------------------------------------------------------
# LLM (Ollama) zero-shot evaluation
# ---------------------------------------------------------------------------

def evaluate_llm(
    texts: list[str],
    labels: list[str],
    politicians: list[str],
    model_name: str,
    output_dir: Path,
    text_max_chars: int = 1500,
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate a local Ollama LLM as a zero-shot tone classifier.

    For each article, sends a prompt asking whether the article portrays the
    given politician positively or negatively.  Uses temperature=0 for
    deterministic output.  Returns (metrics_dict, y_pred).
    """
    import ollama

    _PROMPT = (
        "You are a media bias analyst. Read the news article below and classify "
        "whether it portrays {politician} in a positive or negative light.\n\n"
        "Article:\n{text}\n\n"
        "Respond with exactly one word: positive or negative."
    )

    y_pred: list[str] = []
    n_failed = 0
    n = len(texts)

    for i, (text, politician) in enumerate(zip(texts, politicians), 1):
        prompt = _PROMPT.format(politician=politician, text=text[:text_max_chars])
        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw = resp["message"]["content"].strip().lower()
            if "positive" in raw:
                y_pred.append("positive")
            elif "negative" in raw:
                y_pred.append("negative")
            else:
                y_pred.append("negative")
                n_failed += 1
        except Exception:
            y_pred.append("negative")
            n_failed += 1

        if i % 20 == 0 or i == n:
            print(f"      {i}/{n} artículos procesados...")

    acc    = accuracy_score(labels, y_pred)
    f1     = f1_score(labels, y_pred, average="macro", zero_division=0)
    report = classification_report(labels, y_pred, output_dict=True, zero_division=0)

    label_names = [l for l in _LABEL_ORDER if l in set(labels)]
    cm      = confusion_matrix(labels, y_pred, labels=label_names, normalize="true")
    cm_path = output_dir / "confusion_matrix_llm.png"
    _save_confusion_matrix(cm, label_names, cm_path, f"{model_name} (zero-shot LLM)")

    results: dict[str, Any] = {
        "model":              model_name,
        "mode":               "zero-shot LLM via Ollama (temperature=0)",
        "accuracy":           round(float(acc), 4),
        "f1_macro":           round(float(f1),  4),
        "n_evaluated":        n,
        "n_failed_parse":     n_failed,
        "classification_report": report,
        "confusion_matrix_path": str(cm_path),
    }
    return results, y_pred


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
