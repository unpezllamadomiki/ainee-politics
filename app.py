"""Streamlit dashboard: resultados de sesgo + predicción en tiempo real."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("data")
REPORT_PATH = DATA_DIR / "training_report.json"
TRANSFORMER_MODEL_DIR = DATA_DIR / "finetuned_model"
CLASSICAL_MODEL_PATH = DATA_DIR / "classical_model.joblib"
LABELED_CORPUS_PATH = DATA_DIR / "corpus_labeled.jsonl"
RAG_VECTOR_DIR = DATA_DIR / "chroma_politics_news"

st.set_page_config(
    page_title="Ainee Politics — Bias Dashboard",
    page_icon="📰",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_report() -> dict:
    if not REPORT_PATH.exists():
        return {}
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_transformer_pipeline():
    from transformers import pipeline as hf_pipeline
    import torch
    device = 0 if torch.cuda.is_available() else -1
    return hf_pipeline(
        "text-classification",
        model=str(TRANSFORMER_MODEL_DIR),
        device=device,
        truncation=True,
        max_length=512,
    )


@st.cache_resource
def load_classical_pipeline():
    import joblib
    if not CLASSICAL_MODEL_PATH.exists():
        return None
    return joblib.load(CLASSICAL_MODEL_PATH)


@st.cache_resource
def load_spacy():
    from ainee_politics.infrastructure.nlp.spacy_processor import load_spacy_model as _load_spacy
    return _load_spacy("en_core_web_lg")


def _get_aliases(politician_name: str) -> tuple[str, ...]:
    try:
        from ainee_politics.domain.catalog import DEFAULT_POLITICIANS
        for p in DEFAULT_POLITICIANS:
            if p.name == politician_name:
                return p.aliases
    except Exception:
        pass
    parts = politician_name.strip().split()
    return (politician_name, parts[-1]) if len(parts) > 1 else (politician_name,)


def _score_politician_sentences(text: str, aliases: tuple[str, ...], nlp):
    """Extract sentences mentioning the politician and score them with VADER."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_infer = SentimentIntensityAnalyzer()
    aliases_lower = {a.lower() for a in aliases}
    doc = nlp(text[:5_000])
    sents = [
        sent.text.strip()
        for sent in doc.sents
        if any(alias in sent.text.lower() for alias in aliases_lower)
    ]
    if not sents:
        return [], 0.0, "no_politician_sentences"
    scores = [_vader_infer.polarity_scores(s)["compound"] for s in sents]
    mean_score = round(sum(scores) / len(scores), 4)
    if mean_score >= 0.05:
        label = "positive"
    elif mean_score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return sents, mean_score, label


def _tone_badge(label: str) -> str:
    return "🟢 POSITIVO" if label == "positive" else "🔴 NEGATIVO"


@st.cache_resource
def load_news_vector_store(embedding_model_name: str):
    from ainee_politics.infrastructure.nlp.rag import build_vector_store

    return build_vector_store(
        corpus_path=LABELED_CORPUS_PATH,
        persist_dir=RAG_VECTOR_DIR,
        embedding_model_name=embedding_model_name,
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

report = load_report()

st.title("📰 Ainee Politics — Análisis de Sesgo Mediático")
st.caption(
    "Análisis del tono positivo o negativo de noticias políticas internacionales "
    "y exploración del corpus por político."
)

if not report:
    st.error(
        "No se encontró `data/training_report.json`. "
        "Ejecuta primero: `python main.py train-model --input data/corpus_labeled.jsonl`"
    )
    st.stop()

classical = report.get("classical_model", {})
transformer = report.get("transformer_model", {})
comparison = report.get("comparison", {})
corpus = report.get("corpus_stats", {})
agreement = report.get("label_agreement_gdelt_vs_politician")
tone_label_field = corpus.get("tone_label_field", "unknown")

transformer_name = transformer.get("model", "Transformer")
transformer_mode = transformer.get("mode", "")
transformer_label = (
    f"{transformer_name} (fine-tuned)"
    if "fine-tuned" in transformer_mode
    else f"{transformer_name} (zero-shot)"
)

tab_dashboard, tab_predict, tab_rag = st.tabs(
    ["📊 Dashboard de Resultados", "🔍 Predicción en Tiempo Real", "💬 Chatbot RAG"]
)


# ===========================================================================
# TAB 1 — DASHBOARD
# ===========================================================================
with tab_dashboard:

    # --- Corpus overview ---
    st.subheader("Corpus")
    st.caption(f"Etiqueta objetivo activa en este reporte: `{tone_label_field}`")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Artículos totales", corpus.get("total_labeled_articles", "—"))
    c2.metric("Usados para entrenar", corpus.get("used_for_training", "—"))
    c3.metric("Excluidos (neutro/sin mención)", corpus.get("excluded_from_training", "—"))
    dist = corpus.get("training_class_distribution", {})
    c4.metric("Positivos", dist.get("positive", "—"))
    c5.metric("Negativos", dist.get("negative", "—"))

    if agreement:
        rate = agreement["agreement_rate"]
        c_agree, _ = st.columns([2, 3])
        with c_agree:
            st.info(
                f"Acuerdo GDELT vs etiqueta por político: **{rate:.1%}** "
                f"({agreement['n_agree']}/{agreement['n_comparable']} artículos). "
                f"{'Alta coincidencia.' if rate >= 0.70 else 'Baja coincidencia — la etiqueta por político aporta información distinta al tono GDELT.'}"
            )

    st.divider()

    # --- Model metrics ---
    st.subheader("Comparación de Modelos")
    col_c, col_t = st.columns(2)

    with col_c:
        st.markdown("#### TF-IDF (1-2gram) + LinearSVC")
        st.caption(
            f"Test set compartido: {corpus.get('test_size', classical.get('test_set_size', '—'))} artículos · "
            f"CV auxiliar: {classical.get('cv_folds', 5)} folds sobre train"
        )
        m1, m2 = st.columns(2)
        m1.metric(
            "F1-Macro",
            f"{comparison.get('classical_test_f1_macro', classical.get('test_set_f1_macro', classical.get('f1_macro_mean', 0))):.4f}",
            f"CV train: {classical.get('f1_macro_mean', 0):.4f} ±{classical.get('f1_macro_std', 0):.4f}",
        )
        m2.metric(
            "Accuracy",
            f"{comparison.get('classical_test_accuracy', classical.get('test_set_accuracy', classical.get('accuracy', 0))):.4f}",
        )

        cr = classical.get("test_set_classification_report") or classical.get("classification_report", {})
        if cr:
            rows_cr = []
            for lbl in ("negative", "positive", "macro avg"):
                m = cr.get(lbl, {})
                rows_cr.append({
                    "Clase": lbl,
                    "Precision": round(m.get("precision", 0), 3),
                    "Recall": round(m.get("recall", 0), 3),
                    "F1": round(m.get("f1-score", 0), 3),
                    "Support": int(m.get("support", 0)) if lbl != "macro avg" else "—",
                })
            st.dataframe(pd.DataFrame(rows_cr), hide_index=True, use_container_width=True)

    with col_t:
        st.markdown(f"#### {transformer_label}")
        train_sz = transformer.get("train_size", "—")
        test_sz = transformer.get("test_size", "—")
        st.caption(f"Train: {train_sz} · Test: {test_sz} · {transformer_mode}")
        m1, m2 = st.columns(2)
        delta_f1 = comparison.get(
            "transformer_f1_macro",
            transformer.get("f1_macro", 0),
        ) - comparison.get("classical_test_f1_macro", classical.get("test_set_f1_macro", 0))
        m1.metric(
            "F1-Macro",
            f"{comparison.get('transformer_f1_macro', transformer.get('f1_macro', 0)):.4f}",
            f"{delta_f1:+.4f} vs TF-IDF test",
        )
        m2.metric(
            "Accuracy",
            f"{comparison.get('transformer_accuracy', transformer.get('accuracy', 0)):.4f}",
        )

        cr_t = transformer.get("classification_report", {})
        if cr_t:
            rows_crt = []
            for lbl in ("negative", "positive", "macro avg"):
                m = cr_t.get(lbl, {})
                rows_crt.append({
                    "Clase": lbl,
                    "Precision": round(m.get("precision", 0), 3),
                    "Recall": round(m.get("recall", 0), 3),
                    "F1": round(m.get("f1-score", 0), 3),
                    "Support": int(m.get("support", 0)) if lbl != "macro avg" else "—",
                })
            st.dataframe(pd.DataFrame(rows_crt), hide_index=True, use_container_width=True)

    # LLM column (if available)
    llm = report.get("llm_model")
    if llm:
        with st.expander(f"🤖 {llm.get('model', 'LLM')} (zero-shot Ollama)", expanded=True):
            st.caption(llm.get("mode", ""))
            m1, m2 = st.columns(2)
            l_f1  = llm.get("f1_macro", 0)
            l_acc = llm.get("accuracy", 0)
            delta_lc = comparison.get("llm_vs_classical_delta", l_f1 - classical.get("test_set_f1_macro", 0))
            m1.metric("F1-Macro", f"{l_f1:.4f}", f"{delta_lc:+.4f} vs TF-IDF test")
            m2.metric("Accuracy", f"{l_acc:.4f}")
            n_fail = llm.get("n_failed_parse", 0)
            if n_fail:
                st.caption(f"⚠️ {n_fail} respuestas no parseadas → asignadas a 'negative'")

            cr_l = llm.get("classification_report", {})
            if cr_l:
                rows_crl = []
                for lbl in ("negative", "positive", "macro avg"):
                    m = cr_l.get(lbl, {})
                    rows_crl.append({
                        "Clase": lbl,
                        "Precision": round(m.get("precision", 0), 3),
                        "Recall": round(m.get("recall", 0), 3),
                        "F1": round(m.get("f1-score", 0), 3),
                        "Support": int(m.get("support", 0)) if lbl != "macro avg" else "—",
                    })
                st.dataframe(pd.DataFrame(rows_crl), hide_index=True, use_container_width=True)

    # Winner banner
    winner = comparison.get("winner_by_f1_macro", "")
    delta = comparison.get("f1_macro_delta_classical_minus_transformer", 0)
    l_f1  = comparison.get("llm_f1_macro")

    scores = {
        "TF-IDF (test)": comparison.get("classical_test_f1_macro", 0),
        transformer_label: comparison.get("transformer_f1_macro", 0),
    }
    if l_f1 is not None:
        scores[llm.get("model", "LLM")] = l_f1

    best_name = max(scores, key=scores.get)
    best_val  = scores[best_name]
    st.success(f"🏆 **{best_name}** lidera con F1-Macro **{best_val:.4f}** en el test set compartido")

    st.divider()

    # --- Charts ---
    bias_path = DATA_DIR / "bias_landscape.png"
    comparison_path = DATA_DIR / "comparison_plot.png"

    if bias_path.exists():
        st.subheader("Distribución de Tono por Político (Bias Landscape)")
        st.image(str(bias_path), use_container_width=True)

    if comparison_path.exists():
        st.subheader("Comparación Visual de Modelos")
        st.image(str(comparison_path), use_container_width=True)

    cm_c_path = DATA_DIR / "confusion_matrix_classical.png"
    cm_t_path = DATA_DIR / "confusion_matrix_transformer.png"
    cm_l_path = DATA_DIR / "confusion_matrix_llm.png"
    cm_paths  = [(cm_c_path, "TF-IDF + LinearSVC"), (cm_t_path, transformer_label), (cm_l_path, "LLM (Ollama)")]
    existing  = [(p, lbl) for p, lbl in cm_paths if p.exists()]
    if existing:
        st.subheader("Matrices de Confusión")
        cols = st.columns(len(existing))
        for col, (p, lbl) in zip(cols, existing):
            with col:
                st.caption(lbl)
                st.image(str(p), use_container_width=True)

    st.divider()

    # --- Per-politician table ---
    st.subheader("Accuracy por Político")
    st.caption("⚠️ Las columnas del Transformer incluyen artículos de entrenamiento — los valores pueden estar inflados. Los valores de TF-IDF son out-of-fold (CV).")

    c_per_pol = classical.get("per_politician", {})
    t_per_pol = transformer.get("per_politician", {})

    llm_per_pol = (report.get("llm_model") or {}).get("per_politician", {})
    rows_pol = []
    for pol, cv in c_per_pol.items():
        tv  = t_per_pol.get(pol, {})
        lv  = llm_per_pol.get(pol, {})
        c_acc = cv.get("accuracy", 0)
        t_acc = tv.get("accuracy", 0)
        row = {
            "Político": pol,
            "N": cv.get("n", 0),
            "Pos": cv.get("positive_articles", 0),
            "Neg": cv.get("negative_articles", 0),
            "TF-IDF Acc": c_acc,
            f"{transformer_name[:12]} Acc": t_acc,
            "Delta (T−C)": round(t_acc - c_acc, 3),
        }
        if llm_per_pol:
            row["LLM Acc"] = lv.get("accuracy", None)
        rows_pol.append(row)

    df_pol = (
        pd.DataFrame(rows_pol)
        .sort_values("N", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        df_pol.style.background_gradient(subset=["Delta (T−C)"], cmap="RdYlGn", vmin=-0.2, vmax=0.2),
        hide_index=True,
        use_container_width=True,
    )

    # --- Cross-politician (LOPO) ---
    lopo = report.get("cross_politician_eval")
    if lopo and lopo.get("per_politician"):
        st.divider()
        st.subheader("Evaluación Cross-Político (Leave-One-Politician-Out)")
        st.caption(
            "Cada político se evalúa con un modelo entrenado **sin** sus artículos. "
            "Un drop alto indica que el modelo memorizó ese político en vez de aprender tono real."
        )

        mean_f1 = lopo.get("mean_lopo_f1")
        interpretation = lopo.get("interpretation", "")
        c1, c2 = st.columns([1, 3])
        if mean_f1 is not None:
            c1.metric("F1-Macro medio LOPO", f"{mean_f1:.4f}")
        c2.info(interpretation)

        lopo_rows = []
        for pol, s in lopo["per_politician"].items():
            lopo_rows.append({
                "Político": pol,
                "N test": s.get("n_test", "—"),
                "LOPO Accuracy": s.get("lopo_accuracy", 0),
                "LOPO F1-Macro": s.get("lopo_f1_macro", 0),
                "Within-dist Acc": s.get("within_dist_accuracy", None),
                "Drop (Within−LOPO)": s.get("generalization_drop", None),
            })
        df_lopo = (
            pd.DataFrame(lopo_rows)
            .sort_values("Drop (Within−LOPO)", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        # Only gradient the drop column if it has numeric values
        drop_col = "Drop (Within−LOPO)"
        has_drops = df_lopo[drop_col].notna().any()
        styled = (
            df_lopo.style.background_gradient(subset=[drop_col], cmap="RdYlGn_r", vmin=0, vmax=0.3)
            if has_drops else df_lopo.style
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)


# ===========================================================================
# TAB 2 — INFERENCE
# ===========================================================================
with tab_predict:
    st.subheader("Tono del artículo y menciones al político")
    st.caption(
        "El análisis se centra en las frases que mencionan directamente al político seleccionado, "
        "igual que el proceso de etiquetado del corpus. Pega el texto o usa el fetch de URL."
    )

    try:
        from ainee_politics.domain.catalog import DEFAULT_POLITICIANS
        politician_names = [p.name for p in DEFAULT_POLITICIANS]
    except Exception:
        politician_names = [
            "Donald Trump", "Joe Biden", "Kamala Harris", "Javier Milei",
            "Keir Starmer", "Emmanuel Macron", "Giorgia Meloni", "Gustavo Petro",
            "Claudia Sheinbaum", "Benjamin Netanyahu", "Volodymyr Zelenskyy",
            "Alberto Nunez Feijoo",
        ]

    # --- URL fetch ---
    if "fetched_text" not in st.session_state:
        st.session_state["fetched_text"] = ""
    if "fetch_status" not in st.session_state:
        st.session_state["fetch_status"] = ""

    col_url, col_fetch = st.columns([5, 1])
    with col_url:
        url_input = st.text_input(
            "URL del artículo (opcional)",
            placeholder="https://www.bbc.com/news/...",
        )
    with col_fetch:
        st.write("")
        fetch_clicked = st.button("⬇️ Fetch", use_container_width=True)

    if fetch_clicked:
        if not url_input.strip():
            st.warning("Introduce una URL antes de hacer fetch.")
        else:
            from ainee_politics.infrastructure.text.article_extractor import extract_article_payload
            from ainee_politics.config import REQUEST_TIMEOUT, API_RETRIES
            with st.spinner("Descargando artículo..."):
                payload = extract_article_payload(
                    url=url_input.strip(),
                    timeout=REQUEST_TIMEOUT,
                    retries=API_RETRIES,
                )
            status = payload.get("content_fetch_status", "error")
            content = payload.get("content", "")
            if status == "ok" and content:
                st.session_state["fetched_text"] = content
                st.session_state["fetch_status"] = (
                    f"✅ Extraídos {payload['content_length_words']} palabras "
                    f"({payload['content_length_chars']} caracteres)"
                )
            else:
                st.session_state["fetched_text"] = ""
                st.session_state["fetch_status"] = (
                    f"❌ No se pudo extraer contenido ({status}). "
                    "Prueba a pegar el texto manualmente."
                )

    if st.session_state["fetch_status"]:
        if st.session_state["fetch_status"].startswith("✅"):
            st.success(st.session_state["fetch_status"])
        else:
            st.error(st.session_state["fetch_status"])

    # --- Politician + text ---
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        selected_politician = st.selectbox("Político a analizar", politician_names)

    article_text = st.text_area(
        "Texto del artículo (título + contenido)",
        value=st.session_state["fetched_text"],
        placeholder="Pega aquí el texto del artículo, o usa el fetch de URL de arriba...",
        height=220,
    )

    analyze = st.button("🔍 Analizar tono", type="primary")

    if analyze:
        if not article_text.strip():
            st.warning("Introduce el texto del artículo antes de analizar.")
        else:
            aliases = _get_aliases(selected_politician)

            spacy_ok = True
            try:
                with st.spinner("Procesando texto con spaCy..."):
                    nlp = load_spacy()
            except Exception as exc:
                st.error(
                    f"spaCy no disponible: {exc}\n\n"
                    "Instálalo con: `python -m spacy download en_core_web_lg`"
                )
                spacy_ok = False

            if spacy_ok:
                politician_sents, vader_score, vader_label = _score_politician_sentences(
                    article_text, aliases, nlp
                )
                n_sents = len(politician_sents)

                if n_sents == 0:
                    st.warning(
                        f"No se encontraron frases que mencionen a **{selected_politician}** "
                        f"(aliases buscados: *{', '.join(aliases)}*).\n\n"
                        "¿El artículo usa un nombre diferente para este político?"
                    )
                else:
                    if n_sents < 3:
                        st.warning(
                            f"Solo {n_sents} frase(s) encontrada(s) — la confianza del análisis es baja."
                        )
                    else:
                        st.info(
                            f"📍 **{n_sents} frases** encontradas mencionando a **{selected_politician}**"
                        )

                    with st.expander("Ver frases analizadas"):
                        for i, s in enumerate(politician_sents, 1):
                            st.markdown(f"**{i}.** {s}")

                    st.divider()

                    # --- Primary: VADER on politician sentences ---
                    st.subheader("Tono hacia el político")
                    vcol, _ = st.columns([2, 3])
                    with vcol:
                        if vader_label == "neutral":
                            st.markdown("### 🟡 NEUTRAL")
                        else:
                            st.markdown(f"### {_tone_badge(vader_label)}")
                        st.metric(
                            "VADER score",
                            f"{vader_score:+.3f}",
                            help="Rango: −1 (muy negativo) a +1 (muy positivo). Umbral: ±0.05",
                        )
                        pos_ratio = (vader_score + 1) / 2
                        st.progress(pos_ratio, text="← negativo · positivo →")

                    st.divider()

                    # --- Secondary: ML classifiers on full article ---
                    st.subheader("Clasificadores ML")
                    st.caption(
                        "Predicción del tono general del artículo completo, usando la etiqueta binaria positiva/negativa del corpus."
                    )
                    classifier_input = article_text
                    label_t = pred_c = None

                    col_transformer, col_classical = st.columns(2)

                    with col_transformer:
                        st.markdown(f"#### {transformer_label}")
                        if "fine-tuned" not in transformer_mode:
                            st.info(
                                "El reporte actual usa transformer zero-shot; no hay modelo fine-tuned guardado para inferencia en tiempo real."
                            )
                        elif not TRANSFORMER_MODEL_DIR.exists():
                            st.info("Modelo no encontrado. Ejecuta `train-model` primero.")
                        else:
                            try:
                                with st.spinner("Clasificando..."):
                                    clf_t = load_transformer_pipeline()
                                result = clf_t(classifier_input[:1500])[0]
                                label_t = result["label"]
                                score_t = result["score"]
                                st.markdown(f"### {_tone_badge(label_t)}")
                                st.metric("Confianza", f"{score_t:.1%}")
                                pos_score = score_t if label_t == "positive" else 1 - score_t
                                st.progress(pos_score, text="← negativo · positivo →")
                            except Exception as exc:
                                st.error(f"Error al clasificar: {exc}")

                    with col_classical:
                        st.markdown("#### TF-IDF + LinearSVC")
                        clf_c = load_classical_pipeline()
                        if clf_c is None:
                            st.info("Modelo no disponible. Re-ejecuta `train-model`.")
                        else:
                            try:
                                pred_c = clf_c.predict([classifier_input])[0]
                                st.markdown(f"### {_tone_badge(pred_c)}")
                                decision = clf_c.decision_function([classifier_input])[0]
                                st.metric("Score de decisión", f"{decision:.3f}")
                                st.caption("Positivo → clase positiva · Negativo → clase negativa.")
                            except Exception as exc:
                                st.error(f"Error al clasificar: {exc}")

                    # --- Consensus ---
                    all_labels = [l for l in [vader_label, label_t, pred_c] if l and l != "neutral"]
                    if len(all_labels) >= 2:
                        from collections import Counter as _Counter
                        vote, vote_count = _Counter(all_labels).most_common(1)[0]
                        st.divider()
                        if vote_count == len(all_labels):
                            st.success(
                                f"✅ Los {len(all_labels)} análisis coinciden: **{_tone_badge(vote)}**"
                            )
                        else:
                            st.warning(
                                f"⚠️ Resultados mixtos — mayoría: **{_tone_badge(vote)}** "
                                f"({vote_count}/{len(all_labels)})"
                            )


# ===========================================================================
# TAB 3 — RAG CHATBOT
# ===========================================================================
with tab_rag:
    st.subheader("Chatbot RAG sobre noticias políticas")
    st.caption(
        "Construye una base vectorial del corpus etiquetado y responde preguntas "
        "solo con la información de las noticias, siguiendo el enfoque de chunking + retrieval del notebook de conceptos avanzados."
    )

    if not LABELED_CORPUS_PATH.exists():
        st.error(
            "No se encontró `data/corpus_labeled.jsonl`. Ejecuta primero: "
            "`python main.py label-corpus --input data/corpus_politicos_clean.jsonl`"
        )
    else:
        rag_politician_options = ["Todos"] + politician_names
        col_filter, col_model, col_actions = st.columns([2, 2, 1])

        with col_filter:
            rag_politician = st.selectbox(
                "Filtrar por político",
                rag_politician_options,
                key="rag_politician",
            )

        with col_model:
            rag_model_name = st.text_input(
                "Modelo Ollama para responder",
                value="llama3.1:8b",
                key="rag_model_name",
            )

        with col_actions:
            st.write("")
            rebuild_index = st.button("Reindexar", use_container_width=True)
            clear_chat = st.button("Limpiar chat", use_container_width=True)

        if clear_chat:
            st.session_state["rag_messages"] = []

        if rebuild_index:
            load_news_vector_store.clear()

        try:
            with st.spinner("Preparando base vectorial del corpus..."):
                vector_store, rag_stats = load_news_vector_store(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
        except Exception as exc:
            st.error(
                "No se pudo inicializar el chatbot RAG. Instala las nuevas dependencias con "
                "`pip install -r requirements.txt` y asegúrate de que Ollama está disponible.\n\n"
                f"Detalle: {exc}"
            )
        else:
            st.info(
                f"Corpus indexado: {rag_stats['articles_indexed']} artículos · "
                f"chunks nuevos en esta carga: {rag_stats['chunks_added']}"
            )

            if "rag_messages" not in st.session_state:
                st.session_state["rag_messages"] = []

            for message in st.session_state["rag_messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and message.get("sources"):
                        with st.expander("Fuentes usadas"):
                            for source in message["sources"]:
                                title = source["title"]
                                url = source["url"]
                                domain = source["domain"]
                                politician = source["politician"]
                                tone = source["tone_label"]
                                snippet = source["snippet"]
                                if url:
                                    st.markdown(f"**[{source['rank']}] [{title}]({url})**")
                                else:
                                    st.markdown(f"**[{source['rank']}] {title}**")
                                st.caption(f"{domain} · {politician} · tono {tone}")
                                st.write(snippet)

            prompt = st.chat_input(
                "Pregunta algo sobre el corpus político o pide en qué noticias aparece un político...",
                key="rag_prompt",
            )

            if prompt:
                st.session_state["rag_messages"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Recuperando noticias relevantes y redactando respuesta..."):
                        from ainee_politics.infrastructure.nlp.rag import answer_question

                        rag_response = answer_question(
                            prompt,
                            vector_store=vector_store,
                            ollama_model=rag_model_name,
                            chat_history=st.session_state["rag_messages"][:-1],
                            politician=None if rag_politician == "Todos" else rag_politician,
                        )

                    if rag_response.get("error"):
                        st.warning(rag_response["answer"])
                    else:
                        st.markdown(rag_response["answer"])
                    if rag_response["sources"]:
                        with st.expander("Fuentes usadas", expanded=True):
                            for source in rag_response["sources"]:
                                title = source["title"]
                                url = source["url"]
                                domain = source["domain"]
                                politician = source["politician"]
                                tone = source["tone_label"]
                                snippet = source["snippet"]
                                if url:
                                    st.markdown(f"**[{source['rank']}] [{title}]({url})**")
                                else:
                                    st.markdown(f"**[{source['rank']}] {title}**")
                                st.caption(f"{domain} · {politician} · tono {tone}")
                                st.write(snippet)

                st.session_state["rag_messages"].append(
                    {
                        "role": "assistant",
                        "content": rag_response["answer"],
                        "sources": rag_response["sources"],
                    }
                )
