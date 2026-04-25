# Ainee Politics

Proyecto modular para construir un corpus político en inglés usando GDELT DOC API y GDELT GKG, etiquetarlo con análisis de sentimiento orientado a entidades, y entrenar modelos de clasificación de tono para detección de sesgo en medios.

## Objetivo

Detectar sesgo en noticias digitales sobre políticos internacionales analizando cómo los medios retratan a cada político (tono positivo o negativo en las frases que lo mencionan directamente).

## Pipeline completo

```
build-corpus → prepare-dataset → label-corpus → train-model
```

### Paso 1 — Construir corpus crudo

Descarga artículos de GDELT, extrae el contenido HTML y enriquece con el tono V2Tone del GKG:

```powershell
# Prueba rápida (~5-10 min)
python main.py build-corpus --max-politicians 5 --max-records 20 --timespan 30d

# Corpus completo (~1-2h, recomendado para entrenar)
python main.py build-corpus --max-records 100 --timespan 3m --checkpoint-every 10
```

Salida: `data/corpus_politicos_en.jsonl`, `data/corpus_politicos_en.csv`, `data/resumen_politicos_api.json`

### Paso 2 — Limpiar y preparar dataset

Deduplica, filtra por calidad de contenido y exige que el político aparezca en el texto:

```powershell
python main.py prepare-dataset --input data/corpus_politicos_en.jsonl --min-content-chars 200
```

Salida: `data/corpus_politicos_clean.jsonl`, `data/corpus_politicos_clean.csv`, `data/resumen_preparacion.json`

### Paso 3 — Etiquetar con NLP (spaCy + VADER)

Enriquece el corpus limpio con anotaciones lingüísticas y calcula el tono hacia el político específicamente:

```powershell
python main.py label-corpus --input data/corpus_politicos_clean.jsonl
```

Este paso añade a cada artículo:
- `politician_tone_label` — tono de las frases que mencionan al político (`positive` / `negative` / `neutral` / `no_politician_sentences`), calculado con VADER
- `politician_tone_score` — puntuación compuesta promedio (-1 a +1)
- `politician_tone_n_sentences` — número de frases analizadas
- `spacy_entities` — entidades nombradas detectadas (PERSON, ORG, GPE, NORP…)
- `politician_adjectives` — modificadores sintácticos directos del político (dependencias `amod`/`appos`)
- `sentence_count`, `avg_sentence_length` — estadísticas de estructura textual

Salida: `data/corpus_labeled.jsonl`, `data/corpus_labeled.csv`

> Requiere el modelo spaCy en inglés: `python -m spacy download en_core_web_lg`

### Paso 4 — Entrenar y comparar modelos

Entrena un clasificador clásico (TF-IDF + LinearSVC) y evalúa un transformer preentrenado (DistilBERT, zero-shot) sobre la tarea de clasificación de tono hacia el político:

```powershell
python main.py train-model --input data/corpus_labeled.jsonl
```

Si se ejecuta sobre el corpus limpio sin etiquetar, usa automáticamente `gdelt_tone_label` como fallback e informa en consola.

Salida en `data/`:
- `bias_landscape.png` — distribución de tono positivo/negativo/neutral por político en todo el corpus (output principal de investigación)
- `comparison_plot.png` — accuracy y F1-macro de ambos modelos + accuracy por político
- `confusion_matrix_classical.png`, `confusion_matrix_transformer.png`
- `training_report.json` — métricas completas, distribución por clase y político, acuerdo entre etiquetas GDELT y VADER

## Arquitectura

```text
main.py
ainee_politics/
    config.py
    domain/
        catalog.py          ← 19 políticos internacionales (editable)
        models.py           ← dataclasses de settings y esquemas de columnas
    application/
        summaries.py
        use_cases/
            build_corpus.py
            prepare_dataset.py
            label_corpus.py
            train_model.py
    infrastructure/
        gdelt/
            client.py
            query_builder.py
            tone.py
        storage/
            dataset_store.py
        text/
            article_extractor.py
            normalization.py
        nlp/
            spacy_processor.py  ← NER + extracción de adjetivos + VADER por entidad
            classifier.py       ← TF-IDF+LinearSVC, DistilBERT zero-shot, plots
    presentation/
        cli.py
```

## Requisitos

- Python 3.11 o superior
- Dependencias: `pip install -r requirements.txt`
- Modelo spaCy: `python -m spacy download en_core_web_lg`
- Sin API keys — GDELT es público

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Parámetros principales

### build-corpus
- `--max-politicians` — limita el número de políticos procesados (útil para pruebas)
- `--max-records` — artículos máximos por político
- `--timespan` — ventana temporal de GDELT: `7d`, `30d`, `3m`, `6m`
- `--checkpoint-every` — guarda progreso cada N artículos
- `--gdelt-min-interval` — segundos entre peticiones a GDELT (no reducir, hay rate limit)

### prepare-dataset
- `--min-content-chars` — mínimo de caracteres en el cuerpo del artículo (defecto: 200)
- `--keep-neutral` — conserva artículos con `gdelt_tone_label = neutral`
- `--disable-alias-filter` — no exige que el político aparezca en el texto

### train-model
- `--cv-folds` — folds de validación cruzada estratificada (defecto: 5)
- `--max-features` — features máximas del vectorizador TF-IDF (defecto: 10 000)
- `--transformer-model` — modelo HuggingFace para inferencia zero-shot
- `--text-max-chars` — caracteres máximos enviados al transformer (defecto: 1500)

## Notas

- GDELT aplica rate limiting de ~5 segundos entre peticiones. No reducir `--gdelt-min-interval`.
- `politician_tone_label` (VADER por frases) es más preciso que `gdelt_tone_label` (V2Tone del artículo completo) para el objetivo de detección de sesgo, porque mide cómo se retrata al político específicamente, no el tono general del artículo.
- La carpeta `data/` no se versiona para evitar subir datasets grandes al repositorio.
