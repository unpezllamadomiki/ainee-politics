# Ainee Politics

Proyecto modular para construir un corpus de noticias políticas en inglés usando [GDELT](https://www.gdeltproject.org/), etiquetarlo y entrenar modelos de clasificación binaria del tono de la noticia, además de consultar en qué noticias aparece cada político.

## Objetivo

Medir si una noticia política tiene tono positivo o negativo a nivel de artículo y permitir explorar en qué noticias aparece cada político dentro del corpus.

## Resultados (última ejecución)

| Modelo | F1-Macro (test) | Accuracy (test) |
|---|---|---|
| TF-IDF + LinearSVC | **0.8561** | **0.8596** |
| RoBERTa (fine-tuned, 3 epochs) | 0.7623 | 0.7632 |
| Llama 3.1 8B zero-shot (Ollama) | 0.5529 | 0.6053 |

- **Corpus:** 570 artículos binarios (positive/negative), 12 políticos, split 80/20 compartido entre modelos
- **Evaluación cross-político (LOPO):** F1-Macro medio = 0.5231 — el modelo depende de señales específicas por político, generalización limitada

## Pipeline completo

```
build-corpus → prepare-dataset → label-corpus → train-model → [compare-llm]
```

`label-corpus` y `compare-llm` son opcionales.

---

## Instalación

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Para `compare-llm` se necesita [Ollama](https://ollama.com) instalado localmente:

```bash
ollama pull llama3.1:8b
```

Para fine-tuning con GPU (recomendado), instalar PyTorch con soporte CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

---

## Paso 1 — Construir corpus crudo

Descarga artículos de GDELT, extrae el contenido HTML y enriquece con el tono V2Tone del GKG:

```bash
# Prueba rápida (~5-10 min, 5 políticos × 20 artículos)
python main.py build-corpus --max-politicians 5 --max-records 20 --timespan 30d

# Corpus completo (~1-2h)
python main.py build-corpus --max-records 100 --timespan 3m --checkpoint-every 10
```

Salida: `data/corpus_politicos_en.jsonl`, `data/corpus_politicos_en.csv`, `data/resumen_politicos_api.json`

## Paso 2 — Limpiar y preparar dataset

Deduplica, filtra por calidad de contenido y exige que el político aparezca en el texto:

```bash
python main.py prepare-dataset --input data/corpus_politicos_en.jsonl --min-content-chars 200
```

Salida: `data/corpus_politicos_clean.jsonl`, `data/corpus_politicos_clean.csv`, `data/resumen_preparacion.json`

## Paso 3 — Etiquetar con NLP (spaCy + VADER)

Enriquece el corpus con anotaciones lingüísticas y calcula el tono hacia el político específicamente:

```bash
python main.py label-corpus --input data/corpus_politicos_clean.jsonl
```

Añade a cada artículo:
- `politician_tone_label` — tono de las frases que mencionan al político (`positive` / `negative` / `neutral`)
- `politician_tone_score` — puntuación compuesta promedio (-1 a +1)
- `spacy_entities` — entidades nombradas (PERSON, ORG, GPE…)
- `politician_adjectives` — modificadores sintácticos directos del político
- `sentence_count`, `avg_sentence_length`

Salida: `data/corpus_labeled.jsonl`, `data/corpus_labeled.csv`

## Paso 4 — Entrenar y comparar modelos

Entrena TF-IDF + LinearSVC y hace fine-tuning de un transformer (RoBERTa por defecto) sobre el mismo split 80/20:

```bash
python main.py train-model --input data/corpus_labeled.jsonl

# Sin GPU / sin fine-tuning
python main.py train-model --input data/corpus_labeled.jsonl --no-finetune
```

Salida en `data/`:
- `training_report.json` — métricas completas, distribución por clase y político, LOPO
- `classical_model.joblib` — pipeline TF-IDF+LinearSVC para inferencia
- `finetuned_model/` — modelo transformer fine-tuneado para inferencia
- `bias_landscape.png` — distribución de tono por político (output principal de investigación)
- `comparison_plot.png` — comparación visual de modelos
- `confusion_matrix_classical.png`, `confusion_matrix_transformer.png`

## Paso 5 — Comparar con LLM local (opcional)

Evalúa Llama 3.1 8B (u otro modelo Ollama) sobre el mismo test set:

```bash
python main.py compare-llm --input data/corpus_labeled.jsonl --ollama-model llama3.1:8b
```

Añade la sección `llm_model` al `training_report.json` y genera `confusion_matrix_llm.png`.

---

## Dashboard Streamlit

Visualiza resultados y realiza predicciones en tiempo real:

```bash
streamlit run app.py
```

- **Tab Dashboard:** métricas por modelo, bias landscape, matrices de confusión, accuracy por político, evaluación LOPO
- **Tab Predicción:** pega una URL o texto, selecciona el político, y ambos modelos predicen el tono

---

## Arquitectura

```text
main.py
app.py                          ← dashboard Streamlit
ainee_politics/
    config.py
    domain/
        catalog.py              ← 12 políticos internacionales (editable)
        models.py               ← dataclasses de settings y esquemas de columnas
    application/
        use_cases/
            build_corpus.py
            prepare_dataset.py
            label_corpus.py
            train_model.py
            compare_llm.py
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
            spacy_processor.py  ← NER + VADER por entidad
            classifier.py       ← TF-IDF+LinearSVC, fine-tuning transformer, LOPO, LLM eval
    presentation/
        cli.py
```

---

## Parámetros principales

### build-corpus
| Flag | Defecto | Descripción |
|---|---|---|
| `--max-politicians` | 0 (todos) | Limita el número de políticos (útil para pruebas) |
| `--max-records` | 75 | Artículos máximos por político |
| `--timespan` | `30d` | Ventana temporal GDELT: `7d`, `30d`, `3m`, `6m` |
| `--checkpoint-every` | 10 | Guarda progreso cada N artículos |

### prepare-dataset
| Flag | Defecto | Descripción |
|---|---|---|
| `--min-content-chars` | 200 | Mínimo de caracteres en el cuerpo del artículo |
| `--keep-neutral` | false | Conserva artículos con tono neutral |
| `--disable-alias-filter` | false | No exige que el político aparezca en el texto |

### train-model
| Flag | Defecto | Descripción |
|---|---|---|
| `--transformer-model` | `distilbert-base-uncased` | Modelo HuggingFace base para fine-tuning |
| `--no-finetune` | false | Desactiva fine-tuning (usa el modelo en modo zero-shot) |
| `--finetune-epochs` | 3 | Epochs de fine-tuning |
| `--finetune-batch-size` | 16 | Batch size para fine-tuning |
| `--finetune-lr` | 2e-5 | Learning rate |
| `--finetune-test-size` | 0.2 | Fracción reservada para test (mismo split para ambos modelos) |
| `--cv-folds` | 5 | Folds de validación cruzada del TF-IDF |
| `--max-features` | 10 000 | Features máximas del vectorizador TF-IDF |

### compare-llm
| Flag | Defecto | Descripción |
|---|---|---|
| `--ollama-model` | `llama3.1:8b` | Nombre del modelo en Ollama |
| `--test-size` | 0.2 | Debe coincidir con el valor usado en `train-model` |

---

## Notas

- GDELT aplica rate limiting de ~5 segundos. No reducir `--gdelt-min-interval`.
- El entrenamiento usa `gdelt_tone_label` como objetivo principal para medir el tono general de la noticia; `politician_tone_label` queda como enriquecimiento adicional del corpus.
- El fine-tuning con GPU (CUDA) es significativamente más rápido (~15 min vs varias horas en CPU).
- Los modelos entrenados (`finetuned_model/`, `*.joblib`) no se versionan en git — son regenerables con `train-model`.
- Para añadir políticos: editar `DEFAULT_POLITICIANS` en `domain/catalog.py`.
- Sin API keys — GDELT es completamente público.
