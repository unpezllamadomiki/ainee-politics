# Ainee Politics

Proyecto modular para construir un corpus de noticias políticas en inglés usando [GDELT](https://www.gdeltproject.org/), etiquetarlo y entrenar modelos de clasificación binaria del tono de la noticia, además de consultar en qué noticias aparece cada político.

## Objetivo

Medir si una noticia política tiene tono positivo o negativo a nivel de artículo y permitir explorar en qué noticias aparece cada político dentro del corpus.

## Resultados (última ejecución)

| Modelo | F1-Macro (test) | Accuracy (test) |
|---|---|---|
| TF-IDF + LinearSVC | **0.8008** | **0.8911** |
| DistilBERT (fine-tuned, 3 epochs) | 0.7649 | 0.8911 |
| Llama 3.1 8B (zero-shot, Ollama) | 0.5948 | 0.7525 |

- **Corpus:** 2,104 artículos etiquetados (1,006 usados para entrenamiento), 13 políticos, split 80/20 (train 804 / test 202)
- **Evaluación cross-político (LOPO):** F1-Macro medio = 0.6109 — el modelo depende de señales específicas por político, generalización limitada

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

Entrena TF-IDF + LinearSVC y hace fine-tuning de un transformer (DistilBERT por defecto) sobre el mismo split 80/20:

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

Visualiza resultados, realiza predicciones en tiempo real y consulta el corpus con un chatbot RAG:

```bash
streamlit run app.py
```

- **Tab Dashboard:** métricas por modelo, bias landscape, matrices de confusión, accuracy por político, evaluación LOPO
- **Tab Predicción:** pega una URL o texto, selecciona el político, y ambos modelos predicen el tono
- **Tab Chatbot RAG:** permite preguntar sobre el corpus de noticias y devuelve respuesta con fuentes

### Chatbot RAG

El chat usa un índice vectorial persistido en `data/chroma_politics_news/` construido a partir de `data/corpus_labeled.jsonl`.
Si el índice ya existe, la app lo reutiliza; si no existe, lo crea al abrir la pestaña del chatbot.

Capacidades principales:
- Resumir noticias recientes de un político concreto
- Responder preguntas apoyadas en fragmentos recuperados del corpus
- Listar directamente en qué noticias aparece un político

Ejemplos de preguntas:
- `Resume las noticias recientes sobre Donald Trump`
- `¿En qué noticias aparece Emmanuel Macron?`
- `¿Qué dice el corpus sobre Claudia Sheinbaum?`

Requisitos:
- Tener `data/corpus_labeled.jsonl` generado
- Tener Ollama instalado y el modelo descargado, por ejemplo `ollama pull llama3.1:8b`
- Ejecutar `streamlit run app.py`

Notas de uso:
- El filtro de político en la UI limita la recuperación a ese político cuando es posible
- Las respuestas muestran tarjetas de fuente con título, medio, fecha, tono y URL
- El índice Chroma es regenerable y no se versiona en git

---

## Arquitectura

```text
main.py
app.py                          ← app Streamlit: dashboard, predicción y chatbot RAG
ainee_politics/
    config.py
    domain/
        catalog.py              ← 12 políticos internacionales (editable)
        models.py               ← dataclasses de settings y esquemas de columnas
    application/
        summaries.py            ← helpers para resumir métricas y resultados
        use_cases/
            build_corpus.py     ← descarga y consolida noticias desde GDELT
            prepare_dataset.py  ← limpieza, deduplicación y filtrado del corpus
            label_corpus.py     ← etiquetado de tono y enriquecimiento NLP
            train_model.py      ← entrenamiento clásico + transformer + reporte
            compare_llm.py      ← evaluación zero-shot con Ollama
    infrastructure/
        gdelt/
            client.py           ← cliente HTTP a GDELT
            query_builder.py    ← construcción de consultas por político/ventana
            tone.py             ← normalización del tono GDELT
        storage/
            dataset_store.py    ← lectura/escritura de JSONL, CSV y checkpoints
        text/
            article_extractor.py ← extracción de título, cuerpo y metadatos
            normalization.py     ← normalización textual previa al pipeline
        nlp/
            spacy_processor.py  ← NER + VADER por entidad
            classifier.py       ← TF-IDF+LinearSVC, fine-tuning transformer, LOPO, LLM eval
            rag.py              ← Chroma + embeddings + respuestas RAG sobre el corpus
    presentation/
        cli.py                  ← comandos build-corpus, prepare-dataset, label-corpus, train-model, compare-llm
data/
    corpus_politicos_en.*       ← exportación cruda/normalizada desde GDELT
    corpus_politicos_clean.*    ← corpus limpio listo para etiquetar
    corpus_labeled.*            ← corpus etiquetado para entrenamiento y RAG
    training_report.json        ← métricas agregadas que consume el dashboard
    classical_model.joblib      ← pipeline TF-IDF + LinearSVC persistido
    finetuned_model/            ← transformer fine-tuned para inferencia
    chroma_politics_news/       ← índice vectorial persistido del chatbot RAG
    confusion_matrix_*.png      ← matrices de confusión generadas en evaluación
    comparison_plot.png         ← comparación visual entre enfoques
    bias_landscape.png          ← paisaje de sesgo por político/medio
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
