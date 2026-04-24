# Ainee Politics

Script para construir un corpus politico en ingles usando GDELT DOC API y GDELT GKG.

## Que genera

- `data/corpus_politicos_en.jsonl`
- `data/corpus_politicos_en.csv`
- `data/resumen_politicos_api.json`

Cada noticia incluye:

- metadatos basicos de GDELT
- `content` extraido automaticamente desde la URL
- tono por articulo enlazado desde GDELT GKG (`gdelt_tone_score`, `gdelt_tone_label`, etc.)

## Requisitos

- Python 3.11 o superior
- Dependencias de `requirements.txt`

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecucion rapida

Prueba pequena:

```powershell
python gdelt_corpus_politicos.py --max-politicians 1 --max-records 2 --timespan 7d --gdelt-min-interval 6.5 --request-timeout 60 --retries 3 --sleep-seconds 0.5
```

Ejecucion grande:

```powershell
python gdelt_corpus_politicos.py --max-records 100 --timespan 3m --gdelt-min-interval 6.5 --request-timeout 90 --retries 5 --sleep-seconds 1.5
```

## Parametros principales

- `--max-politicians`: limita el numero de politicos para pruebas.
- `--max-records`: numero maximo de articulos por politico.
- `--timespan`: ventana temporal de GDELT, por ejemplo `7d`, `30d`, `3m`, `6m`.
- `--gdelt-min-interval`: segundos minimos entre peticiones a GDELT.
- `--request-timeout`: timeout de red en segundos.
- `--retries`: numero de reintentos ante errores de red.
- `--sleep-seconds`: pausa entre descargas de articulos.

## Notas

- GDELT aplica rate limiting, por eso el script espacia las peticiones.
- El tono por articulo no se toma de una serie temporal agregada: se enlaza desde GDELT GKG usando la URL y el `seendate` de cada noticia.
- La carpeta `data/` no se versiona para evitar subir datasets grandes al repositorio.