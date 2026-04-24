# Ainee Politics

Proyecto modular para construir un corpus politico en ingles usando GDELT DOC API y GDELT GKG.

## Arquitectura

La aplicacion ya no concentra toda la logica en un unico script. La estructura principal es:

```text
main.py
ainee_politics/
	config.py
	domain/
		catalog.py
		models.py
	application/
		summaries.py
		use_cases/
			build_corpus.py
			prepare_dataset.py
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
	presentation/
		cli.py
```

Responsabilidades:

- `main.py`: punto de entrada principal con CLI moderna por subcomandos.
- `config.py`: constantes de configuracion compartidas.
- `domain/`: entidades, settings y catalogo del dominio.
- `application/`: casos de uso y construccion de resúmenes.
- `infrastructure/`: adaptadores técnicos para GDELT, almacenamiento local y normalizacion textual.
- `presentation/`: CLI y adaptación de argumentos a casos de uso.

## Que genera

- `data/corpus_politicos_en.jsonl`
- `data/corpus_politicos_en.csv`
- `data/resumen_politicos_api.json`

Cada noticia incluye:

- metadatos basicos de GDELT
- `content` extraido automaticamente desde la URL
- tono por articulo enlazado desde GDELT GKG (`gdelt_tone_score`, `gdelt_tone_label`, etc.)

## Segunda capa: dataset limpio

Puedes generar un dataset listo para modelado con:

```powershell
python main.py prepare-dataset --input data/corpus_politicos_en.jsonl --min-content-chars 200
```

Salida:

- `data/corpus_politicos_clean.jsonl`
- `data/corpus_politicos_clean.csv`
- `data/resumen_preparacion.json`

Esta capa:

- deduplica por URL normalizada
- puede descartar filas con contenido insuficiente
- puede exigir que aparezca un alias del politico en `title` o `content`
- genera un campo `text` concatenando `title + content` para modelado

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

CLI principal:

```powershell
python main.py build-corpus --max-politicians 1 --max-records 2 --timespan 7d --gdelt-min-interval 6.5 --request-timeout 60 --retries 3 --sleep-seconds 0.5
```

Ejecucion grande:

```powershell
python main.py build-corpus --max-records 100 --timespan 3m --gdelt-min-interval 6.5 --request-timeout 90 --retries 5 --sleep-seconds 1.5
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
- La deduplicacion ahora normaliza URLs para evitar duplicados triviales como `:443` frente a la misma URL sin puerto.
- El dataset limpio esta pensado como entrada estable para los modulos posteriores de entrenamiento y evaluacion.
- La carpeta `data/` no se versiona para evitar subir datasets grandes al repositorio.