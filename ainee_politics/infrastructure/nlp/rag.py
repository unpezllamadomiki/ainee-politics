"""RAG helpers to query the political news corpus with LangChain + Chroma."""

from __future__ import annotations

import hashlib
import html
import re
import shutil
from pathlib import Path
from typing import Any

from ainee_politics.domain.catalog import DEFAULT_POLITICIANS
from ainee_politics.infrastructure.storage.dataset_store import ensure_output_dir, read_jsonl

DEFAULT_COLLECTION_NAME = "ainee_politics_news"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
_LIST_NEWS_HINTS = (
    "en que noticias",
    "en qué noticias",
    "que noticias aparece",
    "qué noticias aparece",
    "donde aparece",
    "dónde aparece",
    "aparece en que noticias",
    "aparece en qué noticias",
    "listame noticias",
    "listame las noticias",
    "lista noticias",
    "noticias aparece",
)


def _normalize_text(value: str) -> str:
    text = html.unescape(str(value or "")).replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _build_source_id(row: dict[str, Any], index: int) -> str:
    raw = "|".join(
        [
            str(row.get("normalized_url") or row.get("url") or ""),
            str(row.get("politician") or ""),
            str(row.get("title") or ""),
            str(row.get("seendate") or ""),
            str(index),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_news_documents(corpus_path: Path) -> list[Any]:
    """Convert the labeled JSONL corpus into LangChain documents."""

    from langchain_core.documents import Document

    rows = read_jsonl(corpus_path)
    documents: list[Document] = []

    for index, row in enumerate(rows):
        text = _normalize_text(row.get("text") or row.get("content") or "")
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source_id": _build_source_id(row, index),
                    "title": str(row.get("title") or "Sin titulo"),
                    "url": str(row.get("normalized_url") or row.get("url") or ""),
                    "politician": str(row.get("politician") or "unknown"),
                    "tone_label": str(row.get("gdelt_tone_label") or row.get("politician_tone_label") or "unknown"),
                    "domain": str(row.get("domain") or ""),
                    "sourcecountry": str(row.get("sourcecountry") or ""),
                    "seendate": str(row.get("seendate") or ""),
                },
            )
        )

    return documents


def _normalize_question(question: str) -> str:
    return _normalize_text(question).casefold()


def _detect_politician_from_question(question: str) -> str | None:
    normalized_question = _normalize_question(question)
    for politician in DEFAULT_POLITICIANS:
        aliases = (politician.name, *politician.aliases)
        if any(alias.casefold() in normalized_question for alias in aliases):
            return politician.name
    return None


def _is_listing_news_request(question: str) -> bool:
    normalized_question = _normalize_question(question)
    return any(hint in normalized_question for hint in _LIST_NEWS_HINTS)


def _collect_politician_articles(
    vector_store: Any,
    politician: str,
    *,
    limit: int = 12,
) -> list[dict[str, str]]:
    payload = vector_store.get(where={"politician": politician})
    metadatas = payload.get("metadatas", []) or []
    documents = payload.get("documents", []) or []

    unique_articles: dict[str, dict[str, str]] = {}
    for metadata, document in zip(metadatas, documents):
        source_id = str(metadata.get("source_id") or "")
        if not source_id or source_id in unique_articles:
            continue
        unique_articles[source_id] = {
            "rank": "0",
            "title": str(metadata.get("title") or "Sin titulo"),
            "url": str(metadata.get("url") or ""),
            "politician": str(metadata.get("politician") or politician),
            "tone_label": str(metadata.get("tone_label") or "unknown"),
            "domain": str(metadata.get("domain") or ""),
            "seendate": str(metadata.get("seendate") or ""),
            "snippet": str(document or "")[:350].strip(),
        }

    cards = sorted(
        unique_articles.values(),
        key=lambda item: (item["seendate"], item["title"]),
        reverse=True,
    )[:limit]

    for index, card in enumerate(cards, start=1):
        card["rank"] = str(index)
    return cards


def load_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """Load the embedding model used to index and query the news corpus."""

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def chunk_already_indexed(vector_store: Any, source_id: str, chunk_text: str) -> bool:
    """Check whether a chunk already exists in Chroma for incremental updates."""

    probe_text = " ".join(chunk_text.split())[:150]
    if not probe_text:
        return True

    try:
        existing = vector_store.get(
            where={"source_id": source_id},
            where_document={"$contains": probe_text},
        )
    except Exception:
        return False
    return len(existing.get("ids", [])) > 0


def _create_vector_store(
    persist_dir: Path,
    *,
    collection_name: str,
    embeddings: Any,
) -> Any:
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def build_vector_store(
    corpus_path: Path,
    persist_dir: Path,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    force_rebuild: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Build or update the Chroma vector store from the labeled news corpus."""

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if force_rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)

    ensure_output_dir(persist_dir)
    embeddings = load_embeddings(embedding_model_name)
    vector_store = _create_vector_store(
        persist_dir,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    if not force_rebuild:
        try:
            existing_chunks = vector_store._collection.count()
            if existing_chunks > 0:
                return vector_store, {
                    "articles_indexed": len(build_news_documents(corpus_path)),
                    "chunks_added": 0,
                    "persist_dir": str(persist_dir),
                    "collection_name": collection_name,
                    "embedding_model": embedding_model_name,
                }
        except Exception:
            shutil.rmtree(persist_dir, ignore_errors=True)
            ensure_output_dir(persist_dir)
            vector_store = _create_vector_store(
                persist_dir,
                collection_name=collection_name,
                embeddings=embeddings,
            )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        add_start_index=True,
    )

    documents = build_news_documents(corpus_path)
    chunks_added = 0

    for document in documents:
        splits = text_splitter.split_documents([document])
        new_splits = []

        for split in splits:
            split.metadata["source_id"] = document.metadata["source_id"]
            if not chunk_already_indexed(vector_store, document.metadata["source_id"], split.page_content):
                new_splits.append(split)

        if new_splits:
            vector_store.add_documents(new_splits)
            chunks_added += len(new_splits)

    return vector_store, {
        "articles_indexed": len(documents),
        "chunks_added": chunks_added,
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "embedding_model": embedding_model_name,
    }


def retrieve_news_context(
    vector_store: Any,
    question: str,
    *,
    politician: str | None = None,
    k: int = 4,
) -> list[Any]:
    """Retrieve the most relevant chunks for a user question."""

    filter_by = {"politician": politician} if politician else None
    documents: list[Any] = []

    if filter_by:
        try:
            documents = vector_store.similarity_search(question, k=k, filter=filter_by)
        except Exception:
            fallback_docs = vector_store.similarity_search(question, k=max(k * 4, 12))
            documents = [
                doc for doc in fallback_docs
                if str(doc.metadata.get("politician") or "") == politician
            ][:k]
    else:
        documents = vector_store.similarity_search(question, k=k)

    if not documents and politician:
        documents = vector_store.similarity_search(question, k=k)

    return documents


def format_retrieved_context(documents: list[Any]) -> str:
    """Render the retrieved chunks into a grounded context block for the LLM."""

    blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        blocks.append(
            "\n".join(
                [
                    f"[{index}] Politico: {metadata.get('politician', 'unknown')}",
                    f"[{index}] Titulo: {metadata.get('title', 'Sin titulo')}",
                    f"[{index}] Fecha: {metadata.get('seendate', '')}",
                    f"[{index}] Medio: {metadata.get('domain', '')}",
                    f"[{index}] Tono: {metadata.get('tone_label', 'unknown')}",
                    f"[{index}] URL: {metadata.get('url', '')}",
                    f"[{index}] Contenido:\n{document.page_content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_source_cards(documents: list[Any]) -> list[dict[str, str]]:
    """Extract lightweight source cards to render citations in the UI."""

    cards = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        cards.append(
            {
                "rank": str(index),
                "title": str(metadata.get("title") or "Sin titulo"),
                "url": str(metadata.get("url") or ""),
                "politician": str(metadata.get("politician") or "unknown"),
                "tone_label": str(metadata.get("tone_label") or "unknown"),
                "domain": str(metadata.get("domain") or ""),
                "seendate": str(metadata.get("seendate") or ""),
                "snippet": document.page_content[:350].strip(),
            }
        )
    return cards


def answer_question(
    question: str,
    *,
    vector_store: Any,
    ollama_model: str,
    chat_history: list[dict[str, str]] | None = None,
    politician: str | None = None,
    k: int = 4,
) -> dict[str, Any]:
    """Answer a user question using retrieved political-news context."""

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    requested_politician = politician or _detect_politician_from_question(question)
    if requested_politician and _is_listing_news_request(question):
        sources = _collect_politician_articles(vector_store, requested_politician)
        if not sources:
            return {
                "answer": (
                    f"No he encontrado noticias indexadas para {requested_politician} en el corpus actual."
                ),
                "context": "",
                "sources": [],
            }

        lines = [
            f"He encontrado {len(sources)} noticias del corpus en las que aparece {requested_politician}:"
        ]
        for source in sources:
            domain = source["domain"] or "medio no disponible"
            date = source["seendate"] or "fecha no disponible"
            tone = source["tone_label"] or "unknown"
            lines.append(
                f"[{source['rank']}] {source['title']} ({domain}, {date}, tono {tone})"
            )

        return {
            "answer": "\n".join(lines),
            "context": "",
            "sources": sources,
        }

    documents = retrieve_news_context(vector_store, question, politician=requested_politician, k=k)
    context = format_retrieved_context(documents)
    sources = build_source_cards(documents)

    if not context:
        return {
            "answer": (
                "No he encontrado fragmentos relevantes en el corpus politico para responder con fiabilidad. "
                "Prueba a reformular la pregunta o quitar el filtro de politico."
            ),
            "context": "",
            "sources": [],
        }

    system_prompt = """
Eres un asistente RAG especializado en noticias politicas del corpus Ainee Politics.

Tu trabajo es responder solo con la informacion recuperada del corpus.
Reglas:
- Responde en espanol.
- No inventes hechos ni uses conocimiento externo.
- Si el contexto no basta, dilo con claridad.
- Si hay varias fuentes o matices, reflejalos.
- Cita las fuentes al final usando [1], [2], etc.
- Prioriza precision y trazabilidad antes que estilo.
""".strip()

    messages: list[Any] = [SystemMessage(content=system_prompt)]

    for turn in (chat_history or [])[-6:]:
        role = turn.get("role")
        content = str(turn.get("content") or "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    prompt = (
        f"Filtro de politico: {requested_politician or 'sin filtro'}\n\n"
        f"Pregunta del usuario:\n{question}\n\n"
        f"Contexto recuperado:\n{context}\n\n"
        "Redacta una respuesta breve pero informativa, basada unicamente en el contexto."
    )

    try:
        from langchain_ollama import ChatOllama

        model = ChatOllama(model=ollama_model, temperature=0)
        response = model.invoke(messages + [HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as error:
        return {
            "answer": (
                "No he podido generar la respuesta con Ollama. "
                "Comprueba que Ollama esta instalado, que el servicio esta levantado y que el modelo "
                f"`{ollama_model}` existe localmente.\n\n"
                "Pasos tipicos:\n"
                "1. Instalar Ollama.\n"
                f"2. Ejecutar `ollama pull {ollama_model}`.\n"
                "3. Arrancar o dejar disponible el servicio de Ollama.\n\n"
                f"Detalle tecnico: {error}"
            ),
            "context": context,
            "sources": sources,
            "error": str(error),
        }

    return {
        "answer": answer.strip(),
        "context": context,
        "sources": sources,
    }