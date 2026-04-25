"""spaCy-based NLP enrichment for political news articles."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

from ainee_politics.domain.models import ArticleRow

if TYPE_CHECKING:
    import spacy

# Entity types relevant to political bias analysis
_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP"}

# Characters fed to spaCy per article (caps CPU time without losing key info)
_TEXT_CAP = 5_000


def load_spacy_model(model_name: str = "en_core_web_lg"):
    try:
        import spacy as _spacy
        return _spacy.load(model_name)
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model_name}' not found.\n"
            f"Run: python -m spacy download {model_name}"
        )


def enrich_rows(rows: list[ArticleRow], nlp, batch_size: int = 32) -> list[ArticleRow]:
    """Enrich a list of rows using nlp.pipe() for efficiency."""
    texts = [(r.get("text", "") or "")[:_TEXT_CAP] for r in rows]
    enriched: list[ArticleRow] = []
    for row, doc in zip(rows, nlp.pipe(texts, batch_size=batch_size)):
        enriched.append(_enrich_from_doc(row, doc))
    return enriched


def _enrich_from_doc(row: ArticleRow, doc) -> ArticleRow:
    """Add spaCy-derived fields to a single row."""
    # NER: top entities by frequency, filtered to politically relevant types
    ent_counter: Counter = Counter(
        (ent.text.strip(), ent.label_)
        for ent in doc.ents
        if ent.label_ in _ENTITY_TYPES and len(ent.text.strip()) > 1
    )
    top_entities = [
        {"text": text, "label": label, "count": count}
        for (text, label), count in ent_counter.most_common(15)
    ]

    # Adjectives/nouns attached via amod or appos to politician entity mentions
    aliases_raw = row.get("mentioned_aliases", "") or ""
    aliases_lower = {a.strip().lower() for a in aliases_raw.split("|") if a.strip()}
    politician_adj = _extract_politician_modifiers(doc, aliases_lower)

    # Sentence-level stats
    sents = list(doc.sents)
    sentence_count = len(sents)
    avg_sentence_length = (
        round(sum(len(s) for s in sents) / sentence_count, 2) if sentence_count else 0.0
    )

    return {
        **row,
        "spacy_entities": json.dumps(top_entities, ensure_ascii=False),
        "politician_adjectives": "|".join(politician_adj),
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
    }


def _extract_politician_modifiers(doc, aliases_lower: set[str]) -> list[str]:
    """Return lemmatised modifiers (amod/appos) attached to politician entity tokens.

    Mirrors the adjective-extraction technique from notebook CODE 52 (TXT.1).
    """
    if not aliases_lower:
        return []

    modifiers: list[str] = []
    for ent in doc.ents:
        if ent.label_ not in ("PERSON",):
            continue
        ent_text_lower = ent.text.lower()
        # Match if any alias is contained in the entity text or vice-versa
        if not any(
            alias in ent_text_lower or ent_text_lower in alias
            for alias in aliases_lower
        ):
            continue
        for token in ent:
            for child in token.children:
                if child.dep_ in ("amod", "appos") and child.pos_ in ("ADJ", "NOUN"):
                    modifiers.append(child.lemma_.lower())
    return modifiers
