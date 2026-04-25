"""spaCy-based NLP enrichment for political news articles.

Politician tone is computed by running VADER on every sentence that mentions
the politician by alias, then averaging the compound scores.  This is more
targeted than GDELT's article-level V2Tone, which reflects the overall
emotional register of the article rather than how the politician is framed.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ainee_politics.domain.models import ArticleRow

if TYPE_CHECKING:
    import spacy

# Entity types relevant to political bias analysis
_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP"}

# Characters fed to spaCy per article (caps CPU time without losing key info)
_TEXT_CAP = 5_000

# Standard VADER thresholds for positive / negative
_VADER_POS_THRESHOLD = 0.05
_VADER_NEG_THRESHOLD = -0.05

_vader = SentimentIntensityAnalyzer()


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

    # Build alias set once for all downstream uses
    aliases_raw = row.get("mentioned_aliases", "") or ""
    aliases_lower = {a.strip().lower() for a in aliases_raw.split("|") if a.strip()}

    # Adjectives/nouns attached via amod or appos to politician entity mentions
    politician_adj = _extract_politician_modifiers(doc, aliases_lower)

    # Politician-specific tone: VADER on sentences that mention the politician
    tone_score, tone_n_sents, tone_label = _score_politician_sentences(doc, aliases_lower)

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
        "politician_tone_score": tone_score,
        "politician_tone_label": tone_label,
        "politician_tone_n_sentences": tone_n_sents,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
    }


def _score_politician_sentences(
    doc,
    aliases_lower: set[str],
) -> tuple[float, int, str]:
    """Score the tone of sentences that explicitly mention the politician.

    Uses VADER compound scores averaged across all sentences containing at least
    one politician alias.  Returns (mean_score, n_sentences, label) where label
    is 'positive', 'negative', 'neutral', or 'no_politician_sentences' when
    none of the sentences mention the politician.

    Scoring sentences rather than the full text avoids conflating the article's
    overall mood with the framing of the specific politician.
    """
    if not aliases_lower:
        return 0.0, 0, "no_politician_sentences"

    scores: list[float] = []
    for sent in doc.sents:
        sent_lower = sent.text.lower()
        if any(alias in sent_lower for alias in aliases_lower):
            compound = _vader.polarity_scores(sent.text)["compound"]
            scores.append(compound)

    if not scores:
        return 0.0, 0, "no_politician_sentences"

    mean_score = round(sum(scores) / len(scores), 4)
    if mean_score >= _VADER_POS_THRESHOLD:
        label = "positive"
    elif mean_score <= _VADER_NEG_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return mean_score, len(scores), label


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
