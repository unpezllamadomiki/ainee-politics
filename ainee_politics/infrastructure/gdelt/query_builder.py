"""Helpers to build GDELT DOC API queries."""

from __future__ import annotations

from ainee_politics.config import GDELT_SOURCE_LANGUAGE
from ainee_politics.domain.models import Politician


def build_alias_query(aliases: tuple[str, ...]) -> str:
    """Build the quoted alias clause used in GDELT queries."""

    return " OR ".join(f'"{alias}"' for alias in aliases)


def build_query(politician: Politician) -> str:
    """Build the DOC API query for a politician."""

    alias_query = build_alias_query(politician.aliases)
    return f"({alias_query}) sourcelang:{GDELT_SOURCE_LANGUAGE}"