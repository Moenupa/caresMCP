"""Helpers for resolving a query against named choices."""

import re
from typing import Any

from rapidfuzz import fuzz, process

MIN_FUZZY_SCORE = 60


def _normalize(value: str) -> str:
    """Case-fold a name and remove punctuation and whitespace."""
    return re.sub(r"[\W_]", "", value.casefold())


def _prepare(value: str) -> str:
    """Case-fold a name and replace punctuation with spaces for scoring."""
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def fuzzy_match(query: str, **kwargs: Any) -> Any:
    """Return the value associated with the exact or unique fuzzy key match.

    Matching precedence is:

    1. Exact key match.
    2. Case and punctuation insensitive exact match.
    3. A unique prefix/substring match.
    4. A unique best RapidFuzz match scoring at least 60/100.

    Ambiguous or unmatched queries raise ``ValueError`` rather than choosing a
    value arbitrarily.

    Args:
        query: Name to resolve.
        **kwargs: Candidate names and their associated values.

    Returns:
        The value associated with the matched candidate name.

    Raises:
        ValueError: If the query is empty, ambiguous, or has no match.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not kwargs:
        raise ValueError("at least one candidate is required")

    if query in kwargs:
        return kwargs[query]

    normalized_query = _normalize(query)
    if not normalized_query:
        raise ValueError("query must contain at least one letter or number")
    exact_matches = [key for key in kwargs if _normalize(key) == normalized_query]
    if len(exact_matches) == 1:
        return kwargs[exact_matches[0]]
    if len(exact_matches) > 1:
        raise ValueError(f"Ambiguous match for {query!r}: {exact_matches}")

    matches = process.extract(
        _prepare(query),
        {key: _prepare(key) for key in kwargs},
        scorer=fuzz.WRatio,
        limit=2,
        score_cutoff=MIN_FUZZY_SCORE,
    )
    if matches:
        _, best_score, best_name = matches[0]
        if len(matches) > 1 and matches[1][1] == best_score:
            tied_names = [name for _, score, name in matches if score == best_score]
            raise ValueError(f"Ambiguous match for {query!r}: {tied_names}")
        return kwargs[best_name]

    raise ValueError(f"No match found for {query!r}")
