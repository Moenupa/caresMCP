import re

TODO_REGEX = [
    r"\bTODO\b",
    r"\btodo\b",
]


def has_todo(description: str) -> bool:
    """Return True if the description looks like a TODO item."""
    return any(re.search(regex, description) for regex in TODO_REGEX)
