"""The official answer normalisation used by the grader.

Both the expected answer and ours are normalised before comparison:

* converted to upper case,
* all whitespace removed,
* the characters ``-  .  ·  _`` removed.

Keeping an exact copy here lets us compare candidates the same way the
grader does (voting, local evaluation, tests).
"""

from __future__ import annotations

_DROPPED = frozenset("-.·_")


def normalize(text: str) -> str:
    """Return ``text`` in the grader's canonical form."""
    return "".join(ch for ch in text.upper() if not ch.isspace() and ch not in _DROPPED)


def matches(answer: str, expected: str) -> bool:
    """True when ``answer`` would pass the grader for ``expected``."""
    return normalize(answer) == normalize(expected)
