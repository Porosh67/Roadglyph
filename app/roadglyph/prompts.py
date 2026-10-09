"""Prompts and category handling.

The pipeline first asks the model what kind of object it is looking at, then
uses a category-specific transcription prompt. Each prompt encodes the grader's
transcription convention (see the challenge brief).
"""

from __future__ import annotations

import re

PLATE = "PLATE"
STOP = "STOP"
SPEED_LIMIT = "SPEED_LIMIT"
ADVISORY_PLAQUE = "ADVISORY_PLAQUE"
WARNING_SIGN = "WARNING_SIGN"
OTHER = "OTHER"

CATEGORIES = (PLATE, STOP, SPEED_LIMIT, ADVISORY_PLAQUE, WARNING_SIGN, OTHER)

SYSTEM_PROMPT = (
    "You are a precise OCR engine for traffic scenes. You transcribe text exactly as it is "
    "printed, character by character, including non-Latin characters. You never translate, "
    "explain, describe or add words that are not in the image."
)

CLASSIFY_PROMPT = (
    "What is the main object in this image? Answer with exactly one label:\n"
    "PLATE - a vehicle licence plate\n"
    "STOP - a stop sign\n"
    "SPEED_LIMIT - a speed limit sign showing the words SPEED LIMIT and a number\n"
    "ADVISORY_PLAQUE - a small plaque that shows only a number (for example an advisory speed)\n"
    "WARNING_SIGN - a work zone, warning or other sign with several words\n"
    "OTHER - anything else\n"
    "Reply with the label only."
)

_TRANSCRIBE = {
    PLATE: (
        "This is a vehicle licence plate. Output only the registration characters: letters, "
        "digits and, on Chinese plates, the leading province character. Ignore state or region "
        "names, slogans, stickers, frames and dealer text. Keep reading order. "
        "Output the characters only, nothing else."
    ),
    STOP: (
        "This is a traffic sign. Output the exact word printed on it. "
        "Output the text only, nothing else."
    ),
    SPEED_LIMIT: (
        "Transcribe all the text printed on this speed limit sign, top to bottom, on a single "
        "line (for example: SPEED LIMIT 65). Output the text only, nothing else."
    ),
    ADVISORY_PLAQUE: (
        "This is a small advisory plaque. Output only the number printed on it, with no unit "
        "and no words. Output the number only, nothing else."
    ),
    WARNING_SIGN: (
        "Transcribe all the text on this sign from top to bottom, one line per line of text, "
        "exactly as printed. Output the text only, nothing else."
    ),
    OTHER: (
        "Transcribe all the text visible in this image in reading order, exactly as printed. "
        "Output the text only, nothing else."
    ),
}

_SYNONYMS = (
    (ADVISORY_PLAQUE, ("ADVISORY", "PLAQUE")),
    (SPEED_LIMIT, ("SPEED_LIMIT", "SPEEDLIMIT", "SPEED")),
    (PLATE, ("PLATE", "LICENSE", "LICENCE", "NUMBER_PLATE")),
    (STOP, ("STOP",)),
    (WARNING_SIGN, ("WARNING", "WORK_ZONE", "WORKZONE", "ROAD_WORK", "CONSTRUCTION")),
    (OTHER, ("OTHER",)),
)


def transcribe_prompt(category: str) -> str:
    return _TRANSCRIBE.get(category, _TRANSCRIBE[OTHER])


def parse_category(raw: str) -> str:
    """Map the model's free-text reply onto one of ``CATEGORIES``."""
    cleaned = re.sub(r"[\s\-]+", "_", (raw or "").upper()).strip("_.:*`\"'")
    if not cleaned:
        return OTHER
    for label in CATEGORIES:
        if cleaned == label:
            return label
    for label, needles in _SYNONYMS:
        if any(needle in cleaned for needle in needles):
            return label
    return OTHER
