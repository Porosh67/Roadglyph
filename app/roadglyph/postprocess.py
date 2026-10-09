"""Deterministic post-processing of raw model output.

Models are good at *reading* but sloppy at following a transcription
convention, so the convention is enforced in code:

* licence plates: drop jurisdiction banners and slogans, keep the Chinese
  province character as part of the registration;
* stop signs: ``STOP``;
* speed limit signs: keep the printed words together with the number;
* advisory plaques: digits only, no unit;
* everything else: join lines top to bottom with single spaces.
"""

from __future__ import annotations

import re
import unicodedata

from . import prompts

_FENCE = re.compile(r"```[a-zA-Z]*|```")
_LEAD_IN = re.compile(
    r"^\s*(?:the\s+)?(?:text|answer|output|result|transcription|"
    r"(?:license|licence)\s+plate(?:\s+number)?|plate(?:\s+number)?)\s*(?:is\s*:?|:|-)\s*",
    re.IGNORECASE,
)
_DOTS = str.maketrans({"・": "·", "•": "·", "‧": "·", "∙": "·", "⋅": "·", "●": "·"})

_CN_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
_CN_PLATE = re.compile(
    rf"([{_CN_PROVINCES}])\s*([A-Z])\s*[·.\-\s]?\s*([A-Z0-9]{{5,6}}[挂学警港澳]?)"
)

_US_STATES = (
    "ALABAMA",
    "ALASKA",
    "ARIZONA",
    "ARKANSAS",
    "CALIFORNIA",
    "COLORADO",
    "CONNECTICUT",
    "DELAWARE",
    "FLORIDA",
    "GEORGIA",
    "HAWAII",
    "IDAHO",
    "ILLINOIS",
    "INDIANA",
    "IOWA",
    "KANSAS",
    "KENTUCKY",
    "LOUISIANA",
    "MAINE",
    "MARYLAND",
    "MASSACHUSETTS",
    "MICHIGAN",
    "MINNESOTA",
    "MISSISSIPPI",
    "MISSOURI",
    "MONTANA",
    "NEBRASKA",
    "NEVADA",
    "NEW HAMPSHIRE",
    "NEW JERSEY",
    "NEW MEXICO",
    "NEW YORK",
    "NORTH CAROLINA",
    "NORTH DAKOTA",
    "OHIO",
    "OKLAHOMA",
    "OREGON",
    "PENNSYLVANIA",
    "RHODE ISLAND",
    "SOUTH CAROLINA",
    "SOUTH DAKOTA",
    "TENNESSEE",
    "TEXAS",
    "UTAH",
    "VERMONT",
    "VIRGINIA",
    "WASHINGTON",
    "WEST VIRGINIA",
    "WISCONSIN",
    "WYOMING",
    "DISTRICT OF COLUMBIA",
)
_SLOGANS = (
    "THE LONE STAR STATE",
    "LONE STAR STATE",
    "THE GOLDEN STATE",
    "GOLDEN STATE",
    "THE EMPIRE STATE",
    "EMPIRE STATE",
    "THE SUNSHINE STATE",
    "SUNSHINE STATE",
    "THE GARDEN STATE",
    "GARDEN STATE",
    "LAND OF ENCHANTMENT",
    "LAND OF LINCOLN",
    "GREAT LAKES STATE",
    "LIVE FREE OR DIE",
    "THE PEACH STATE",
    "PEACH STATE",
    "THE GRAND CANYON STATE",
    "GRAND CANYON STATE",
    "THE BEEHIVE STATE",
    "BEEHIVE STATE",
    "THE SILVER STATE",
    "SILVER STATE",
    "THE BUCKEYE STATE",
    "THE KEYSTONE STATE",
    "KEYSTONE STATE",
    "THE EVERGREEN STATE",
    "EVERGREEN STATE",
    "THE SUNFLOWER STATE",
    "THE VOLUNTEER STATE",
    "THE BAY STATE",
    "THE OCEAN STATE",
    "THE FIRST STATE",
    "THE PALMETTO STATE",
    "THE MOUNTAIN STATE",
    "WILD AND WONDERFUL",
    "OKLAHOMA IS OK",
    "ENDLESS MOUNTAINS",
    "THE SHOW ME STATE",
    "SHOW ME STATE",
    "DEFEND FREEDOM",
    "SCENIC",
)
# Longest phrases first so "NEW YORK" wins over any shorter overlapping phrase.
_BANNER_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(p) for p in sorted(set(_US_STATES + _SLOGANS), key=len, reverse=True))
    + r")\b"
)


def _clean_lines(raw: str) -> list[str]:
    text = _FENCE.sub("", raw or "")
    text = unicodedata.normalize("NFKC", text).translate(_DOTS)
    text = text.replace("**", "").replace("`", "")
    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip().strip("\"'“”‘’").strip()
        line = _LEAD_IN.sub("", line).strip()
        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)
    return lines


def _alnum(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def _strip_banners(line: str) -> str:
    return re.sub(r"\s+", " ", _BANNER_RE.sub(" ", line.upper())).strip()


def _plate(lines: list[str]) -> str:
    joined = " ".join(lines)
    chinese = _CN_PLATE.search(joined.upper())
    if chinese:
        province, letter, rest = chinese.groups()
        return f"{province}{letter}·{rest}"

    best_text, best_score = "", -1
    for line in lines:
        stripped = _strip_banners(line)
        core = _alnum(stripped)
        if not core:
            continue
        score = 0
        if any(ch.isdigit() for ch in core):
            score += 2
        if any(ch.isalpha() for ch in core):
            score += 1
        if 5 <= len(core) <= 8:
            score += 2
        elif 2 <= len(core) <= 9:
            score += 1
        if score > best_score:
            best_text, best_score = stripped, score
    if best_text:
        return best_text
    return max(lines, key=len).upper() if lines else ""


def _stop(lines: list[str]) -> str:
    text = " ".join(lines).upper()
    if re.search(r"\bST[O0]P\b", text):
        return "STOP"
    return text


def _speed_limit(lines: list[str]) -> str:
    text = " ".join(lines).upper()
    numbers = re.findall(r"\d{1,3}", text)
    if not numbers:
        return text
    if "SPEED" in text or "LIMIT" in text:
        return text
    return f"SPEED LIMIT {numbers[-1]}"


def _advisory_plaque(lines: list[str]) -> str:
    text = " ".join(lines)
    runs = re.findall(r"\d+", text)
    if not runs:
        return text.upper()
    preferred = [r for r in runs if 2 <= len(r) <= 3]
    return (preferred or runs)[0]


def _join_lines(lines: list[str]) -> str:
    kept = [line for line in lines if _alnum(line)]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def postprocess(category: str, raw: str) -> str:
    """Apply the transcription convention for ``category`` to ``raw`` model output."""
    lines = _clean_lines(raw)
    if not lines:
        return ""
    if category == prompts.PLATE:
        return _plate(lines)
    if category == prompts.STOP:
        return _stop(lines)
    if category == prompts.SPEED_LIMIT:
        return _speed_limit(lines)
    if category == prompts.ADVISORY_PLAQUE:
        return _advisory_plaque(lines)
    return _join_lines(lines)
