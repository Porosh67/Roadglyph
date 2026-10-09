"""The OCR pipeline: classify, transcribe under augmentation, vote.

Design goals
------------
* *Anytime*: after the first model call there is always a usable answer, and
  extra passes only run while the time budget allows. A late answer is worth
  nothing (the grader zeroes the whole run on a timeout) so the budget wins.
* *Deterministic convention*: the grader's transcription rules are enforced in
  :mod:`roadglyph.postprocess`, not left to the prompt.
* *Cheap consensus*: the original image and one enhanced variant usually agree;
  when they do we stop after two calls.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

from PIL import Image

from . import prompts
from .backends.base import VisionBackend
from .config import Settings
from .normalize import normalize
from .postprocess import postprocess
from .preprocess import build_variants, compute_stats
from .timing import Deadline

log = logging.getLogger("roadglyph.pipeline")

CLASSIFY_TOKENS = 8
SAFETY_MARGIN_S = 1.0
DEFAULT_CALL_S = 3.0


@dataclass
class Candidate:
    variant: str
    raw: str
    text: str


@dataclass
class Prediction:
    text: str
    confidence: float
    category: str
    candidates: list[Candidate] = field(default_factory=list)
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def vote(candidates: list[Candidate]) -> tuple[str, float]:
    """Pick the answer most variants agree on (earliest wins ties).

    Candidates are compared with the grader's normalisation, so ``7-ABC 123``
    and ``7ABC123`` count as the same answer.
    """
    groups: dict[str, list[Candidate]] = {}
    for cand in candidates:
        key = normalize(cand.text)
        if key:
            groups.setdefault(key, []).append(cand)
    if not groups:
        return "", 0.0
    best = max(groups.values(), key=len)  # insertion order => earliest wins ties
    agreement = len(best) / len(candidates)
    evidence = min(1.0, 0.5 + 0.25 * len(candidates))
    return best[0].text, round(min(0.99, agreement * evidence), 4)


class OCRPipeline:
    def __init__(self, backend: VisionBackend, settings: Settings) -> None:
        self._backend = backend
        self._settings = settings

    # ----------------------------------------------------------------- helpers
    def _call(
        self,
        image: Image.Image,
        user: str,
        max_new_tokens: int,
        deadline: Deadline,
        timings: list[float],
        notes: list[str],
        label: str,
    ) -> str | None:
        started = time.monotonic()
        try:
            out = self._backend.generate(
                image,
                system=prompts.SYSTEM_PROMPT,
                user=user,
                max_new_tokens=max_new_tokens,
                timeout_s=max(1.0, deadline.remaining() - 0.3),
            )
        except Exception as exc:  # noqa: BLE001 - a backend failure must not crash the run
            log.warning("backend call failed (%s): %s", label, exc)
            notes.append(f"{label}: backend error: {exc}")
            return None
        timings.append(time.monotonic() - started)
        return out

    @staticmethod
    def _can_afford(deadline: Deadline, timings: list[float]) -> bool:
        typical = (sum(timings) / len(timings)) if timings else DEFAULT_CALL_S
        return deadline.remaining() > SAFETY_MARGIN_S + 1.5 * typical

    @staticmethod
    def _agreed(candidates: list[Candidate]) -> bool:
        keys = [normalize(c.text) for c in candidates if normalize(c.text)]
        return any(keys.count(k) >= 2 for k in set(keys))

    # ------------------------------------------------------------------- main
    def predict(self, image: Image.Image, deadline: Deadline | None = None) -> Prediction:
        started = time.monotonic()
        deadline = deadline or Deadline(self._settings.request_budget_s)
        notes: list[str] = []
        timings: list[float] = []

        stats = compute_stats(image)
        variants = build_variants(
            image,
            stats,
            max_variants=self._settings.max_variants,
            min_side=self._settings.min_side,
            max_side=self._settings.max_side,
        )

        category = prompts.OTHER
        reply = self._call(
            variants[0][1],
            prompts.CLASSIFY_PROMPT,
            CLASSIFY_TOKENS,
            deadline,
            timings,
            notes,
            "classify",
        )
        if reply is not None:
            category = prompts.parse_category(reply)
        user_prompt = prompts.transcribe_prompt(category)

        candidates: list[Candidate] = []
        for index, (name, variant_image) in enumerate(variants):
            if index > 0 and not self._can_afford(deadline, timings):
                notes.append("time budget reached; skipped remaining variants")
                break
            raw = self._call(
                variant_image,
                user_prompt,
                self._settings.max_new_tokens,
                deadline,
                timings,
                notes,
                name,
            )
            if raw is None:
                continue
            candidates.append(Candidate(name, raw, postprocess(category, raw)))
            if self._agreed(candidates):
                break

        text, confidence = vote(candidates)
        return Prediction(
            text=text,
            confidence=confidence,
            category=category,
            candidates=candidates,
            elapsed_s=round(time.monotonic() - started, 3),
            notes=notes,
        )
