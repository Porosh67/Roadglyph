"""A deterministic stand-in backend for tests and CPU-only development."""

from __future__ import annotations

import time
from collections.abc import Callable

from PIL import Image

from .. import prompts

Responder = Callable[[Image.Image, str, str], str]


def _default_responder(image: Image.Image, system: str, user: str) -> str:
    if user == prompts.CLASSIFY_PROMPT:
        return prompts.OTHER
    return "MOCK"


class MockBackend:
    """Returns canned answers; never touches a GPU.

    ``responder(image, system, user)`` decides the reply, ``delay_s`` simulates
    inference latency (bounded by the caller's ``timeout_s``).
    """

    name = "mock"

    def __init__(self, responder: Responder | None = None, delay_s: float = 0.0) -> None:
        self._responder = responder or _default_responder
        self._delay_s = delay_s
        self.calls: list[tuple[str, str]] = []

    def load(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def generate(
        self,
        image: Image.Image,
        *,
        system: str,
        user: str,
        max_new_tokens: int,
        timeout_s: float,
    ) -> str:
        self.calls.append((system, user))
        if self._delay_s > 0:
            time.sleep(min(self._delay_s, max(0.0, timeout_s)))
        return self._responder(image, system, user)
