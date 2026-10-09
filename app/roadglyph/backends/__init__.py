"""Backend registry."""

from __future__ import annotations

from ..config import Settings
from .base import VisionBackend
from .mock import MockBackend


def create_backend(settings: Settings) -> VisionBackend:
    """Instantiate the backend named in ``settings.backend`` (not yet loaded)."""
    if settings.backend == "mock":
        return MockBackend()
    from .qwen import QwenVLBackend  # lazy: pulls in torch/transformers

    return QwenVLBackend(settings)


__all__ = ["MockBackend", "VisionBackend", "create_backend"]
