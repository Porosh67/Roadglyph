"""The interface every vision-language backend implements."""

from __future__ import annotations

from typing import Protocol

from PIL import Image


class VisionBackend(Protocol):
    """A model that answers a text prompt about one image."""

    name: str

    def load(self) -> None:
        """Load weights onto the device. Called once at start-up."""

    def warmup(self) -> None:
        """Run a tiny request so kernels are compiled before the first real image."""

    def generate(
        self,
        image: Image.Image,
        *,
        system: str,
        user: str,
        max_new_tokens: int,
        timeout_s: float,
    ) -> str:
        """Return the model's text reply. Must stop generating after ``timeout_s``."""
