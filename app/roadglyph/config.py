"""Runtime configuration, resolved from environment variables.

Every knob has a safe default so the container works with zero configuration.
Override with ``ROADGLYPH_*`` environment variables (see README).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BAKED_MODEL_DIR = "/models/model"

MODES = ("auto", "server", "standalone")
BACKENDS = ("qwen", "mock")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_model_path() -> str:
    explicit = os.environ.get("ROADGLYPH_MODEL_PATH")
    if explicit:
        return explicit
    if Path(BAKED_MODEL_DIR).is_dir():
        return BAKED_MODEL_DIR
    return DEFAULT_MODEL_ID


@dataclass(frozen=True)
class Settings:
    """Immutable settings object passed through the whole application."""

    # How ``app.py --input-image`` obtains a prediction.
    #   auto       - use the resident server if it answers, else run in-process
    #   server     - always use the resident server (never load a model per call)
    #   standalone - always load the model in this process (slow, for development)
    mode: str = "auto"
    backend: str = "qwen"
    model_path: str = DEFAULT_MODEL_ID
    socket_path: str = "/tmp/roadglyph.sock"
    output_dir: str = "/app/output"

    # Time budgets (seconds). The harness allows 30 s per image; we stay below.
    client_deadline_s: float = 26.0
    request_budget_s: float = 21.0

    # Pipeline tuning.
    max_variants: int = 3
    min_side: int = 448
    max_side: int = 1792
    max_new_tokens: int = 64

    # Resource guard-rails.
    vram_cap_gib: float = 44.0
    allow_cpu: bool = False

    trace_dir: str = ""
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        mode = _env_str("ROADGLYPH_MODE", "auto").lower()
        backend = _env_str("ROADGLYPH_BACKEND", "qwen").lower()
        return cls(
            mode=mode if mode in MODES else "auto",
            backend=backend if backend in BACKENDS else "qwen",
            model_path=_default_model_path(),
            socket_path=_env_str("ROADGLYPH_SOCKET", "/tmp/roadglyph.sock"),
            output_dir=_env_str("ROADGLYPH_OUTPUT_DIR", "/app/output"),
            client_deadline_s=_env_float("ROADGLYPH_CLIENT_DEADLINE_S", 26.0),
            request_budget_s=_env_float("ROADGLYPH_REQUEST_BUDGET_S", 21.0),
            max_variants=max(1, _env_int("ROADGLYPH_MAX_VARIANTS", 3)),
            min_side=_env_int("ROADGLYPH_MIN_SIDE", 448),
            max_side=_env_int("ROADGLYPH_MAX_SIDE", 1792),
            max_new_tokens=_env_int("ROADGLYPH_MAX_NEW_TOKENS", 64),
            vram_cap_gib=_env_float("ROADGLYPH_VRAM_CAP_GIB", 44.0),
            allow_cpu=_env_bool("ROADGLYPH_ALLOW_CPU", False),
            trace_dir=_env_str("ROADGLYPH_TRACE_DIR", ""),
            log_level=_env_str("ROADGLYPH_LOG_LEVEL", "INFO").upper(),
        )
