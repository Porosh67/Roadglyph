"""Qwen3-VL backend built on Hugging Face ``transformers``.

``torch`` and ``transformers`` are imported lazily so the rest of the package
(and its tests) work on machines without a GPU stack.

The mandated ROCm base image ships ``torch`` and ``PIL`` only. If loading the
processor fails with a ``torchvision`` error, see "Known risks" in the README.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from ..config import Settings

log = logging.getLogger("roadglyph.qwen")


class QwenVLBackend:
    name = "qwen3-vl"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._torch: Any = None
        self._processor: Any = None
        self._model: Any = None

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        path = self._settings.model_path
        use_gpu = torch.cuda.is_available()  # True on ROCm builds as well
        if not use_gpu and not self._settings.allow_cpu:
            raise RuntimeError(
                "No GPU visible to PyTorch. Start the container with "
                "--device=/dev/kfd --device=/dev/dri (or set ROADGLYPH_ALLOW_CPU=1 for dev)."
            )

        if use_gpu:
            total = torch.cuda.get_device_properties(0).total_memory
            cap = self._settings.vram_cap_gib * 1024**3
            if cap < total:
                # The grader rejects runs whose peak VRAM exceeds 48 GiB.
                torch.cuda.set_per_process_memory_fraction(cap / total)

        log.info("loading %s (gpu=%s)", path, use_gpu)
        self._processor = AutoProcessor.from_pretrained(path)
        dtype = torch.bfloat16 if use_gpu else torch.float32
        kwargs: dict[str, Any] = {"attn_implementation": "sdpa"}
        if use_gpu:
            kwargs["device_map"] = "cuda"
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, **kwargs)
        except TypeError:  # older transformers spell it torch_dtype
            self._model = AutoModelForImageTextToText.from_pretrained(
                path, torch_dtype=dtype, **kwargs
            )
        self._model.eval()
        log.info("model ready")

    def warmup(self) -> None:
        blank = Image.new("RGB", (224, 224), (255, 255, 255))
        self.generate(
            blank,
            system="You are an OCR engine.",
            user="Reply with the single word OK.",
            max_new_tokens=2,
            timeout_s=60.0,
        )

    # ---------------------------------------------------------------- inference
    def generate(
        self,
        image: Image.Image,
        *,
        system: str,
        user: str,
        max_new_tokens: int,
        timeout_s: float,
    ) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("backend not loaded")
        torch = self._torch
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": user}],
            },
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True
        ).to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                max_time=max(0.5, timeout_s),
            )
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text.strip()
