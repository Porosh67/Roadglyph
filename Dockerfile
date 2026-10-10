# syntax=docker/dockerfile:1.7
#
# RoadGlyph: scene-text OCR for licence plates and road signs (AMD ROCm).
#
# Rules this file follows (they are hard gates in the challenge):
#   * the FINAL image starts from the mandated ROCm base image;
#   * never squash/flatten the image (it destroys the layer identity the grader checks);
#   * everything lives under /app, weights under /models;
#   * no secrets, tokens or .env files are ever copied in (the image is public).
#
# Build:   docker build -t roadglyph:local .
# Model weights are NOT baked (official rules allow runtime download).
# Image stays under the 60 GiB limit and is easy to push.

ARG BASE_IMAGE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
FROM ${BASE_IMAGE}

ARG MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
ARG MODEL_REVISION=main
ARG TORCHVISION_SPEC=""

LABEL org.opencontainers.image.title="RoadGlyph" \
      org.opencontainers.image.description="Scene-text OCR for licence plates and road signs on AMD ROCm" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

# 1) Python dependencies. torch is pinned to the ROCm build from the base image.
COPY app/requirements.txt /app/requirements.txt
RUN python3 -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton-rocm)==' > /tmp/torch-constraints.txt || true \
 && echo "pinned from base image:" && cat /tmp/torch-constraints.txt \
 && python3 -m pip install -c /tmp/torch-constraints.txt -r /app/requirements.txt \
 && if [ -n "${TORCHVISION_SPEC}" ]; then python3 -m pip install --no-deps ${TORCHVISION_SPEC}; fi \
 && python3 -c "import torch; assert torch.version.hip, 'torch lost its ROCm build'; print('torch', torch.__version__, 'hip', torch.version.hip)" \
 && rm -f /tmp/torch-constraints.txt

# 2) NO model bake — download at runtime (allowed by official rules).
#    Keeps image ~10-20 GB instead of 80+ GB.
RUN mkdir -p /models

# 3) Application code
COPY app/ /app/
COPY scripts/ /app/scripts/
RUN mkdir -p /app/input /app/output && chmod 777 /app/output

# HF id as model path → transformers downloads on first load.
# Offline flags OFF so runtime download works.
ENV ROADGLYPH_MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
    ROADGLYPH_MODE=server \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD ["python3", "/app/app.py", "--healthcheck"]

CMD ["python3", "/app/app.py", "--serve"]