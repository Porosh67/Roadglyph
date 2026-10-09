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
# Pin the weights for a reproducible build:
#          docker build --build-arg MODEL_REVISION=<commit-sha> -t roadglyph:local .

ARG BASE_IMAGE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
FROM ${BASE_IMAGE}

ARG MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
ARG MODEL_REVISION=main
# Only needed if the processor complains about torchvision (see README > Known risks), e.g.
#   --build-arg TORCHVISION_SPEC="torchvision==<matching> --index-url https://download.pytorch.org/whl/rocm<ver>"
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

# 1) Python dependencies. torch/torchvision/triton are pinned to what the base already ships so
#    pip can never swap the ROCm build of torch for a CUDA wheel; the last line fails the
#    build if that ever happens.
COPY app/requirements.txt /app/requirements.txt
RUN python3 -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton-rocm)==' > /tmp/torch-constraints.txt || true \
 && echo "pinned from base image:" && cat /tmp/torch-constraints.txt \
 && python3 -m pip install -c /tmp/torch-constraints.txt -r /app/requirements.txt \
 && if [ -n "${TORCHVISION_SPEC}" ]; then python3 -m pip install --no-deps ${TORCHVISION_SPEC}; fi \
 && python3 -c "import torch; assert torch.version.hip, 'torch lost its ROCm build'; print('torch', torch.__version__, 'hip', torch.version.hip)" \
 && rm -f /tmp/torch-constraints.txt

# 2) Model weights are baked into the image: the grader's network access is not guaranteed and
#    downloading at start-up would eat into the 10 minute budget.
COPY scripts/download_model.py /tmp/download_model.py
RUN python3 /tmp/download_model.py --model-id "${MODEL_ID}" --revision "${MODEL_REVISION}" --dest /models/model \
 && rm -f /tmp/download_model.py \
 && rm -rf /root/.cache

# 3) Application code (last, so code edits do not invalidate the big layers above).
COPY app/ /app/
RUN mkdir -p /app/input /app/output && chmod 777 /app/output

ENV ROADGLYPH_MODEL_PATH=/models/model \
    ROADGLYPH_MODE=server \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# The grader runs `python3 /app/app.py --input-image ...` once per image, each in a new process.
# The container's main process is therefore a resident server that keeps the model loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD ["python3", "/app/app.py", "--healthcheck"]

CMD ["python3", "/app/app.py", "--serve"]
