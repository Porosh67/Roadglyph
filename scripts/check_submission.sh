#!/usr/bin/env bash
# Pre-submission checks, mirroring the grader's hard gates.
#
#   ./scripts/check_submission.sh IMAGE[:TAG] [SAMPLES_DIR]
#
# Needs docker and an AMD GPU (/dev/kfd, /dev/dri). If SAMPLES_DIR is omitted,
# synthetic PNG/JPEG/TIFF images are generated. Exit code 0 means every gate passed.
set -uo pipefail

IMAGE="${1:?usage: $0 IMAGE[:TAG] [SAMPLES_DIR]}"
SAMPLES="${2:-}"
BASE_IMAGE="${BASE_IMAGE:-rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0}"
SIZE_LIMIT_GIB=60
STARTUP_LIMIT_S=600
PER_IMAGE_LIMIT_S=30
NAME="roadglyph-check-$$"
FAILED=0

pass() { printf '  [ ok ] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; FAILED=1; }
info() { printf '\n== %s\n' "$1"; }

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; [ -n "${TMP:-}" ] && rm -rf "$TMP"; }
trap cleanup EXIT

command -v docker >/dev/null || { echo "docker not found"; exit 2; }
TMP="$(mktemp -d)"

info "1. Image size (uncompressed, limit ${SIZE_LIMIT_GIB} GiB)"
BYTES="$(docker image inspect "$IMAGE" --format '{{.Size}}')" || { echo "image not found: $IMAGE"; exit 2; }
GIB="$(python3 -c "print(round($BYTES/1024**3, 2))")"
if python3 -c "import sys; sys.exit(0 if $BYTES <= $SIZE_LIMIT_GIB*1024**3 else 1)"; then
  pass "${GIB} GiB"
else
  fail "${GIB} GiB is over the limit"
fi

info "2. Built on the mandated base image (layer identity, no squash)"
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull -q "$BASE_IMAGE" >/dev/null 2>&1
if python3 - "$IMAGE" "$BASE_IMAGE" <<'PY'
import json, subprocess, sys
def layers(ref):
    out = subprocess.check_output(["docker", "image", "inspect", ref, "--format", "{{json .RootFS.Layers}}"])
    return json.loads(out)
mine, base = layers(sys.argv[1]), layers(sys.argv[2])
sys.exit(0 if mine[: len(base)] == base else 1)
PY
then pass "base layers are the first layers of the image"; else fail "image is not built on $BASE_IMAGE (squashed or wrong base?)"; fi

info "3. Start the container (startup limit ${STARTUP_LIMIT_S}s)"
docker run -d --name "$NAME" --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --security-opt seccomp=unconfined "$IMAGE" >/dev/null || { fail "container did not start"; exit 1; }
START="$(date +%s)"
READY=0
while [ $(( $(date +%s) - START )) -lt "$STARTUP_LIMIT_S" ]; do
  if docker exec "$NAME" python3 /app/app.py --healthcheck >/dev/null 2>&1; then READY=1; break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then break; fi
  sleep 3
done
if [ "$READY" = 1 ]; then pass "model server ready after $(( $(date +%s) - START ))s"; else
  fail "server not ready in time"; docker logs --tail 40 "$NAME"; exit 1; fi

info "4. Run every sample exactly like the grader (new process per image, limit ${PER_IMAGE_LIMIT_S}s)"
if [ -z "$SAMPLES" ]; then
  SAMPLES="$TMP/samples"
  python3 "$(dirname "$0")/make_synthetic_dataset.py" --out "$SAMPLES" --count 6 >/dev/null
fi
docker exec "$NAME" mkdir -p /app/input /app/output
TOTAL=0
for f in "$SAMPLES"/*.png "$SAMPLES"/*.jpg "$SAMPLES"/*.jpeg "$SAMPLES"/*.tif "$SAMPLES"/*.tiff; do
  [ -e "$f" ] || continue
  base="$(basename "$f")"; stem="${base%.*}"
  docker cp "$f" "$NAME:/app/input/$base" >/dev/null
  t0="$(date +%s.%N)"
  docker exec "$NAME" python3 /app/app.py --input-image "/app/input/$base"; rc=$?
  t1="$(date +%s.%N)"
  secs="$(python3 -c "print(round($t1-$t0, 2))")"
  out="$(docker exec "$NAME" cat "/app/output/${stem}_output.json" 2>/dev/null)"
  if [ $rc -eq 0 ] && python3 -c "import json,sys; d=json.loads(sys.argv[1]); assert isinstance(d['text'], str)" "$out" 2>/dev/null; then
    ok=1; else ok=0; fi
  if [ $ok -eq 1 ] && python3 -c "import sys; sys.exit(0 if $secs < $PER_IMAGE_LIMIT_S else 1)"; then
    pass "$base -> $out  (${secs}s)"
  else
    fail "$base -> '${out:-<no file>}' rc=$rc (${secs}s)"
  fi
  TOTAL=$((TOTAL + 1))
done
[ "$TOTAL" -gt 0 ] || fail "no sample images found in $SAMPLES"

info "5. GPU memory (must stay between 1 and 48 GiB)"
if command -v amd-smi >/dev/null; then
  amd-smi metric --mem 2>/dev/null | grep -i -E 'used_vram|total_vram' | head -4 | sed 's/^/  /'
  echo "  (the server keeps the model resident, so this is the steady-state figure)"
else
  echo "  amd-smi not found on the host; check manually: watch -n1 'amd-smi metric --mem'"
fi

info "Result"
if [ "$FAILED" = 0 ]; then echo "  all checks passed"; else echo "  SOME CHECKS FAILED"; fi
echo "  Reminder: these samples are synthetic. Also test the real sample images from the brief."
exit "$FAILED"
