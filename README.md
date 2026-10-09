# RoadGlyph

**Scene-text OCR for licence plates and road signs, built for AMD ROCm.**

RoadGlyph reads the text in one photo of a plate or traffic sign and writes it in the exact
convention the AMD AI League *Mini Challenge 2* grader expects. It combines a
vision-language model (Qwen3-VL) with image-aware test-time augmentation, category-specific
prompts, deterministic post-processing and consensus voting, all inside a hard time budget.

```
image ─► load & normalise ─► analyse (dark? noisy? glare? blurry?) ─► build 1-3 variants
                                                                          │
         ┌────────────────────────────────────────────────────────────────┘
         ▼
   classify (PLATE / STOP / SPEED_LIMIT / ADVISORY_PLAQUE / WARNING_SIGN / OTHER)
         ▼
   transcribe each variant with a category prompt ─► enforce the convention in code ─► vote
         ▼
   /app/output/<name>_output.json   {"text": "...", "confidence": 0.94}
```

## Why it is built this way

| Challenge rule | What RoadGlyph does |
|---|---|
| A new `python3 /app/app.py` process per image, 30 s each | A **resident server** (container `CMD`) keeps the model loaded; `app.py` is a tiny client (~0.2 s overhead). |
| One timeout zeroes the whole run | An **anytime pipeline** (always has an answer after the first model call), a per-image **deadline**, and a **watchdog** that writes the answer file and exits before 30 s. A valid JSON file is on disk from the first moment. |
| Transcription convention (drop state banners, keep `京 A`, digits only for plaques...) | Enforced in `postprocess.py`, not left to the prompt. |
| Normalisation: uppercase, no whitespace, no `- . · _` | `normalize.py` is a copy of the grader's rule; voting compares answers the same way. |
| VRAM must stay in 1-48 GiB | The backend caps the process at `ROADGLYPH_VRAM_CAP_GIB` (default 44). |
| Harder, hidden images: noise, blur, glare, low light, angles | Image statistics pick enhancement variants (lift, de-glare, denoise, sharpen, contrast, zoom); the model gets several independent looks and votes. |
| PNG / JPEG / TIFF of any size | 16-bit and multi-frame TIFF, alpha, palette, CMYK, EXIF rotation, huge and tiny images are all handled and tested. |

## Layout

```
app/
  app.py                  harness entry point  (python3 /app/app.py --input-image ...)
  requirements.txt
  roadglyph/              the package: pipeline, backends, server, client, postprocess ...
scripts/
  smoke_test_model.py     RUN THIS FIRST on the AMD notebook
  evaluate.py             score on a folder of images, like the grader
  make_synthetic_dataset.py
  download_model.py       used by the Dockerfile
  check_submission.sh     pre-submission gates against the built image
tests/                    63 tests, no GPU needed
Dockerfile  .dockerignore  .gitignore  docker-compose.yml  Makefile  pyproject.toml
docs/SUBMISSION.md        ready-to-paste text for the lablab.ai form
```

## Workflow (what to do, in order)

1. **Smoke-test on the AMD notebook** (`notebooks.amd.com/hackathon`, 3 GPU-hours per day):
   ```bash
   pip install -r app/requirements.txt
   python scripts/smoke_test_model.py          # GPU? ROCm torch? model loads? reads a plate?
   ```
   This answers the three risky questions before any Docker work. Turn the session off afterwards
   ("Turn-off Session") because an idle session still burns quota.
2. **Score on images** (the real samples from the brief, plus synthetic ones):
   ```bash
   python scripts/make_synthetic_dataset.py --out samples/synthetic
   python scripts/evaluate.py --images samples/synthetic
   ```
   For the ten brief samples, make a `labels.csv` with `filename,expected` and pass `--images`/`--labels`.
3. **Build the image on a machine that has Docker** (see *Docker* below).
4. **Run the gates**: `./scripts/check_submission.sh roadglyph:local`
5. **Push to a public registry** and submit the image reference on lablab.ai (`docs/SUBMISSION.md`).

## Docker

The image must be built `FROM` the mandated base image and must **not** be squashed.

```bash
docker build -t roadglyph:local .
# reproducible weights: docker build --build-arg MODEL_REVISION=<commit-sha> -t roadglyph:local .

docker run -d --name rg --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --security-opt seccomp=unconfined roadglyph:local

docker cp image_01.png rg:/app/input/
docker exec rg python3 /app/app.py --input-image /app/input/image_01.png
docker exec rg cat /app/output/image_01_output.json
```

Where to build: the AMD notebook is a Jupyter pod and probably has no Docker daemon; an
**AMD Developer Cloud GPU droplet** has Docker pre-installed (other participants report this) and is
the safe choice. Building there also avoids pushing a ~40 GB image over a home connection.

Publish:

```bash
docker tag roadglyph:local ghcr.io/<you>/<name>:v1     # or Docker Hub
docker push ghcr.io/<you>/<name>:v1                    # make the package PUBLIC
# verify it is pullable without credentials:
docker logout ghcr.io && docker pull ghcr.io/<you>/<name>:v1
```

> **Do not commit the image reference to a public repository.** The brief asks participants not
> to advertise where their image lives. `.gitignore` already excludes `IMAGE_REF*`.
> Never put tokens or `.env` files in the image: it is public.

## Configuration

All optional; set as environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `ROADGLYPH_MODEL_PATH` | `/models/model` if present, else `Qwen/Qwen3-VL-8B-Instruct` | weights directory or Hugging Face id |
| `ROADGLYPH_MODE` | `auto` (image sets `server`) | `auto`, `server` or `standalone` |
| `ROADGLYPH_CLIENT_DEADLINE_S` | `26` | hard per-image deadline (the limit is 30) |
| `ROADGLYPH_MAX_VARIANTS` | `3` | maximum image variants per request |
| `ROADGLYPH_VRAM_CAP_GIB` | `44` | per-process GPU memory cap |
| `ROADGLYPH_TRACE_DIR` | unset | write a debug trace JSON per image |
| `ROADGLYPH_BACKEND` | `qwen` | `mock` runs everything without a GPU |
| `ROADGLYPH_ALLOW_CPU` | `0` | allow CPU inference for development |
| `ROADGLYPH_SOCKET` | `/tmp/roadglyph.sock` | server socket |

Choosing another model: set `MODEL_ID` at build time, or `ROADGLYPH_MODEL_PATH` at run time. Any
`AutoModelForImageTextToText` model with a chat template should work; Qwen3-VL is the tested design.

## Development

```bash
make install-dev
make test      # 63 tests: normalisation, image loading, preprocessing, post-processing,
               # pipeline voting/deadlines, the output contract, the server/client protocol
make lint
```

Everything except the real model is covered by tests; `ROADGLYPH_BACKEND=mock` exercises the
whole container contract on a laptop.

## Known risks (verify these first)

This code was developed and unit-tested **without access to an AMD GPU, ROCm or Python 3.14**, so
these are the places reality may differ:

1. **`torchvision` missing.** The base image ships only `torch` and `PIL`. Recent `transformers`
   Qwen3-VL processors may import `torchvision`. If `smoke_test_model.py` fails with a torchvision
   error, install the ROCm build that matches your torch *with `--no-deps`* (never the PyPI CUDA
   wheel), and bake the same into the image with `--build-arg TORCHVISION_SPEC="..."`.
2. **Python 3.14 wheels.** If `pip install -r app/requirements.txt` cannot find a wheel for some
   package, relax or change the pin in `requirements.txt`.
3. **No network at grading?** The challenge PDF says the container has outbound network access;
   the Match 3 reports say it does not. RoadGlyph never needs the network at run time: weights are in
   the image and `HF_HUB_OFFLINE=1`.
4. **Server start-up race.** The harness might run the first image before the model has finished
   loading. The client waits for the server until its deadline and otherwise returns an empty
   (valid) answer rather than timing out. Keep load time small (smaller model, local disk).
5. **Heuristics for stop/speed signs.** `STOP` is returned for any stop sign, and `SPEED LIMIT` is
   added if the model returned only the number. Check these against the real samples.
6. **Official self-checks can pass without a GPU.** Always score real images on a real GPU.

## How the answer is decided

1. `imaging.py` converts the file to upright 8-bit RGB.
2. `preprocess.py` measures brightness, glare, noise (Immerkaer estimate on a native-resolution
   crop) and sharpness, then proposes variants, always starting with the plain image.
3. The model classifies the object; a category prompt transcribes each variant.
4. `postprocess.py` enforces the convention: strips state names/slogans from plates, keeps the
   Chinese province character, returns `STOP`, keeps `SPEED LIMIT 65`, digits only for plaques,
   multi-line text joined top to bottom.
5. Variants vote using the grader's normalisation; two agreeing answers end the run early, and no
   further model call starts unless it fits in the remaining time.
6. The answer is written atomically; the confidence reflects how many variants agreed.

## License

MIT. See `LICENSE`.
