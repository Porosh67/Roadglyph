.RECIPEPREFIX := >
.PHONY: help install-dev lint format test samples build check clean

IMAGE ?= roadglyph:local

help:
> @echo "make install-dev  install ruff + pytest (optional: tests also run with unittest)"
> @echo "make lint         ruff check + format check"
> @echo "make test         run the unit tests (no GPU needed)"
> @echo "make samples      generate synthetic PNG/JPEG/TIFF test images in samples/synthetic"
> @echo "make build        docker build -t $(IMAGE) ."
> @echo "make check        pre-submission checks against a built image (needs docker + AMD GPU)"

install-dev:
> python3 -m pip install ruff pytest Pillow numpy

lint:
> ruff check .
> ruff format --check .

format:
> ruff check --fix .
> ruff format .

test:
> python3 -m unittest discover -s tests -v

samples:
> python3 scripts/make_synthetic_dataset.py --out samples/synthetic --count 15

build:
> docker build -t $(IMAGE) .

check:
> ./scripts/check_submission.sh $(IMAGE)

clean:
> rm -rf output .pytest_cache .ruff_cache
> find . -name __pycache__ -type d -prune -exec rm -rf {} +
