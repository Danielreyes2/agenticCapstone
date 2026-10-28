# Always invoke tools through the active interpreter, so mypy and ruff see the
# same site-packages pytest does. A bare `mypy` can silently miss pydantic and
# report two dozen phantom errors.
PY ?= python

.PHONY: check test lint format types install

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check ato tests viz $(wildcard scripts)
	$(PY) -m ruff format --check ato tests viz $(wildcard scripts)

format:
	$(PY) -m ruff check --fix ato tests viz $(wildcard scripts)
	$(PY) -m ruff format ato tests viz $(wildcard scripts)

types:
	$(PY) -m mypy ato

check: lint types test
