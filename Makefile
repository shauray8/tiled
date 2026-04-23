VENV := .venv
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

.PHONY: install test test-fast bench clean

install:
	uv venv $(VENV) --python 3.12 --system-site-packages
	uv pip install -e ".[dev]" --python $(PYTHON)

test:
	$(PYTEST) tests/ -v

test-fast:
	$(PYTEST) tests/ -v -m "not slow" -n auto

bench:
	$(PYTHON) benchmarks/run_all.sh

clean:
	rm -rf $(VENV) dist build *.egg-info .pytest_cache __pycache__
