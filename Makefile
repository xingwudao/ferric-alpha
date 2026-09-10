VENV := .venv
PYTHON := $(VENV)/bin/python
MATURIN := $(VENV)/bin/maturin
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
SETUP_STAMP := $(VENV)/.setup-stamp

.PHONY: setup develop format lint test verify golden-check visual-check docs-site-check

setup: $(SETUP_STAMP)

$(SETUP_STAMP): pyproject.toml
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install "maturin>=1.15,<2" packaging pytest ruff
	touch $(SETUP_STAMP)

develop: setup
	env -u CONDA_PREFIX VIRTUAL_ENV="$(CURDIR)/$(VENV)" \
		PATH="$(CURDIR)/$(VENV)/bin:$(PATH)" maturin develop

format: setup
	cargo fmt --all --check
	$(RUFF) format --check python tests/python scripts

lint: setup
	cargo clippy --workspace --all-targets -- -D warnings
	cargo clippy -p ferric-alpha-py --all-targets \
		--features extension-module -- -D warnings
	$(RUFF) check python tests/python scripts

test: develop
	cargo test --workspace
	$(PYTEST) -q tests/python

verify: format lint test

golden-check: develop
	$(PYTHON) scripts/generate_render_golden.py --check

visual-check: setup
	$(PYTHON) -m pip install "playwright==1.55.0"
	$(PYTHON) -m playwright install chromium
	FERRIC_ALPHA_BROWSER_TESTS=1 $(PYTEST) -q tests/python/test_render_browser.py

docs-site-check:
	cd docs-site && npm install --no-audit && npm run docs:build
