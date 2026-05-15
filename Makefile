# Executive Evasion Index — developer Makefile
# Use a recent GNU Make on Windows via Git Bash, WSL, or chocolatey.

PY ?= python
PIP ?= $(PY) -m pip
PYTEST ?= $(PY) -m pytest

.PHONY: install test test-fast coverage coverage-open lint format clean run-pipeline dashboard docker-build docker-run

install:
	$(PIP) install -U -r requirements.txt

test:
	$(PYTEST) --cov=src --cov=config --cov-report=term-missing --cov-report=html

test-fast:
	$(PYTEST) -m "not integration" --cov=src --cov=config --cov-report=term-missing

coverage:
	$(PYTEST) --cov=src --cov=config --cov-report=html
	@echo "HTML report at htmlcov/index.html"

coverage-open: coverage
	@$(PY) -c "import webbrowser, pathlib; webbrowser.open('file://' + str(pathlib.Path('htmlcov/index.html').resolve()))"

lint:
	$(PY) -m black --check src/ config.py tests/
	$(PY) -m flake8 src/ config.py tests/ --max-line-length=110 --ignore=E203,W503,E501

format:
	$(PY) -m isort src/ config.py tests/
	$(PY) -m black src/ config.py tests/

run-pipeline:
	$(PY) src/1_scraper.py --synthetic-only
	$(PY) src/2_parser.py
	$(PY) src/3_evasion_scorer.py --mode heuristic
	$(PY) src/4_backtester.py

dashboard:
	$(PY) -m streamlit run src/5_dashboard.py

clean:
	@$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in (pathlib.Path('htmlcov'), pathlib.Path('.pytest_cache'), pathlib.Path('.coverage'))]"
	@echo "cleaned cache + coverage artifacts"

docker-build:
	docker build -t eei-dashboard:latest .

docker-run:
	docker run --rm -p 8501:8501 --env-file .env -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs eei-dashboard:latest
