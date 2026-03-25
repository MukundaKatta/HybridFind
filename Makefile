.PHONY: install dev test lint fmt clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]" || pip install -e .
	pip install pytest ruff

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
