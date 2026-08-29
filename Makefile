.PHONY: install dev ingest evaluate run test lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

ingest:
	python scripts/ingest_data.py

evaluate:
	python scripts/run_evaluation.py

benchmark:
	python scripts/benchmark_retrievers.py

run:
	streamlit run src/web/app.py

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/

clean:
	rm -rf data/chroma_db/*
	rm -rf data/processed/*
	rm -rf data/raw/*
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
