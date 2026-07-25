.PHONY: install lint test prepare simulate serve

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ scripts/
	mypy src/datathon_offerexp/ --ignore-missing-imports
	bandit -r src/datathon_offerexp/ -c pyproject.toml

test:
	pytest src/tests/ --cov=src/datathon_offerexp --cov-fail-under=60

prepare:
	python scripts/prepare_data.py

simulate:
	python scripts/run_simulation.py

serve:
	uvicorn src.datathon_offerexp.app:app --host 0.0.0.0 --port 8000 --reload

all: install prepare simulate test
