# Atalhos de desenvolvimento e reprodução.
.PHONY: help install data train test lint format experiments validate reproduce clean

help:
	@echo "install     - instala o pacote + deps de dev (editable)"
	@echo "data        - coleta 250k candles da Binance -> SQLite"
	@echo "test        - roda a suite de testes com cobertura"
	@echo "lint        - ruff check"
	@echo "format      - ruff format + --fix"
	@echo "experiments - roda backtest, estudo de horizonte e de alvos"
	@echo "validate    - CV purgada + Deflated Sharpe + baselines"
	@echo "reproduce   - pipeline completo end-to-end (data -> reports)"

install:
	pip install -e ".[dev]"
	pre-commit install

data:
	python main.py collect --backfill 250000

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests && ruff check --fix src tests

experiments:
	python main.py backtest
	python main.py experiment
	python main.py targets
	python main.py vol-economics

validate:
	python main.py validate

reproduce: data experiments validate
	@echo "Pipeline completo concluído. Veja reports/."

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
