.PHONY: run lint format typecheck check test

run:
	poetry run uvicorn ocr_service.main:app --host 0.0.0.0 --port 8000 --reload

lint:
	poetry run ruff check .

format:
	poetry run black .
	poetry run ruff check . --fix

typecheck:
	poetry run mypy .

check: lint typecheck

test:
	poetry run pytest -q
