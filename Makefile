.PHONY: lint test format

lint:
	ruff check .

format:
	ruff format .

test:
	pytest

check: lint test
