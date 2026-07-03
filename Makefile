.PHONY: install test lint check docker-build docker-test docker-shell docker-review

install:
	python3.11 -m venv .venv
	.venv/bin/pip install -e '.[dev,otel]'

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

check: lint test

docker-build:
	docker compose build

docker-test:
	docker compose run --rm test

docker-shell:
	docker compose run --rm dev

docker-review:
	docker compose run --rm review
