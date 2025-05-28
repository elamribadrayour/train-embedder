run:
	uv run src/job/main.py

format:
	uv run ruff format src

lint:
	uv run ruff check src

type:
	uv run mypy src/job
