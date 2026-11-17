.PHONY: pipeline pipeline-local docker-pipeline dashboard lint test check

pipeline: docker-pipeline

pipeline-local:
	uv run python -m src.pipeline

docker-pipeline:
	docker compose up -d --wait mlflow && docker compose --profile pipeline run --rm --build pipeline

dashboard:
	uv run streamlit run src/app.py

lint:
	uv run ruff check .

test:
	uv run pytest -v

check: lint test
