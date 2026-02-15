.PHONY: venv install test dvc train api docker compose smoke

venv:
	python -m venv .venv

install:
	pip install -r requirements.txt

test:
	pytest -q

dvc:
	dvc repro

train:
	python -m src.models.train --params params.yaml

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

docker:
	docker build -t cats-dogs:latest .

compose:
	docker compose -f deploy/docker-compose.yml up -d --build

smoke:
	python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --image tests/assets/sample_dog.jpg
