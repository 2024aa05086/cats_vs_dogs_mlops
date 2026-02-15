# Cats vs Dogs – End-to-End MLOps Pipeline (Assignment 2)

This repository is a **complete, reproducible MLOps codebase** for **binary image classification (Cats vs Dogs)** with:

- **Git + DVC** for code/data versioning
- **Model training** (baseline CNN + optional transfer learning)
- **MLflow** experiment tracking (params, metrics, artifacts)
- **FastAPI** inference service (health + predict)
- **Docker** containerization
- **Pytest** unit tests
- **CI/CD with GitHub Actions** (tests → build → push image → deploy → smoke tests)
- **Monitoring** (request count + latency via Prometheus metrics endpoint, plus structured logs)

> Dataset: Kaggle "Dogs vs Cats" (or any Kaggle Cats vs Dogs dataset with images).  
> Input preprocessing: **224×224 RGB**, split **80/10/10**, augmentation enabled.

---

## 1) Quickstart (local, no Docker)

### Prereqs
- Python **3.10+**
- Git
- DVC
- (Optional) Kaggle API credentials for dataset download

### Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

### Download + preprocess data (DVC)
1) Put Kaggle token at `~/.kaggle/kaggle.json` (standard Kaggle API setup)
2) Run pipeline:
```bash
dvc repro
```

Artifacts produced:
- `data/raw/` (downloaded)
- `data/processed/` (224x224 RGB + splits)
- `models/model.keras` (trained model)
- `reports/` (confusion matrix, curves)

### Run MLflow UI
```bash
mlflow ui --backend-store-uri mlruns
```
Open: http://127.0.0.1:5000

### Run inference service (local)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Health:
```bash
curl -s http://127.0.0.1:8000/health
```

Predict:
```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@path/to/cat_or_dog.jpg"
```

Prometheus metrics:
```bash
curl -s http://127.0.0.1:8000/metrics | head
```

---

## 2) Docker (Inference)

### Build
```bash
docker build -t cats-dogs:latest .
```

### Run
```bash
docker run --rm -p 8000:8000 cats-dogs:latest
```

---

## 3) Deployment (Docker Compose)

### Run compose
```bash
docker compose up -d --build
```

Smoke test:
```bash
python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --image tests/assets/sample_dog.jpg
```

---

## 4) Kubernetes (optional)

Manifests are under `deploy/k8s/`.
Example (minikube):
```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

---

## 5) CI/CD (GitHub Actions)

Workflows:
- `.github/workflows/ci.yml`:
  - installs deps
  - runs pytest
  - builds Docker image
  - pushes to **GitHub Container Registry** (GHCR) on `main`
- `.github/workflows/cd.yml`:
  - deploys via Docker Compose on a self-hosted runner (recommended) OR runs a Kind-based deploy smoke test

### Required Secrets (for GHCR push)
- `GHCR_TOKEN` (PAT with `write:packages`, `read:packages`)
- `GHCR_USERNAME` (your GitHub username)

---

## 6) What to submit (per assignment)

1) **Zip file** of this repo including: source, configs, CI/CD, Docker, manifests, and model artifacts (or link if large)
2) **< 5 min screen recording** demo:
   - `git push`
   - CI runs tests + builds image + pushes to registry
   - CD deploys updated service
   - run one prediction and show output

---

## Repo layout

```
cats-vs-dogs-mlops/
  src/
    data/            # download + preprocess
    models/          # training + evaluation utilities
    api/             # FastAPI inference app
  scripts/           # smoke tests, offline perf checks
  tests/             # pytest tests + sample images
  deploy/            # docker-compose + k8s manifests
  dvc.yaml           # pipeline stages
  params.yaml        # hyperparameters + paths
  requirements.txt
  Dockerfile
```

---

## Notes
- You can switch between `baseline_cnn` and `mobilenetv2_transfer` models using `params.yaml`.
- For reproducibility, key libraries are pinned in `requirements.txt`.
