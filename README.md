# Cats vs Dogs MLOps Project

This repository contains a complete, end-to-end Machine Learning Operations (MLOps) pipeline for classifying images as either cats or dogs. 

The goal of this project is to provide a clean, reproducible, and production-ready setup for training, tracking, and deploying a machine learning model. We use open-source tools to handle data, build the model, and serve it via an API and a web user interface.

##  Project Structure and Folders
Here is a breakdown of the key folders in this project and what they do:

- **`src/`**: The core source code of the project.
  - **`src/data/`**: Scripts for downloading the dataset from Kaggle and preprocessing the images.
  - **`src/models/`**: Code for building, training, and evaluating the machine learning models.
  - **`src/api/`**: The FastAPI application that serves our trained model as a REST API.
  - **`src/ui/`**: The Gradio web interface that allows users to easily test the model by uploading images.
- **`deploy/`**: Configuration files for deployment. Contains `docker-compose.yml` for running all services together locally, and `k8s/` for Kubernetes deployment manifests.
- **`tests/`**: Automated tests using Pytest to ensure everything works correctly before deploying.
- **`.github/workflows/`**: Continuous Integration and Continuous Deployment (CI/CD) pipelines managed by GitHub Actions (e.g. testing and building automatically when code changes).
- **`monitoring/`**: Configuration files for Prometheus to collect metrics and monitor the application's health.
- **`scripts/`**: Useful utilities like smoke tests to verify the deployed application.

## Key Components Explained

### 1. Data & DVC (Data Version Control)
We use data from Kaggle's "Dogs vs Cats" dataset. Managing large datasets and machine learning models can be difficult with Git alone. Therefore, we use **DVC (Data Version Control)** to version our data, track our machine learning experiments, and define our data pipeline (`dvc.yaml`). This ensures that anyone can identically reproduce the data download, preprocessing, and model training steps.

### 2. Model & Training
The project trains a Convolutional Neural Network (CNN) to distinguish between cats and dogs. We use **MLflow** for experiment tracking. This means every time we train the model, MLflow records the parameters used, the accuracy metrics achieved, and saves the final model artifacts so we can easily compare results.

### 3. API
The model is served using **FastAPI**, a modern and fast web framework for building APIs in Python. It provides endpoints like `/predict` to send an image and receive a prediction, and `/health` to check if the API is running smoothly.

### 4. UI
We included a User Interface built with **Gradio**. This gives you a simple web page where you can upload an image of a cat or a dog from your computer, and the underlying API will tell you what it is in real-time.

### 5. Docker & Docker Compose
To ensure the application runs consistently on any computer without environment issues, we use **Docker**. It packages the API and UI into isolated containers. **Docker Compose** lets us run the API, UI, and even monitoring tools (like Prometheus and Grafana) all together with a single command.

### 6. CI/CD (Continuous Integration & Delivery)
We use **GitHub Actions** to automate our testing, building, and deployment processes. 
- **Continuous Integration (CI)**: Whenever new code is pushed, the CI pipeline automatically runs tests to ensure nothing is broken. If the tests pass, it builds a new Docker image and pushes it to a container registry.
- **Continuous Deployment (CD)**: The CD pipeline safely deploys the newly built application to the target environment.

##  How to Run Locally

Below are the options to run this project on local machine.

### Option A: Running with Docker Compose
This is the easiest way to launch the API, the UI, and monitoring tools all at once.
1. Make sure you have [Docker](https://www.docker.com/) installed on your machine.
2. Open your terminal in the project root folder.
3. Run the following command:
   ```bash
   docker compose -f deploy/docker-compose.yml up -d
   ```
4. **Access the UI**: Open your web browser and go to `http://localhost:7860`.
5.**Access API Metrices**: Open your web browser and go to `http://localhost:8000/metrics`.
6.**Access Prometheus**: Open your web browser and go to `http://localhost:9090`.
7. **Access Grafana (monitoring)**: Go to `http://localhost:3000` (Default login is admin/admin).

### Option B: Running purely Local (No Docker)
1. **Setup Environment**: Create and activate a Python virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-ui.txt
   ```
3. **Run the API**:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
4. **Run the UI** (in a new terminal):
   ```bash
   # Make sure your virtual environment is activated
   export API_URL=http://localhost:8000/predict  # On Windows CMD use: set API_URL=http://localhost:8000/predict
   python src/ui/ui_gradio.py
   ```
   *The UI will be available at `http://localhost:7860`.*

### Reproducing Data and Training Locally
To reproduce the data fetching and model training:
1. Ensure your Kaggle API token (`kaggle.json`) is set up in your user directory.
2. Run `dvc repro` to execute the data pipeline and train the model.
3. Run `mlflow ui --backend-store-uri mlruns` to view the training metrics at `http://localhost:5000`.

##  Conclusion
This repository demonstrates a fully structured, production-ready MLOps workflow. It bridges the gap between raw data and a deployed application by incorporating industry-standard tools for versioning (Git/DVC), tracking (MLflow), containerization (Docker), API creation (FastAPI), web interfaces (Gradio), and CI/CD (GitHub Actions). We hope this serves as a great learning resource and a solid foundation for your own machine learning projects!
