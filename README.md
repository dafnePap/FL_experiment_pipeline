# Federated Learning Experiment Platform

A full-stack application developed as an extension of my diploma thesis at the Technical University of Crete. My thesis focused on comparing Federated Learning algorithms, including FedAvg, FedProx, SCAFFOLD, FedAdam, and FDA-OPT. Building on this work, I developed a web-based platform that allows users to submit experiments, compare algorithms, and visualize their results.

### [Live Demo](https://fl-experiment-pipeline.vercel.app/) | [GitHub Repository](https://github.com/dafnePap/FL_experiment_pipeline)

The public demo displays results from one completed experiment. Running new experiments requires the local application.

## Features

- Five federated learning algorithms: FedAvg, FedProx, SCAFFOLD, FedAdam, and FDA-OPT, implemented in PyTorch.
- Adjustable data heterogeneity: control how non-IID the client data is with `s_iid` (0 = sorted by label, 1 = fully IID).
- Docker and API: run experiments in Docker containers, and start them and check their status through a FastAPI REST API.
- Result plots: compare accuracy, loss, and communication cost across algorithms.
- Automated CI/CD: GitHub Actions runs linting, unit tests, and a smoke test on every push, then publishes the Docker image to GitHub Container Registry.

## How It Works

1. The user submits an experiment through React interface.
2. FastAPI validates the experiment parameters and manages its execution.
3. Docker prepares the dataset and runs the selected algorithms.
4. Results are saved, plots are generated, and both are displayed in the dashboard.

All algorithms within a comparison use the same client data partitions.

## Running Locally

Requirements: Docker, Docker Compose, Python, Node.js, and npm.

Clone the repository:

```bash
git clone https://github.com/dafnePap/FL_experiment_pipeline.git
cd FL_experiment_pipeline
```

Build the Docker image:

```bash
docker compose build
```

Start the backend after installing its Python dependencies:

```bash
uvicorn api.main:app --reload --port 8000
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Datasets and Models

- Datasets: MNIST; EMNIST and CIFAR-10 
- Models: Logistic Regression, LeNet-5, CNN, VGG16, ResNet18.

Model compatibility depends on the selected dataset.

## Public Demo

The [Vercel demo](https://fl-experiment-pipeline.vercel.app/) uses saved experiment parameters, results, and plots.

It does not execute new training jobs. The complete training pipeline is available when running the project locally.
