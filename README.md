# Cats vs Dogs MLOps Project

An end-to-end MLOps project for binary image classification of cats and dogs using a CNN model, FastAPI, Docker, DVC, MLflow, GitHub Actions, and Prometheus.

## Project Overview

This project covers the machine learning lifecycle from dataset versioning and model training through API deployment, CI/CD, monitoring, and post-deployment evaluation.

### Technologies

- TensorFlow / Keras — CNN model
- FastAPI — inference API
- Docker / Docker Compose — containerization
- DVC — dataset versioning
- MLflow — experiment and model tracking
- GitHub Actions — CI/CD
- Prometheus — monitoring
- Pytest — testing

## Project Structure

```text
cats-dogs-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── config/
├── models/
│   ├── baseline_cnn.keras
│   └── baseline_cnn.h5
├── reports/
│   ├── confusion_matrix.png
│   ├── training_validation_accuracy.png
│   ├── training_validation_loss.png
│   └── post_deployment_predictions.csv
├── scripts/
│   └── evaluate_deployed_model.py
├── src/
├── tests/
├── data/
│   └── raw/PetImages.dvc
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
├── pytest.ini
├── mlflow.db
└── README.md