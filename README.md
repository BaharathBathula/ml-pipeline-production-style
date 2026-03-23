# Machine Learning Pipeline (Production Style)

A production-style end-to-end machine learning pipeline built with Python, scikit-learn, FastAPI, MLflow, Docker, and GitHub Actions.

## Overview

This project demonstrates how to design a modular ML pipeline for real-world production use cases. It includes:

- Data ingestion
- Data validation
- Data transformation
- Model training
- Model evaluation
- Model packaging
- Real-time inference API
- Docker support
- CI/CD workflows
- Unit tests

## Use Case

Customer Churn Prediction

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- FastAPI
- Uvicorn
- MLflow
- Docker
- GitHub Actions
- Pytest
- Joblib
- PyYAML

## Architecture

```mermaid
flowchart LR
    A[Raw Data] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Data Transformation]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Model Artifact Storage]
    G --> H[FastAPI Inference Service]
    H --> I[Prediction Response]
