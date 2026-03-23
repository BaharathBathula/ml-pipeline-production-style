# Machine Learning Pipeline (Production Style)

A production-style end-to-end machine learning pipeline project built with Python, FastAPI, scikit-learn, MLflow, Docker, and GitHub Actions.

## Overview

This project demonstrates how to design and deploy a modular machine learning pipeline suitable for real-world environments. It covers the full lifecycle:

- Data ingestion
- Data validation
- Data transformation
- Model training
- Model evaluation
- Model packaging
- Inference API deployment
- CI/CD automation

## Tech Stack

- Python
- scikit-learn
- pandas
- numpy
- FastAPI
- Uvicorn
- MLflow
- Docker
- GitHub Actions
- PyYAML
- pytest

## Project Architecture

```text
Raw Data -> Validation -> Transformation -> Training -> Evaluation -> Model Artifact -> FastAPI Inference
