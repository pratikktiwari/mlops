# MLOps Pipeline: AG News Classification

## Overview

This project implements an end-to-end MLOps pipeline for news article classification using DistilBERT. The pipeline covers data preprocessing, model training, experiment tracking, model deployment, containerization, and automated CI/CD workflows.

The model classifies news articles into four categories:

- World
- Sports
- Business
- Sci/Tech

## Links

| Resource | Link |
|-----------|------|
| GitHub Repository | https://github.com/pratikktiwari/mlops |
| Hugging Face Model | https://huggingface.co/mehtayash12345678/mlops-ag_news_classification-distilbert |
| W&B Project | https://wandb.ai/g25ait2133-indian-institute-technology-jodhpur/mlops-ag_news_classification-distilbert |
| Kaggle EDA Notebook | https://www.kaggle.com/code/rahulsolankijodhpur/eda-and-data-cleaning |
| Kaggle Training Notebook | https://www.kaggle.com/code/datadrivenyash/ag-news-kaggle-training |
| DockerHub Image | https://hub.docker.com/r/sai5979/mlops-group-project |
| GitHub Inference Workflow | https://github.com/pratikktiwari/mlops/actions/workflows/inference.yml |
| GitHub Linting Workflow | https://github.com/pratikktiwari/mlops/actions/workflows/ci.yml |
| Docker build Workflow | https://github.com/pratikktiwari/mlops/blob/main/.github/workflows/docker-build.yml |

## Dataset

**Dataset:** AG News Classification

- Original Training Samples: 120,000
- Original Test Samples: 7,600
- Classes: 4
- Balanced subset used for training and experimentation
- Combined article title and description used as model input

### Data Preprocessing

The following preprocessing steps were applied:

- Combined title and description into a single text field
- Removed URLs
- Cleaned HTML encoding artifacts
- Removed duplicate records
- Removed null values
- Normalized whitespace
- Converted labels to zero-based indexing
- Applied balanced class sampling

## Model Selection

### Chosen Model

`distilbert-base-uncased`

### Why DistilBERT?

- Smaller than BERT while maintaining strong performance
- Faster inference
- Suitable for deployment environments with size constraints
- Good balance between accuracy and efficiency

## Experiments

Three model versions were trained and compared.

| Version | Learning Rate | Batch Size | Epochs |
|----------|---------------|------------|---------|
| v1 | 2e-5 | 32 | 3 |
| v2 | 2e-5 | 32 | 5 |
| v3 | 3e-5 | 32 | 3 |

### Best Model

Version **v3** achieved the best overall performance and was selected for deployment.

| Metric | v1 | v2 | v3 |
|----------|----------|----------|----------|
| Accuracy | 0.92684 | 0.92605 | 0.92803 |
| F1 Score (Weighted) | 0.92677 | 0.92617 | 0.92800 |
| Evaluation Loss | 0.44230 | 0.47405 | 0.43121 |

## Experiment Tracking

All experiments were tracked using Weights & Biases (W&B).

Tracked metrics include:

- Training Loss
- Validation Loss
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Reports

## Model Deployment

The best-performing model is published to Hugging Face Hub.

Features:

- Automated model upload after training
- Version-controlled deployment
- Reusable inference endpoint

## Inference Pipeline

A GitHub Actions workflow enables manual inference.

### Workflow

1. Enter input news text.
2. Trigger workflow manually.
3. Download trained model.
4. Run prediction.
5. Log inference metrics to W&B.
6. Display predicted class and confidence score.

## Docker Containerization

The project is fully containerized using Docker.

### Pull Docker Image

```bash
docker pull sai5979/mlops-group-project:534a4915bfb793097185dd5fbba9c9a8a651b6b9
```

### Required Environment Variables

```env
INPUT_TEXT=India A vs Afghanistan A Live Score, Tri Series 2026
WANDB_API_KEY=<your_wandb_api_key>
```

## CI/CD Pipeline

GitHub Actions automates the project workflow.

### Continuous Integration

- Code checkout
- Environment setup
- Dependency installation
- Linting and validation
- Build verification

### Docker Build Pipeline

- Build Docker image
- Authenticate with Docker Hub
- Push image automatically
- Maintain versioned releases

## Project Structure

```text
.
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── inference.py
│   └── utils/
├── notebooks/
├── .github/
│   └── workflows/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Hub
- Weights & Biases (W&B)
- Docker
- GitHub Actions
- Kaggle

## Key Features

- Automated data preprocessing
- Reproducible training pipeline
- Experiment tracking
- Model versioning
- Automated deployment
- Containerized inference
- CI/CD automation
- Monitoring through W&B

## Future Improvements

- Hyperparameter optimization
- Automated retraining pipeline
- Model monitoring dashboard
- Drift detection
- REST API deployment
- Kubernetes-based scaling

## Conclusion

This project demonstrates a complete MLOps workflow for text classification, covering the full machine learning lifecycle from data preparation and experimentation to deployment, monitoring, and automated delivery.
