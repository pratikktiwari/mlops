# Use lightweight Python image
FROM python:3.11-slim

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Hugging Face model argument
ARG HF_MODEL_NAME=mlops-ag_news_classification-distilbert

# Make model accessible inside container
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# Copy dependency file
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src

# id2label.json is currently unused by src/inference.py; omit it from the image to reduce clutter.

# Default command to run inference
CMD ["python", "src/inference.py"]
