# Use lightweight Python image
FROM python:3.11-slim

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# HF_MODEL_NAME build arg is currently unused by src/inference.py (MODEL_ID is hard-coded).
# If you want the model to be configurable, update src/inference.py to read it from env and reintroduce this ARG/ENV.

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
