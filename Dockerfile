# Use official Python 3.12 image
FROM python:3.12.3-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (OpenCV + YOLO need these)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command
ENTRYPOINT ["python", "./src/produce/produce.py"]
