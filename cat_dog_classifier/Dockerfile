# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies required for Pillow and numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install only necessary Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy application code and model
COPY app/ ./app/
COPY models/ ./models/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
